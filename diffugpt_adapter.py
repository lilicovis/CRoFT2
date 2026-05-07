import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from LLaMA_Factory.attention_patch import replace_attention_mask
from LLaMA_Factory.model import DiscreteDiffusionModel
from LLaMA_Factory.trainer import transition, get_anneal_attn_mask, LinearNoise


def build_diffugpt_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str | None = None,
    trust_remote_code: bool = True,
    use_fast: bool = True,
):
    replace_attention_mask()

    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import os as _os
    _ckpt_files = sorted(_os.listdir(model_path)) if _os.path.isdir(model_path) else ["<not a dir>"]
    print(f"[CKPT DIAG] checkpoint files: {_ckpt_files}")
    print(f"[CKPT DIAG] tokenizer loaded: vocab_size={len(tokenizer)}, "
          f"mask_token={tokenizer.mask_token!r}, mask_token_id={tokenizer.mask_token_id}")

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        print(f"[CKPT DIAG] mask_token added: mask_token_id={tokenizer.mask_token_id}, "
              f"new vocab_size={len(tokenizer)}")
    else:
        print(f"[CKPT DIAG] mask_token already in tokenizer, skipping add_special_tokens")

    import os as _os
    from safetensors.torch import load_file as _load_safetensors

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    # 1. 用 from_config 构建随机初始化的架构（不加载 checkpoint 权重）
    #    因为 checkpoint 是以 DiscreteDiffusionModel 的 key 格式保存的
    #    （embed_tokens.weight / denoise_model.h.N.xxx），
    #    而 AutoModelForCausalLM.from_pretrained 期望 transformer.h.N.xxx，
    #    key 不匹配 → 所有权重被随机初始化 → CE ≈ 11（随机水平）。
    base_model = AutoModelForCausalLM.from_config(config)

    ckpt_vocab = base_model.get_input_embeddings().num_embeddings
    print(f"[CKPT DIAG] checkpoint vocab_size={ckpt_vocab}, tokenizer vocab_size={len(tokenizer)}")

    if len(tokenizer) > ckpt_vocab:
        print(f"[CKPT DIAG] ⚠️  RESIZE needed: {ckpt_vocab} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        print(f"[CKPT DIAG] ✓ no resize needed")

    # 2. 先包装成 DiscreteDiffusionModel（key 格式变为 embed_tokens.* / denoise_model.*）
    model = DiscreteDiffusionModel(base_model, config, model_args=None)

    # 3. 直接把 checkpoint 的 state_dict 加载进 DiscreteDiffusionModel
    safetensors_path = _os.path.join(model_path, "model.safetensors")
    pytorch_path = _os.path.join(model_path, "pytorch_model.bin")
    if _os.path.exists(safetensors_path):
        state_dict = _load_safetensors(safetensors_path, device="cpu")
    elif _os.path.exists(pytorch_path):
        import torch as _torch
        state_dict = _torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")


    ckpt_keys = set(state_dict.keys())
    print(f"[CKPT DIAG] checkpoint keys sample: {sorted(ckpt_keys)[:8]}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[CKPT DIAG] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[CKPT DIAG]   missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"[CKPT DIAG]   unexpected (first 5): {unexpected[:5]}")

    return model, tokenizer


def compute_diffusion_loss(
    model,
    batch,
    tokenizer,
    global_step: int,
    anneal_steps: int = 1,
    shift: bool = True,
    sampling_eps: float = 1e-3,
    return_stats: bool = False,   # 新增
    dynfuse_state=None,           # 可选：离线路由 / DynFuse 在线验证（与连续时间 t 对齐）
):
    device = next(model.parameters()).device

    x = batch["input_ids"]
    if isinstance(x, list):
        if len(x) > 0 and isinstance(x[0], torch.Tensor):
            x = torch.stack([t.to(device) for t in x], dim=0)
        else:
            x = torch.tensor(x, device=device)
    else:
        x = x.to(device)

    x = x.clone().contiguous().long()

    # 与 5-diffugpt-s 行为对齐：所有 token（包括 EOS padding）均可参与扩散和 loss 计算。
    # 5-diffugpt-s 的原始数据集含约 52% EOS padding，inner_forward 不检查 attention_mask，
    # 因此 EOS token 也会被随机 mask 并计入 CE（但 EOS 极易预测，CE ≈ 0，拉低整体指标）。
    # 保持 src_mask = all-False，使 DiLoCo 与 5-diffugpt-s 处于同一数据分布和 loss 口径。
    if "src_mask" in batch:
        src_mask = batch["src_mask"]
        if isinstance(src_mask, list):
            if len(src_mask) > 0 and isinstance(src_mask[0], torch.Tensor):
                src_mask = torch.stack([t.to(device) for t in src_mask], dim=0)
            else:
                src_mask = torch.tensor(src_mask, device=device)
        else:
            src_mask = src_mask.to(device)
        src_mask = src_mask.clone().contiguous().bool()
    else:
        # 不对 attention_mask==0 的 padding token 施加保护，与 5-diffugpt-s 对齐
        src_mask = torch.zeros_like(x, dtype=torch.bool)

    if x.dim() != 2:
        raise ValueError(f"input_ids must be [batch, seq], got {tuple(x.shape)}")
    if src_mask.shape != x.shape:
        raise ValueError(f"src_mask shape {tuple(src_mask.shape)} != input_ids shape {tuple(x.shape)}")

    if tokenizer.mask_token_id is None:
        raise ValueError("tokenizer.mask_token_id is None")

    batch_size = x.size(0)
    seq_len = x.size(1)

    noiser = LinearNoise()

    t = (1 - sampling_eps) * torch.rand(batch_size, device=x.device) + sampling_eps
    sigma = noiser.total_noise(t)
    dsigma = noiser.rate_noise(t)

    x_t = transition(
        x.clone().contiguous(),
        sigma[:, None],
        maskable_mask=~src_mask,
        mask_token_id=tokenizer.mask_token_id,
    ).clone().contiguous()

    if anneal_steps is None or anneal_steps <= 1:
        attn_mask_ratio = 1.0
    else:
        attn_mask_ratio = min(1.0, (global_step + 1) / anneal_steps)

    model_ref = model.module if hasattr(model, "module") else model
    embed_dtype = model_ref.get_input_embeddings().weight.dtype

    attention_mask = get_anneal_attn_mask(
        seq_len,
        batch_size,
        dtype=embed_dtype,
        device=x.device,
        attn_mask_ratio=attn_mask_ratio,
    )

    # DynFuse / 时间步条件 router：与 LinearNoise 采样的 t∈(eps,1) 一致，而非近似 masked_ratio
    if dynfuse_state is not None:
        dynfuse_state.set_timestep(t)

    logits = model(x_t, attention_mask=attention_mask)
    vocab_size = logits.size(-1)

    loss_mask = x_t == tokenizer.mask_token_id

    if shift:
        logits = logits[:, :-1].contiguous()
        loss_mask = loss_mask[:, 1:].contiguous()
        x = x[:, 1:].contiguous()

    masked_tokens = loss_mask.sum()
    if masked_tokens.item() == 0:
        zero = logits.sum() * 0.0
        if return_stats:
            return zero, {
                "masked_ce_without_1_over_t": zero.detach(),
                "masked_ratio": torch.tensor(0.0, device=logits.device),
                "masked_tokens": 0,
            }
        return zero

    # token-level CE
    ce = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        x.reshape(-1),
        reduction="none",
    ).float().reshape(batch_size, -1)

    # 只保留 mask 位置
    ce_masked = ce.masked_fill(~loss_mask, 0.0)

    # 你原来的 diffusion weighted loss
    final_loss = (dsigma[:, None] * ce_masked).sum() / masked_tokens

    # 新增：不带 1/t / dsigma 重加权的 masked CE
    masked_ce_without_1_over_t = ce_masked.sum() / masked_tokens

    # 新增：mask 比例
    masked_ratio = loss_mask.float().mean()

    if return_stats:
        stats = {
            "masked_ce_without_1_over_t": masked_ce_without_1_over_t.detach(),
            "masked_ratio": masked_ratio.detach(),
            "masked_tokens": int(masked_tokens.item()),
        }
        return final_loss, stats

    return final_loss