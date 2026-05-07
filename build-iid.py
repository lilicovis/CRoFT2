import os
from datasets import load_from_disk

INPUT_DATASET = "/mnt/dataset/fineweb-26/fineweb_s_26_gpt2_tokenized"
OUTPUT_ROOT = "/mnt/dataset/fineweb-26/fineweb_s_26_balanced_shards_4"
NUM_SHARDS = 4
SEED = 42

def main():
    ds = load_from_disk(INPUT_DATASET)

    # 先整体打乱，消除原始顺序偏置
    ds = ds.shuffle(seed=SEED)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for shard_id in range(NUM_SHARDS):
        shard_indices = list(range(shard_id, len(ds), NUM_SHARDS))
        shard_ds = ds.select(shard_indices)

        out_dir = os.path.join(OUTPUT_ROOT, f"shard_{shard_id:03d}")
        shard_ds.save_to_disk(out_dir)
        print(f"saved {out_dir}, size={len(shard_ds)}")

if __name__ == "__main__":
    main()