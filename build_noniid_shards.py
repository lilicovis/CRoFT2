import sys

if __name__ == "__main__":
    raise SystemExit(
        "Do not build offline non-iid shards. "
        "Use full tokenized dataset and split by node at training time."
    )