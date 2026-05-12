import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a balanced sample from the cleaned Reddit gaming dataset for manual labeling."
    )
    parser.add_argument("--input", default="data/reddit_gaming_clean.csv")
    parser.add_argument("--output", default="data/reddit_gaming_labeling_sample.csv")
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    subreddits = sorted(df["subreddit"].dropna().unique())
    base_per_subreddit = args.sample_size // len(subreddits)
    remainder = args.sample_size % len(subreddits)

    parts = []
    for index, subreddit in enumerate(subreddits):
        target = base_per_subreddit + (1 if index < remainder else 0)
        subreddit_df = df[df["subreddit"] == subreddit]
        take = min(target, len(subreddit_df))
        parts.append(subreddit_df.sample(n=take, random_state=args.seed + index))

    sampled = pd.concat(parts, ignore_index=True)

    if len(sampled) < args.sample_size:
        remaining = df.drop(sampled.index, errors="ignore")
        needed = args.sample_size - len(sampled)
        extra = remaining.sample(n=min(needed, len(remaining)), random_state=args.seed)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled = sampled.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sampled.insert(0, "id", range(1, len(sampled) + 1))
    sampled["sentiment"] = ""
    sampled["reason"] = ""
    sampled["label_notes"] = ""

    columns = [
        "id",
        "text",
        "subreddit",
        "data_type",
        "datetime",
        "sentiment",
        "reason",
        "label_notes",
        "source_dataset",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sampled[columns].to_csv(output, index=False)

    print(f"Saved {len(sampled)} rows to {output}")
    print(sampled["subreddit"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
