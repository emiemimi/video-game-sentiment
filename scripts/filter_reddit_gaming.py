import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


TARGET_SUBREDDITS = {
    "gaming",
    "videogames",
    "pcgaming",
    "steam",
    "nintendoswitch",
    "ps5",
    "xboxseriesx",
    "truegaming",
    "games",
    "gamingnews",
}

REMOVED_TEXT = {
    "",
    "[deleted]",
    "[removed]",
    "deleted",
    "removed",
}

SPAM_PHRASES = (
    "i am a bot",
    "automatically generated",
    "please contact the moderators",
    "this action was performed automatically",
    "subscribe to my",
    "check out my channel",
    "join my discord",
    "free nitro",
    "onlyfans",
)

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_subreddit(value: str) -> str:
    value = (value or "").strip().lower()
    if value.startswith("r/"):
        value = value[2:]
    return value


def clean_text(value: str) -> str:
    value = (value or "").replace("\x00", " ")
    return WS_RE.sub(" ", value).strip()


def dedupe_key(text: str) -> str:
    lowered = text.lower()
    lowered = URL_RE.sub(" ", lowered)
    lowered = NON_WORD_RE.sub(" ", lowered)
    return WS_RE.sub(" ", lowered).strip()


def is_probably_spam_or_low_quality(text: str, min_words: int) -> bool:
    normalized = text.strip().lower()
    if normalized in REMOVED_TEXT:
        return True

    words = re.findall(r"\w+", text)
    if len(words) < min_words:
        return True

    if any(phrase in normalized for phrase in SPAM_PHRASES):
        return True

    url_count = len(URL_RE.findall(text))
    if url_count >= 2:
        return True

    if url_count == 1 and len(words) < 12:
        return True

    alpha_chars = sum(ch.isalpha() for ch in text)
    if alpha_chars < 20:
        return True

    if len(set(normalized)) <= 4 and len(normalized) > 20:
        return True

    repeated_word_runs = re.search(r"\b(\w+)(?:\s+\1\b){3,}", normalized)
    if repeated_word_runs:
        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream gk4u/reddit_dataset_218 and save a cleaned gaming subreddit subset."
    )
    parser.add_argument("--output", default="data/reddit_gaming_clean.csv")
    parser.add_argument("--repo", default="gk4u/reddit_dataset_218")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-kept", type=int, default=50000)
    parser.add_argument("--max-scanned", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.repo, split=args.split, streaming=True)
    seen = set()
    counts = Counter()

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "text",
                "subreddit",
                "data_type",
                "datetime",
                "source_dataset",
            ],
        )
        writer.writeheader()

        for scanned, row in enumerate(ds, start=1):
            counts["scanned"] += 1
            subreddit = normalize_subreddit(row.get("communityName") or row.get("label"))
            if subreddit not in TARGET_SUBREDDITS:
                continue
            counts["target_subreddit_rows"] += 1

            text = clean_text(row.get("text", ""))
            if is_probably_spam_or_low_quality(text, args.min_words):
                counts["spam_or_short"] += 1
                continue

            key = dedupe_key(text)
            if key in seen:
                counts["duplicates"] += 1
                continue
            seen.add(key)

            writer.writerow(
                {
                    "text": text,
                    "subreddit": subreddit,
                    "data_type": row.get("dataType", ""),
                    "datetime": row.get("datetime", ""),
                    "source_dataset": args.repo,
                }
            )
            counts["kept"] += 1

            if args.progress_every and scanned % args.progress_every == 0:
                print(dict(counts), flush=True)

            if args.max_kept and counts["kept"] >= args.max_kept:
                break

            if args.max_scanned and scanned >= args.max_scanned:
                break

    print(dict(counts))
    print(f"Saved {counts['kept']} rows to {output}")


if __name__ == "__main__":
    main()
