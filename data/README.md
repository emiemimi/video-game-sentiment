# Reddit Gaming Community Subset

Source dataset: `gk4u/reddit_dataset_218` on Hugging Face.

This subset was streamed from Hugging Face and filtered to gaming-related Reddit communities for the video game opinion classifier project.

## Files

- `reddit_gaming_clean.csv`: cleaned CSV dataset with 50,000 rows.
- `reddit_gaming_clean.parquet`: the same data in Parquet format.

## Columns

- `text`: Reddit post/comment text.
- `subreddit`: normalized subreddit name without the `r/` prefix.
- `data_type`: source row type, such as `comment`.
- `datetime`: source date from the original dataset.
- `source_dataset`: Hugging Face source dataset id.

## Filtering

Target communities:

- `gaming`
- `videogames`
- `pcgaming`
- `Steam`
- `NintendoSwitch`
- `PS5`
- `XboxSeriesX`
- `truegaming`
- `Games`
- `gamingnews`

Cleaning steps:

- Kept only rows whose `communityName` matched the target gaming communities.
- Removed exact/normalized duplicate text.
- Removed deleted or removed content.
- Removed very short rows with fewer than 5 words.
- Removed likely spam, including URL-heavy rows, common bot/moderation phrases, repeated-word spam, and low-alphabetic-content rows.

## Generation

Regenerate with:

```bash
python3 scripts/filter_reddit_gaming.py --output data/reddit_gaming_clean.csv --max-kept 50000 --min-words 5 --progress-every 50000
```

Run summary:

- Scanned source rows: 22,514,197
- Target subreddit rows found before cap: 60,620
- Kept rows: 50,000
- Removed spam/very short rows: 10,479
- Removed duplicates: 141

