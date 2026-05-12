---
license: mit
task_categories:
- text-classification
language:
- en
pretty_name: Video Game Sentiment Balanced 10k
---

# Video Game Sentiment Balanced 10k

This dataset contains 10,000 labeled gaming-community Reddit comments for sentiment classification.

Only the balanced 10k dataset is included in this Hugging Face dataset repository.

## Columns

- `text`: Reddit comment text.
- `subreddit`: source subreddit.
- `data_type`: original row type.
- `datetime`: source date.
- `source_dataset`: original Hugging Face dataset id.
- `sentiment`: target label: `positive`, `negative`, `mixed`, or `neutral`.
- `reason`: auxiliary label from the labeling process.
- `label_notes`: notes from labeling when uncertain.
- `confidence`: LLM-assigned labeling confidence.

## Label Distribution

| Sentiment | Rows |
|---|---:|
| positive | 2,676 |
| negative | 2,545 |
| neutral | 2,500 |
| mixed | 2,279 |

## Label Quality Note

Labels were created with LLM assistance and minimal human review. Some labels may be noisy or incorrect.

