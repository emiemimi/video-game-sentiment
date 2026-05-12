# Gaming Opinion Labeling Guide

Use this guide for `reddit_gaming_labeling_sample.csv`.

## Sentiment Labels

- `positive`: The writer clearly likes, praises, recommends, or supports the game/company/topic.
- `negative`: The writer clearly dislikes, criticizes, complains, or rejects the game/company/topic.
- `mixed`: The writer includes both positive and negative judgment in the same comment.
- `neutral`: The writer is mainly asking, informing, joking, quoting, or discussing without a clear like/dislike judgment.

## Reason Labels

Choose the main reason for the sentiment. If the text has no clear opinion, use `other`.

- `gameplay`: mechanics, combat, controls, difficulty, balance, progression, level design.
- `graphics`: visuals, art style, animation, UI appearance, visual fidelity.
- `story`: plot, writing, characters, lore, dialogue, quests.
- `performance`: FPS, lag, optimization, loading, stuttering, hardware performance.
- `bugs`: crashes, glitches, broken quests, errors, missing functionality.
- `price`: cost, discounts, refunds, value for money.
- `company_reputation`: developer, publisher, platform holder, corporate behavior, trust, public controversy.
- `monetization`: microtransactions, DLC, battle passes, loot boxes, subscriptions, pay-to-win.
- `multiplayer`: online play, matchmaking, servers, co-op, PvP, player community.
- `updates_support`: patches, updates, nerfs/buffs, roadmap, support response, maintenance.
- `other`: opinion exists but none of the above is the main reason, or no clear reason is present.

## Labeling Rules

- Label the text itself, not what you personally think about the game.
- Prefer `mixed` when a comment clearly praises one thing and criticizes another.
- Use one primary reason only.
- If the comment mentions a company because of microtransactions, use `monetization`, not `company_reputation`.
- If the comment mentions crashes, broken behavior, or glitches, use `bugs`; use `performance` for speed/FPS/lag/optimization.
- If the text is too vague, purely factual, or unclear, use `neutral` and `other`.

