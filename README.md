# Averta

CPU-native failure prediction for AI coding agents.

> Can lightweight, CPU-native ML predict when a coding agent is heading toward
> failure early enough to save tokens — without killing sessions that would have
> recovered?

## Status

Phase 1 — trajectory ingestion. No model results yet. Every number in this
README is measured before it is written.

## Background

When a coding agent works a task, the session either resolves it or it doesn't.
Unsuccessful sessions tend to run longer and revisit approaches that have
already failed, which means the tokens spent after a certain point produce
nothing. Detecting that early is worth doing, but stopping a session that would
have recovered is worse than letting it run.

[Fail-Fast, Restart-Smart](https://arxiv.org/abs/2608.03222) (Wang et al., 2026)
approaches this with a 0.6B neural monitor over observable trajectory prefixes.
Averta asks whether engineered trajectory features and a gradient-boosted
classifier can do useful work at a fraction of that inference cost, cheaply
enough to run locally alongside an agent on CPU.

## Approach

Public agent trajectory datasets are normalized into a single schema, features
are extracted from session prefixes, and a classifier predicts the eventual
outcome from what is observable at a given turn.

Three constraints shape the design:

- **Prefix-only features.** A feature computed at turn *t* reads `turns[0:t]`
  and nothing else. Unsuccessful sessions run longer, so leaking session length
  leaks the label.
- **Absolute turn indices.** Evaluating at "40% through the session" requires
  knowing the total length, which is unavailable at inference time. Evaluation
  uses turns 5, 10, and 20.
- **Grouped splits.** Train/test splits group by repository, never by
  trajectory.

## Data

Training data comes from
[SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories),
which records OpenHands agent attempts at SWE-bench-style issues along with
whether each attempt resolved the issue.

A survey of six public trajectory corpora found this to be the only one
carrying both outcome classes. The others — `nebius/SWE-rebench-openhands`,
`nvidia/SWE-Zero`, `nvidia/SWE-Hero`, `SWE-Gym/OpenHands-SFT` — are
supervised fine-tuning corpora that store the agent's patch but no resolution
label.

Measured over all 6,055 trajectories:

| | |
|---|---|
| Label coverage | 100% |
| Resolve rate | 8.1% (491 / 6,055) |
| Distinct repositories | 11 |
| Distinct task instances | 2,438 |
| Turns per session | median 31, p75 61, max 101 |
| Per-step token counts | not recorded |

Two properties of this data shape the evaluation. The positive class is rare
at 8.1%, so metrics that tolerate imbalance are required. And **session length
barely separates the classes** — resolved sessions average 39.86 turns against
39.23 for unresolved. Prior work has described unsuccessful agent runs as
tending to be longer; that does not reproduce here, which makes turn count a
near-useless predictor on its own and raises the bar for what the remaining
features must contribute.

Because the source carries no declared license, this repository does not
redistribute it. Raw trajectories are downloaded locally and excluded from
version control; only derived statistics, figures, and model artifacts are
published.

## Evaluation

Reported metrics are AUROC, AUPRC, recall at 5% FPR, AUC by turn index,
calibration, simulated token savings, and the rate at which successful sessions
would have been incorrectly terminated. Accuracy is not reported — outcome
classes are imbalanced and it flatters trivial models. Trivial baselines
(majority class, turn index alone) are reported alongside every model.

## Development

Requires Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## License

TBD — will be selected to be compatible with the upstream dataset licenses.
