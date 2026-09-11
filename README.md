# Averta

CPU-native failure prediction for AI coding agents.

> Can lightweight, CPU-native ML predict when a coding agent is heading toward
> failure early enough to save tokens — without killing sessions that would have
> recovered?

## Status

First attempt complete. The pre-registered decision gate **was not met** — see
[Results](#results). Every number in this README was measured before it was
written; none are projections.

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

The positive class is **failure** — the event the tool would warn about. That
makes the false positive rate mean "sessions that would have resolved but were
flagged", which is the quantity worth constraining. Average precision is
reported on the minority class (resolution, 11.2% at the gate cut) since
average precision is informative about the rare class.

Accuracy is never reported: at an 89% failure rate a model that always predicts
failure scores 0.89 and is useless. Trivial baselines are reported alongside
every model, and cross-validation folds group by repository so no model is
tested on a codebase it trained on.

Success criteria were **pre-registered in `src/averta/thresholds.py` and
committed before any model existed**, so they could not be relaxed to fit a
disappointing result.

## Results

At the pre-registered gate — turn 10, 4,381 sessions, 88.79% failure rate,
five repository-grouped folds, out-of-fold predictions pooled, confidence
intervals resampling repositories:

| model | AUROC | 95% CI | AUPRC | R@5%FPR | p50 latency | size |
|---|---|---|---|---|---|---|
| majority | 0.500 | [0.500, 0.500] | 0.127 | 0.000 | — | — |
| turn index only | 0.500 | [0.500, 0.500] | 0.127 | 0.000 | — | — |
| **logistic** | **0.668** | [0.645, 0.689] | 0.198 | 0.081 | **0.18 ms** | **1.8 KB** |
| random forest | 0.653 | [0.620, 0.672] | 0.188 | 0.148 | 26.9 ms | 11.6 MB |
| hist gradient boosting | 0.623 | [0.589, 0.645] | 0.168 | 0.155 | 13.9 ms | 524 KB |
| xgboost | 0.648 | [0.603, 0.673] | 0.184 | 0.151 | 0.18 ms | 494 KB |

**Gate verdict: not met.** Four of five criteria passed; recall at a 5% false
positive budget reached 0.155 at best against a required 0.25. The shortfall
holds at every cut point tested (3, 5, 10, 20, 40 — best anywhere is 0.182) and
under any model-selection rule.

### What was established

**Predictive signal exists and is not marginal.** AUROC 0.668 with a
confidence interval of [0.645, 0.689] separates decisively from three
baselines pinned at 0.500–0.521.

**Signal emerges around turn 5 and peaks near turn 10.** At turn 3 every model
sits at chance — a system prompt, a task statement and one action carry no
evidence. Past turn 20 performance decays, partly through survivorship: 37% of
resolved sessions reach turn 40 against 41% of unresolved.

**A 1.8 KB linear model was the most accurate**, beating gradient boosting
(0.668 against 0.623–0.653) at 0.18 ms single-row CPU inference. Plausible
cause: with 23 features, 491 positives and repository-grouped evaluation, trees
fit repository-specific thresholds that do not survive the group boundary. For
scale, the neural monitor this work reproduces uses 0.6B parameters.

**Repeated identical tool calls carry the signal, not repeated errors.**
Permutation importance under the same grouped folds:

| feature | AUROC drop |
|---|---|
| `n_repeated_calls` | 0.198 |
| `n_edits` | 0.072 |
| `chars_recent` | 0.038 |
| `max_call_repeat` | 0.031 |
| `max_error_repeat` | 0.008 |

The prior expectation was that a recurring error signature would dominate. It
is close to worthless in isolation (AUROC 0.521 alone, 0.008 permutation
drop). What predicts failure is the agent reissuing a tool call it has already
issued with byte-identical arguments.

**No successful session in this corpus ends before turn 10** — 491 of 491
resolved sessions reach it, against 71% of unresolved ones. Early termination
is therefore perfectly associated with failure, which is why the base rate
shifts from 8.2% to 11.2% between turn 5 and turn 10.

### Limitations

- Every feature is an aggregate count over the prefix. There is no sequence
  representation, so ordering is captured only through crude repeat counts —
  a plausible explanation for the operating-point shortfall, given the
  strongest feature is itself a repetition proxy.
- One corpus, one agent scaffold, 11 repositories. Grouped folds mean 11
  grouping units, which widens every interval.
- Token savings are not yet reported. The source records no per-message token
  counts, so any figure would be an estimate from content length and is
  withheld until it can be labelled as such.

## Development

Requires Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## License

TBD — will be selected to be compatible with the upstream dataset licenses.
