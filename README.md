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

Two attempts were made. Both are reported, and the success criteria were
**identical for each** — a second attempt judged against a relaxed bar would
prove nothing.

At the pre-registered gate — turn 10, 4,381 sessions, 88.79% failure rate,
five repository-grouped folds, out-of-fold predictions pooled, confidence
intervals resampling repositories:

| model | AUROC | 95% CI | AUPRC | R@5%FPR | p50 latency | size |
|---|---|---|---|---|---|---|
| majority | 0.500 | [0.500, 0.500] | 0.127 | 0.000 | — | — |
| turn index only | 0.500 | [0.500, 0.500] | 0.127 | 0.000 | — | — |
| **logistic** | **0.677** | [0.651, 0.697] | 0.198 | 0.191 | **0.18 ms** | **2.1 KB** |
| random forest | 0.655 | [0.626, 0.673] | 0.190 | 0.166 | 26.6 ms | 11.3 MB |
| hist gradient boosting | 0.629 | [0.591, 0.651] | 0.168 | 0.172 | 11.6 ms | 524 KB |
| xgboost | 0.650 | [0.605, 0.673] | 0.187 | 0.154 | 0.18 ms | 490 KB |

**Gate verdict: not met, in both attempts.** Four of five criteria passed;
recall at a 5% false-positive budget reached 0.191 against a required 0.25.
The shortfall holds at every cut point tested (5, 10, 20, 40) and under any
model-selection rule.

### The two attempts

**Attempt 1** used 23 aggregate counts over the prefix: error tallies, tool
counts, file-edit counts, output volume. AUROC 0.668, recall at 5% FPR 0.155.
Permutation importance showed one feature dominating — `n_repeated_calls`, a
flat count of tool calls reissued with byte-identical arguments, at 0.198
AUROC drop — while every error-based feature was negligible.

That suggested the missing ingredient was **ordering**. A count knows an action
recurred; it cannot express that the agent is cycling A-B-A-B, or how far back
it reached to repeat itself.

**Attempt 2** added 10 sequence-structure features: n-gram recurrence,
identical runs, alternation, return distance, action-stream compressibility,
repeat acceleration. Recall at 5% FPR rose from 0.155 to 0.191 and AUROC from
0.668 to 0.677 — real movement on precisely the failing criterion, but not
enough to clear it.

The importance ranking confirmed the mechanism:

| feature | attempt 1 | attempt 2 |
|---|---|---|
| `distinct_action_ratio` | — | **0.159** |
| `novelty_rate_recent` | — | **0.153** |
| `n_errors` | 0.023 | 0.084 |
| `action_bigram_repeat_max` | — | 0.078 |
| `n_repeated_calls` | **0.198** | 0.030 |

`n_repeated_calls` collapsed once ordering was represented properly — it had
been a proxy for structure it could not express. The two strongest features
now both measure **declining action novelty**: the fraction of actions that are
distinct, and the fraction of recent actions never issued before.

Modelling stopped after two attempts. The gains were real but shrinking, the
mechanism is understood, and a third round of feature invention against a bar
that had already failed twice would be threshold-shopping under another name.

### What was established

**Predictive signal exists and is not marginal.** AUROC 0.677 with a
confidence interval of [0.651, 0.697] separates decisively from three
baselines pinned at 0.500–0.521.

**Signal emerges around turn 5 and peaks near turn 10.** At turn 3 every model
sits at chance — a system prompt, a task statement and one action carry no
evidence. Past turn 20 performance decays, partly through survivorship: 37% of
resolved sessions reach turn 40 against 41% of unresolved.

**A 2.1 KB linear model was the most accurate**, beating gradient boosting
(0.677 against 0.629–0.655) at 0.18 ms single-row CPU inference. Plausible
cause: with 33 features, 491 positives and repository-grouped evaluation, trees
fit repository-specific thresholds that do not survive the group boundary. For
scale, the neural monitor this work reproduces uses 0.6B parameters.

**Failure is predicted by declining action novelty, not by errors.** The prior
expectation was that a recurring error signature would dominate. It is close
to worthless in isolation — AUROC 0.521 alone, 0.017 permutation drop. A
session heading nowhere is one that has stopped trying new things, which is a
different phenomenon from one that keeps hitting the same wall.

The error-fingerprinting layer — normalizing paths, line numbers and addresses
so recurring failures collapse to one identity — was built on the assumption
that repeated errors were the signal. It still earns its place, since
`n_errors` rose to 0.084 once ordering was represented, but the central
hypothesis was wrong and measurement is what corrected it.

**No successful session in this corpus ends before turn 10** — 491 of 491
resolved sessions reach it, against 71% of unresolved ones. Early termination
is therefore perfectly associated with failure, which is why the base rate
shifts from 8.2% to 11.2% between turn 5 and turn 10.

### Limitations

- The operating point the product needs is not reachable here. Two attempts,
  the second targeting the diagnosed gap, both fell short of 0.25 recall at a
  5% false-positive budget. Closing it likely needs a genuine sequence model
  over the action stream rather than more engineered summaries of it.
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
