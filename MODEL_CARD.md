# Model Card: Averta failure predictor

Last updated 2026-09-12. Every figure here is measured; see `artifacts/` for
the JSON each one comes from.

## Summary

| | |
|---|---|
| **Task** | Binary classification, will this coding-agent session fail to resolve its task? |
| **Positive class** | **Failure** (session does not resolve) |
| **Input** | 33 numeric features over the first *N* turns of a session |
| **Output** | Calibrated probability of failure |
| **Architecture** | Logistic regression with standardization and class weighting, plus an isotonic calibration layer |
| **Size** | 2.1 KB (3.9 KB including the calibrator) |
| **Latency** | 0.180 ms p50, 0.187 ms p95, single row, one CPU core |
| **Training data** | 5,976 sessions / 22,557 prefix rows from `SWE-Gym/OpenHands-Sampled-Trajectories` |
| **License** | MIT (code). Training corpus is not redistributed. |

## Intended use

Advisory context for a developer or agent inspecting a coding session. It
pairs with deterministic repetition detection, which carries the load: the
counts of recurring errors and reissued tool calls are exact, and the model
score is supporting context.

**It did not meet its acceptance criteria.** See [Evaluation](#evaluation).

## Out of scope: do not use it for these

- **Automatically terminating sessions.** At a 5% false-positive budget it
  catches 19.1% of failing sessions. Four in five doomed sessions pass
  unflagged, and acting automatically would destroy recoverable work for a
  minority of the waste.
- **Any consequential or irreversible decision**, billing, access, scheduling,
  or evaluating a person's work.
- **Agent scaffolds other than OpenHands**, without re-measuring. Applied to
  Claude Code transcripts, every comparable feature shifts by more than 1.3
  standard deviations and two fall outside the training range entirely.
  Cross-scaffold accuracy is **unmeasured**.
- **Sessions beyond 40 turns.** Features are cumulative over the prefix, so
  anything longer is out of distribution. The scorer truncates to the largest
  evaluated cut and says so rather than extrapolating, before that guard, a
  935-turn session scored a meaningless 100.0%.
- **Sessions under 5 turns.** Every model scored at chance at turn 3. No score
  is returned.

## Training data

[SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories)
- OpenHands agent attempts at SWE-bench-style issues, each labelled with
whether it resolved the issue.

Chosen because a survey of six public corpora found it to be the **only one
carrying both outcome classes**; the rest are supervised fine-tuning sets that
store the agent's patch but never whether it worked.

| | |
|---|---|
| Trajectories scanned | 6,055 |
| After collapsing exact duplicates | 5,976 |
| Turn records | 237,132 |
| Label coverage | 100% |
| Resolve rate | 8.1% overall, 11.2% at the gate cut point |
| Distinct repositories | 11 |
| Distinct task instances | 2,438 |

**Known composition limits.** Eleven repositories is a small number of
grouping units, which widens every confidence interval. One agent scaffold,
one model family, one task type (GitHub issue resolution). Nothing here speaks
to agents doing greenfield work, refactoring, or operating outside a
repository.

The source declares no license, so it is not redistributed here. It is
downloaded at build time and excluded from version control; only derived
statistics, figures and model artifacts are published.

## Evaluation

Five repository-grouped folds, out-of-fold predictions pooled, confidence
intervals resampling **repositories** rather than rows because sessions within
a repository are correlated.

At the pre-registered gate (turn 10, 4,381 rows, 88.79% failure rate):

| metric | value | criterion | |
|---|---|---|---|
| AUROC | 0.677 [0.651, 0.697] | ≥ 0.65 | pass |
| AUROC CI lower | 0.651 | ≥ 0.60 | pass |
| AUPRC (minority class) | 0.198 | ≥ 0.18 | pass |
| **Recall @ 5% FPR** | **0.191** | **≥ 0.25** | **NOT MET** |
| Beats turn-index baseline | yes | required | pass |

**Verdict: not met**, across two attempts judged against identical thresholds.
The shortfall holds at every cut point tested and under any model-selection
rule.

Baselines: majority-class and turn-index-only both sit at exactly 0.500;
a single-feature error-repetition baseline reaches 0.521.

Accuracy is not reported. At an 89% failure rate, always predicting failure
scores 0.89 and is useless.

### Calibration

Class weighting is necessary for ranking but leaves raw scores on a
re-balanced scale, a raw 0.25 corresponded to an observed failure rate near
0.70. An isotonic layer fitted on **out-of-fold** predictions corrects this,
moving the Brier score from **0.2323 to 0.0826**.

Because most sessions in the corpus fail, an absolute probability carries
little information on its own. Output always shows the corpus base rate
alongside the score, and a probability at or below it is reported as *no clear
signal*.

## What the model keys on

Permutation importance under the same grouped folds:

| feature | AUROC drop |
|---|---|
| `distinct_action_ratio` | 0.159 |
| `novelty_rate_recent` | 0.153 |
| `n_errors` | 0.084 |
| `action_bigram_repeat_max` | 0.078 |
| `n_repeated_calls` | 0.030 |

The prior expectation, that recurring error signatures would dominate, was
wrong. Failure is predicted by **declining action novelty**: a session heading
nowhere is one that has stopped trying new things, which is a different
phenomenon from one that keeps hitting the same wall.

## Ethical and practical considerations

**The harm is asymmetric and is reported separately.** A wrongly flagged
session is destroyed work; a missed failure is only wasted tokens. Savings and
sessions-terminated are never combined into one net figure, because a net
number lets the harm disappear into an aggregate.

**It must not be used to evaluate people.** It scores an agent trajectory, not
a developer. A session that looks repetitive may be careful iterative work;
the repetition detector explicitly ranks by density within a short window for
this reason, because editing one file 27 times across a thousand turns is
ordinary development.

**Human rejections are not agent failures.** Claude Code marks a declined tool
call as an error; counting those would make a closely supervised session look
like a struggling one. They are tracked separately.

**Privacy.** Everything runs locally. Session transcripts are read from disk,
never transmitted. No API key is required and no network call is made at
inference time.

## Reproducing

```bash
averta ingest && averta features    # rebuild the corpus (~12 min)
averta train                        # cross-validate and apply the gate
averta diagnose                     # importance, calibration, inference cost
averta fit                          # fit and calibrate the shipped model
```

`averta train` exits non-zero when the gate is not met, which is its expected
state. Random seeds are fixed; changing anything under `features/` invalidates
the stored matrix, the model and the gate results.

## Maintenance

Unmaintained beyond this study. If the upstream corpus changes, results will
drift and must be regenerated. `MODEL_CARD.md`, `README.md` and the results
page are all derived from `artifacts/` and should be regenerated together.
