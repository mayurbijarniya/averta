# Averta

**Predicting when an AI coding agent is about to fail — from a 2 KB model that
scores a live session in 0.18 ms on one CPU core.**

![tests](https://img.shields.io/badge/tests-272%20passing-2f7d4f)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)
![gate](https://img.shields.io/badge/pre--registered%20gate-not%20met-b0342f)

When a coding agent gets stuck, it keeps spending tokens that produce nothing.
Averta watches a session and reports what is going wrong — and, separately,
estimates whether the session is heading toward failure.

```
session 2e9e5250 · 1,417 turns

MEASURED — exact, no model involved
  agent errors            8
  user rejections         4  (not agent failures)

  clustered repetition  (>=3x within 25 turns)
    10x  file edit  turns 1357-1381   adapters/claude_code.py
     6x  file edit  turns 1251-1271   site.py

ESTIMATED — model did not clear its gate; context only
  risk over turns 5-40   █▇▁█
  corpus base rate 90.6% — compare against this, not against zero
```

## What this actually is

A **pre-registered evaluation study**. The success criteria were written into
source control *before any model was trained*, so they could not be relaxed
later to fit a disappointing result. They were then missed — twice, across two
disclosed attempts judged against identical thresholds.

That is the point of the project, not a footnote to it.

| | |
|---|---|
| **The question** | Can a cheap CPU model replace a 0.6B neural monitor for early failure detection? |
| **The answer** | It recovers roughly **half** the token savings — 8.1% against a reported 14.6–20.4% — from a 2.1 KB model rather than a 0.6B-parameter one |
| **The gate** | Required 0.25 recall at a 5% false-positive budget. Reached **0.191**. Not met. |
| **What is solid** | AUROC **0.677** [0.651, 0.697] against three baselines pinned at 0.500 |

Anyone can publish a model with a good number. The harder thing — and what
this repository is really a demonstration of — is committing to a bar in
advance, missing it, diagnosing *why*, retrying against the **unchanged** bar,
missing again, and reporting both attempts.

**[→ Full results, charts and methodology](site/index.html)** · rendered by
`averta site` from the committed artifacts, so the page cannot drift from the
numbers behind it.

**[→ Model card](MODEL_CARD.md)** · intended use, measured limits, and what
this model must not be used for.

## How it works

Public recordings of AI coding agents — 5,976 sessions where the outcome is
known — are normalized into one schema. Features are extracted from the
*first N turns only*, and a classifier predicts the eventual outcome from what
was observable at that point.

The useful half needs no model at all: recurring error signatures, tool calls
reissued with byte-identical arguments, and edits clustered tightly in time are
counted directly from the transcript. Output always separates what is
**measured** from what is **estimated**.

This reproduces and extends
[Fail-Fast, Restart-Smart](https://arxiv.org/abs/2608.03222) (Wang et al.,
2026), which uses a 0.6B neural monitor over the same kind of trajectory
prefixes. The contribution here is the cheap-model comparison, measured.

### Contents

[Data](#data) · [Evaluation](#evaluation) · [Results](#results) ·
[Two attempts](#the-two-attempts) · [Limitations](#limitations) ·
[Usage](#usage) · [MCP server](#mcp-server)

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

Three constraints guard against the usual ways of fooling yourself:

- **Prefix-only features.** A feature computed at turn *t* reads `turns[0:t]`
  and nothing else — never the total length, never the outcome. A test suite
  shuffles, truncates and extends the unseen tail and asserts the feature
  vector is byte-identical.
- **Absolute turn indices.** Evaluating at "40% through the session" needs the
  total length, which is unknown while a session runs. Evaluation uses turns
  3, 5, 10, 20 and 40.
- **Repository-grouped splits.** Every row from a repository lands in one
  fold, so no model is tested on a codebase it trained on. Confidence
  intervals resample repositories rather than rows for the same reason.

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

### Token savings, and the comparison to the paper

Sweeping the intervention threshold at turn 10 trades savings against harm.
The corpus records no token counts, so tokens here are **estimated** from
content length at four characters per token. Savings count only what would
have been spent *after* the cut, and sessions wrongly terminated are never
netted off the savings.

| threshold | recall | FPR | est. tokens saved | of total | successes killed |
|---|---|---|---|---|---|
| 0.55 | 0.501 | 0.246 | 25,207,866 | 33.6% | 121 |
| 0.65 | 0.283 | 0.096 | 11,932,608 | 15.9% | 47 |
| **0.80** | **0.181** | **0.047** | **6,103,533** | **8.1%** | **23** |
| 0.90 | 0.051 | 0.010 | 1,822,964 | 2.4% | 5 |

At a comparable false-positive budget — 4.7% against the paper's 5% target —
this reaches **8.1% estimated token savings** where the 0.6B neural monitor
reports 14.6–20.4%.

So the honest summary of the central research question: a 2.1 KB linear model
running in 0.18 ms on one CPU core recovers roughly **half** the token savings
of a 0.6B neural monitor. That is the result, and it cuts both ways. The cost
reduction is enormous and the capability gap is real.

Caveats on that comparison: different corpus, different agent scaffold, and
our tokens are estimated rather than counted. It is the closest like-for-like
available, not a controlled replication.

### Cross-scaffold transfer is unmeasured

The plan was to train on OpenHands trajectories and test on hand-labelled
Claude Code sessions, reporting the AUROC gap. That is not reported, because
only three local transcripts are long enough and none carry outcome labels.
A transfer AUROC on n=3 would be noise presented as a result.

What is measurable without labels is whether the features compute comparably
at all — a prerequisite for transfer rather than a substitute for measuring it.
At cut point 40, corpus (n=2,409) against local (n=3):

| feature | corpus | local | std diff |
|---|---|---|---|
| `n_steps` | 18.72 | 23.33 | +5.29 (outside training range) |
| `n_tool_calls` | 18.84 | 10.00 | −4.40 |
| `turns_since_error` | 9.30 | 40.00 | +2.74 |
| `error_rate` | 0.23 | 0.00 | −1.77 |

Every comparable feature shifts by more than 1.3 standard deviations, and two
fall outside the range the model was fitted on. Claude Code emits more
assistant turns per tool call than OpenHands, because thinking blocks become
their own turns — so a "turn" is not the same unit across scaffolds.

Consequently the live monitor labels its output **indicative**, in every code
path including over MCP.

This check also caught a real bug. `n_edits` and `n_files_touched` initially
reported exactly 0.00 on sessions full of edits: `edited_path` was keyed to
the OpenHands `str_replace_editor` argument schema, while Claude Code uses
separate `Edit`/`Write`/`MultiEdit` tools with `file_path`. The feature failed
silently to zero and the model extrapolated on it. Both vocabularies are now
recognised. That is the concrete form of the cross-schema problem — not a
slightly worse score, but a confident zero for something that happened
dozens of times.

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

## Usage

Requires Python 3.11+. Nothing calls a network API at inference time, and no
session data leaves the machine.

xgboost needs an OpenMP runtime, which is not bundled:

```bash
brew install libomp          # macOS
sudo apt-get install libgomp1  # Debian/Ubuntu
```

Then:

```bash
git clone <repo> && cd averta
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

### Analyse your own sessions

These work immediately — the trained model is committed, so no download is
needed:

```bash
averta explain     # full analysis of your most recent coding session
averta sessions    # list local Claude Code transcripts
averta score       # just the risk estimate
averta drift       # compare your sessions against the training corpus
averta site        # rebuild the results page
```

`averta explain` is the one worth running. It separates what is **measured**
from what is **estimated**, and never mixes them:

```
session 2e9e5250-7972-4be0-a210-bd8e7ab3c7e4
turns: 1,417

MEASURED — exact, no model involved
  agent errors            8
  user rejections         4  (not agent failures)
  turns since a clean result  1

  clustered repetition  (>=3x within 25 turns)
    10x  file edit  turns 1357-1381  2,808 chars, 13x overall
         src/averta/adapters/claude_code.py
     6x  file edit  turns 1251-1271  4,038 chars, 8x overall
         src/averta/site.py

ESTIMATED — model did not clear its gate; context only
  risk over turns 5-40   █▇▁█  (flat)
    turn  10   87.2%
    turn  40   89.4%
  corpus base rate 90.6% — compare against this, not against zero
  stops at turn 40: beyond the largest evaluated prefix, so no curve is drawn
```

The measured half needs no model and is as reliable as the transcript itself:
recurring error signatures, tool calls reissued with byte-identical arguments,
and edits clustered tightly in time. Repetition is ranked by **density**, not
total count — editing one file 27 times across a thousand turns is ordinary
iterative work, while five identical commands in twelve turns is a loop.

The estimated half is labelled as such, shown against the base rate rather
than against zero, and stops at the largest evaluated prefix instead of
extrapolating a curve into territory the model never saw.

Two details behind that output are worth stating.

**The base rate is shown next to the probability.** Most sessions in the corpus
fail, so 89.7% sounds alarming until you see that 90.6% is typical. Reporting
the probability alone hid the fact that the model had no real signal. Scores are
isotonically calibrated on out-of-fold predictions, because class weighting is
needed for ranking but leaves raw scores on a re-balanced scale — that fix moved
the Brier score from 0.2323 to 0.0826.

**Long sessions are truncated, not extrapolated.** Features are cumulative over
the prefix, so a 900-turn session scored against a model fitted on prefixes of
at most 40 turns puts every count far outside the fitted range; that produced a
meaningless 100.0% before the fix. The report always states which turn it
actually scored.


### Reproduce the study

```bash
averta ingest      # normalize the public corpus into DuckDB (~12 min)
averta validate    # structural checks; must print "all checks passed"
averta features    # build the prefix matrix at cuts 3/5/10/20/40
averta train       # cross-validate every model and apply the gate
averta diagnose    # permutation importance and CPU inference cost
averta savings     # token savings against sessions wrongly terminated
averta figures     # render the three result figures
averta site        # build a static results page from the artifacts
```

`averta train` exits non-zero when the gate is not met, which is its normal
state here. Changing anything under `features/` invalidates the stored matrix,
the model and the gate results — re-run `features → train → fit`.

### MCP server

Exposes the same analysis to a coding agent over stdio. The agent asks about
its own session; the agent's tokens pay for the conversation, and the
prediction itself costs nothing.

```bash
claude mcp add averta -- /absolute/path/to/.venv/bin/averta-mcp
```

| tool | returns |
|---|---|
| `get_session_risk` | failure probability, contributing features, caveats |
| `get_repeated_failures` | recurring error signatures and reissued tool calls — measured, no model |
| `should_i_restart` | threshold applied to the risk estimate, with evidence |
| `list_sessions` | local sessions available to inspect |

Every response carries the gate result and the cross-scaffold caveat, so an
agent relaying a number also relays its limits.

## Development

```bash
.venv/bin/python -m pytest -q                    # 187 tests
.venv/bin/python -m ruff check src tests scripts
```

The leakage suite is the one to keep green: it shuffles, truncates and extends
the unseen tail of a session and asserts the feature vector is byte-identical,
and it checks that features at turn 6 are the same whether the session runs to
12 turns or 200.

## License

MIT, covering the source in this repository. The training corpus carries no
declared license and is not redistributed here — see [LICENSE](LICENSE) for
the distinction.
