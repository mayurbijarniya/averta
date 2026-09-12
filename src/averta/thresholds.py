"""Pre-registered success criteria for the Phase 3 gate.

Committed before any model was trained. Revised once, on 2026-09-10, after
Phase 0 measured turn counts as near-identical across outcomes (39.86 resolved
against 39.66 unresolved), which left the turn-index baseline close to chance
and made the original relative threshold trivial to clear. That revision
responded to data statistics, not to model results.

These values are frozen. A run that misses them is a negative result and the
documented fallback applies, see `.claude/phases/03-models-gate.md`.

## Attempt history

**Attempt 1 (2026-09-11), NOT MET.** 23 aggregate features over the prefix.
Best AUROC 0.668 [0.645, 0.689]; recall at 5% FPR reached 0.155 against the
required 0.25, and failed at every cut point. Permutation importance showed
`n_repeated_calls` dominating at 0.198 AUROC drop while error-based features
contributed almost nothing.

**Attempt 2 (2026-09-11), in progress.** Adds 10 sequence-structure features
(`features/sequence.py`): n-gram recurrence, identical runs, alternation,
return distance, action-stream compressibility, repeat acceleration. Motivated
by attempt 1's own importance ranking, its strongest signal was a flat count
of repetition, which is precisely what ordering would express properly.

The thresholds below are **unchanged from attempt 1 on purpose.** A second
attempt judged against a lowered bar would prove nothing. Attempt 2 either
clears the original criteria or it does not, and either way both attempts are
reported.
"""

from __future__ import annotations

from dataclasses import dataclass

GATE_CUT_POINT = 10

MIN_AUROC = 0.65
MIN_AUROC_CI_LOWER = 0.60
MIN_AUPRC = 0.18
MIN_RECALL_AT_TARGET_FPR = 0.25
TARGET_FPR = 0.05
MUST_BEAT_TURN_BASELINE = True

BOOTSTRAP_SAMPLES = 2000
RANDOM_SEED = 17


@dataclass(frozen=True)
class GateResult:
    auroc: float
    auroc_ci_lower: float
    auprc: float
    recall_at_target_fpr: float
    beats_turn_baseline: bool

    @property
    def checks(self) -> dict[str, bool]:
        return {
            f"auroc >= {MIN_AUROC}": self.auroc >= MIN_AUROC,
            f"auroc ci lower >= {MIN_AUROC_CI_LOWER}": self.auroc_ci_lower >= MIN_AUROC_CI_LOWER,
            f"auprc >= {MIN_AUPRC}": self.auprc >= MIN_AUPRC,
            f"recall@{TARGET_FPR:.0%} fpr >= {MIN_RECALL_AT_TARGET_FPR}": (
                self.recall_at_target_fpr >= MIN_RECALL_AT_TARGET_FPR
            ),
            "beats turn-index baseline": (
                self.beats_turn_baseline or not MUST_BEAT_TURN_BASELINE
            ),
        }

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def render(self) -> str:
        lines = [f"Phase 3 gate at cut point {GATE_CUT_POINT}", ""]
        for label, ok in self.checks.items():
            lines.append(f"  {'PASS' if ok else 'FAIL'}  {label}")
        lines.append("")
        lines.append(f"  verdict: {'GO' if self.passed else 'NO-GO'}")
        return "\n".join(lines)
