from dataclasses import replace

from averta.thresholds import (
    MIN_AUPRC,
    MIN_AUROC,
    MIN_AUROC_CI_LOWER,
    MIN_RECALL_AT_TARGET_FPR,
    GateResult,
)

PASSING = GateResult(
    auroc=0.71,
    auroc_ci_lower=0.66,
    auprc=0.24,
    recall_at_target_fpr=0.31,
    beats_turn_baseline=True,
)


class TestGateResult:
    def test_clearly_passing_run(self):
        assert PASSING.passed

    def test_auroc_below_threshold_fails(self):
        assert not replace(PASSING, auroc=MIN_AUROC - 0.01).passed

    def test_wide_confidence_interval_fails(self):
        assert not replace(PASSING, auroc_ci_lower=MIN_AUROC_CI_LOWER - 0.01).passed

    def test_low_auprc_fails(self):
        assert not replace(PASSING, auprc=MIN_AUPRC - 0.01).passed

    def test_low_recall_fails(self):
        assert not replace(
            PASSING, recall_at_target_fpr=MIN_RECALL_AT_TARGET_FPR - 0.01
        ).passed

    def test_failing_to_beat_the_baseline_fails(self):
        assert not replace(PASSING, beats_turn_baseline=False).passed

    def test_exact_threshold_values_pass(self):
        borderline = GateResult(
            auroc=MIN_AUROC,
            auroc_ci_lower=MIN_AUROC_CI_LOWER,
            auprc=MIN_AUPRC,
            recall_at_target_fpr=MIN_RECALL_AT_TARGET_FPR,
            beats_turn_baseline=True,
        )
        assert borderline.passed

    def test_render_reports_the_verdict(self):
        assert "GO" in PASSING.render()
        assert "NO-GO" in replace(PASSING, auroc=0.5).render()

    def test_every_criterion_is_checked(self):
        assert len(PASSING.checks) == 5
