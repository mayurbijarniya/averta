"""Prefix feature extraction.

Every extractor receives `turns[0:t]` and nothing else. No extractor may read
the total session length, the outcome, or any turn at or beyond index *t* —
see rule 3 in CLAUDE.md. `tests/test_leakage.py` enforces this by truncating
and shuffling the unseen tail and asserting the feature vector is unchanged.
"""

from averta.features.extractors import EXTRACTORS, FEATURE_NAMES, extract
from averta.features.view import TurnView, to_views

__all__ = ["EXTRACTORS", "FEATURE_NAMES", "TurnView", "extract", "to_views"]
