"""Enforces the prefix-only rule.

A feature computed at turn t must depend on turns[0:t] alone. If any extractor
starts reading the unseen tail, or the eventual outcome, these tests fail.
"""

import inspect
import random

import pytest

from averta.features import EXTRACTORS, extract
from averta.testing import action, observation, session

CUTS = [1, 3, 5, 10, 20]


@pytest.mark.parametrize("cut", CUTS)
def test_features_ignore_a_shuffled_tail(cut):
    turns = session(30)
    baseline = extract(turns[:cut])

    rng = random.Random(0)
    tail = turns[cut:]
    rng.shuffle(tail)

    assert extract((turns[:cut] + tail)[:cut]) == baseline


@pytest.mark.parametrize("cut", CUTS)
def test_features_ignore_a_truncated_tail(cut):
    turns = session(30)
    assert extract(turns[:cut]) == extract(turns[:cut])
    assert extract((turns[:cut] + turns[cut : cut + 2])[:cut]) == extract(turns[:cut])


@pytest.mark.parametrize("cut", CUTS)
def test_features_ignore_an_extended_tail(cut):
    turns = session(30)
    baseline = extract(turns[:cut])

    noisy = turns[:cut] + [
        observation(99, error="CatastrophicError: everything broke"),
        action(100, path="/workspace/zzz.py"),
    ]
    assert extract(noisy[:cut]) == baseline


def test_session_length_is_not_recoverable_from_the_feature_vector():
    short = extract(session(12)[:6])
    long = extract(session(200)[:6])
    assert short == long, "features at turn 6 must not vary with eventual session length"


def test_every_extractor_takes_exactly_one_argument():
    for name, fn in EXTRACTORS.items():
        parameters = list(inspect.signature(fn).parameters)
        assert len(parameters) == 1, f"{name} must accept only a prefix, got {parameters}"


def test_extractors_do_not_mutate_the_prefix():
    turns = session(20)
    before = list(turns)
    extract(turns[:10])
    assert turns == before


def test_empty_prefix_is_safe():
    values = extract([])
    assert all(isinstance(value, float) for value in values.values())
    assert set(values) == set(EXTRACTORS)
