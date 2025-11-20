#!/usr/bin/env python3
"""Unit tests covering the per-note delta semantics for motif matching."""

import os
import sys
import unittest

# Ensure src modules are importable when running the tests directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from constants import SEPARATOR_1, SEPARATOR_2, SOLID_ALPHABET  # type: ignore
from suffix_tree import SuffixTree  # type: ignore

SOLID_OFFSET = ord('A')
SEPARATORS = {SEPARATOR_1, SEPARATOR_2}
SOLID_SET = set(SOLID_ALPHABET)


def _pitch_class(char: str) -> int:
    """Convert a solid character (A-L) into a pitch class."""
    return ord(char) - SOLID_OFFSET


def _pitch_diff(pc1: int, pc2: int) -> int:
    """Return the circular distance between two pitch classes."""
    return min((pc1 - pc2) % 12, (pc2 - pc1) % 12)


def simple_match(char1: str, char2: str):
    """Replicate the solid matching rules without chords for focused testing."""
    if char1 in SEPARATORS or char2 in SEPARATORS:
        return (char1 == char2, 0.0 if char1 == char2 else float('inf'), 0)

    if char1 in SOLID_SET and char2 in SOLID_SET:
        pc1 = _pitch_class(char1)
        pc2 = _pitch_class(char2)
        if pc1 == pc2:
            return True, 0.0, 0
        diff = _pitch_diff(pc1, pc2)
        return False, float(diff), diff

    # Non-solid symbols are not used in these tests
    return False, float('inf'), float('inf')


def run_lce(text: str, pattern: str, delta: int, gamma: int) -> int:
    """Helper that runs the LCE query for the provided solid strings."""
    combined = text + SEPARATOR_1 + pattern + SEPARATOR_2
    tree = SuffixTree(combined)
    start_pattern = len(text) + 1
    return tree.lce_k_gamma_query(0, start_pattern, delta, gamma, simple_match, max_length=len(pattern))


class TestDeltaSemantics(unittest.TestCase):
    """Focused tests for the updated delta semantics."""

    def test_does_not_accept_large_single_difference(self):
        """Matching stops when a single note exceeds the per-note delta bound."""
        length = run_lce("ABF", "ABC", delta=2, gamma=10)
        self.assertEqual(length, 2)

    def test_accepts_difference_within_delta(self):
        """Matching continues when the pitch gap is within the delta limit."""
        length = run_lce("ABF", "ABC", delta=4, gamma=10)
        self.assertEqual(length, 3)

    def test_gamma_still_limits_even_if_delta_allows(self):
        """Gamma bound remains enforced when per-note delta passes."""
        length = run_lce("ABK", "ABC", delta=5, gamma=3)
        self.assertEqual(length, 2)


if __name__ == '__main__':
    unittest.main()
