"""F11 regression — lexical consensus must not score negation-contradicted
answers as high-agreement quorum.

The previous ``_compute_quorum`` scored per-pair textual agreement via
``SequenceMatcher`` on whitespace-normalized lowercased text. Two short
phrases that share most characters but mean opposite things — the
canonical example is ``"safe"`` vs ``"not safe"`` — produced a high
score (~0.92) and were reported as a quorum of agreeing providers.

This test pins:
  1. The cheap negation-contradiction heuristic
     (``_detect_negation_contradiction``) catches the exact
     ``"safe"`` vs ``"not safe"`` shape AND a clear second pair.
  2. The textual-agreement composite function
     (``_textual_agreement``) floors negation-contradicted pairs.
  3. ``_compute_quorum`` (the consumer of those two) reports
     ``quorum_reached=False`` and a low ``consensus_score`` for a
     two-muse ``safe`` / ``not safe`` lineup at the default threshold
     (the exact F11 reproduction).
  4. ``_compute_quorum`` emits a ``contradiction_pairs`` audit map so
     operators can see when the textual-agreement floor actually
     mattered.
  5. Non-negation, high-overlap answers still reach quorum (so the
     fix doesn't over-penalize genuine agreement).
"""

from __future__ import annotations

import pytest

from mnemos.domain.graeae.engine import (
    _compute_quorum,
    _detect_negation_contradiction,
    _textual_agreement,
    _text_similarity,
)


# ── Heuristic-level tests ─────────────────────────────────────────────────────


def test_detect_negation_contradiction_safe_vs_not_safe() -> None:
    """The exact reproduction from the F11 review.

    Before the fix, ``SequenceMatcher('safe','not safe').ratio() == 0.92``,
    and ``_compute_quorum`` would have classified two muses returning
    these as in-agreement."""
    assert _detect_negation_contradiction("safe", "not safe") is True, (
        "negation-contradiction heuristic missed the canonical safe/not-safe pair"
    )


def test_detect_negation_contradiction_hot_vs_not_hot() -> None:
    """A second clear negation-contradiction pair the heuristic must
    catch. "hot" vs "not hot" means opposite temperature assessments."""
    assert _detect_negation_contradiction("hot", "not hot") is True


def test_detect_negation_contradiction_approve_vs_not_approve() -> None:
    """Third idiom: ``approve`` vs ``do not approve``."""
    assert _detect_negation_contradiction(
        "approve",
        "do not approve",
    ) is True


def test_detect_negation_contradiction_negative_pair_does_not_fire() -> None:
    """``not hot`` vs ``not cold`` are BOTH negated but share no
    content word. The heuristic must NOT flag this — they don't
    contradict; they're both non-assertions about distinct topics."""
    assert _detect_negation_contradiction("not hot", "not cold") is False, (
        "negation-contradiction heuristic must require a shared content "
        "word — 'not hot' and 'not cold' do not contradict"
    )


def test_detect_negation_contradiction_opposite_antonyms_dont_fire() -> None:
    """``good`` vs ``bad`` are opposite antonyms but lack a negation
    cue on either side — so this case is OUT OF SCOPE for the
    heuristic (it covers negation-contradiction only, not general
    antonymy). Confirm we don't falsely flag it as a contradiction
    (and don't silently treat it as agreement either — the textual
    score should remain low because the words are unrelated)."""
    assert _detect_negation_contradiction("good", "bad") is False, (
        "antonym heuristic out of scope: 'good'/'bad' should not be "
        "flagged as negation-contradiction"
    )
    # And the underlying textual score is low because the texts are
    # unrelated — no false-agreement here either.
    assert _text_similarity("good", "bad") < 0.5


def test_detect_negation_contradiction_identical_texts_dont_fire() -> None:
    """``safe`` vs ``safe`` (or any identical inputs) must not be
    flagged — they're not contradicting, they're agreeing."""
    assert _detect_negation_contradiction("safe", "safe") is False
    assert _detect_negation_contradiction(
        "the answer is approve",
        "the answer is approve",
    ) is False


def test_detect_negation_contradiction_handles_contractions() -> None:
    """``isn't safe`` vs ``safe`` is the same negation as ``not safe``
    vs ``safe`` — the heuristic must handle the ``n't`` contraction."""
    assert _detect_negation_contradiction("safe", "isn't safe") is True
    assert _detect_negation_contradiction("hot", "isn't hot") is True


def test_detect_negation_contradiction_handles_never_cue() -> None:
    """``never safe`` is still a negation of ``safe``."""
    assert _detect_negation_contradiction("safe", "never safe") is True


def test_textual_agreement_safe_vs_not_safe_floors_to_low_score() -> None:
    """Composite: even when the raw SequenceMatcher of the negation-
    contradicted pair is high (e.g. "the answer is safe" vs "the
    answer is not safe" scores ~0.90), the composite ``_textual_agreement``
    must floor it to 0.05 so quorum math cannot accidentally elect the
    pair as agreeing. We don't pin the raw score from outside (different
    difflib versions or inputs yield different absolute numbers) — we
    pin only that ``composite <= raw`` AND ``composite <= 0.05``."""
    raw_short = _text_similarity("safe", "not safe")
    composite_short = _textual_agreement("safe", "not safe")
    assert composite_short <= 0.05, (
        f"composite textual_agreement for a negation-contradicted pair must "
        f"be floored below the default 0.66 quorum threshold, got "
        f"{composite_short} (raw was {raw_short})"
    )
    assert composite_short <= raw_short, (
        "composite must never exceed raw — it can only equal or floor"
    )

    # And repeat with a longer phrase where the raw score IS high enough
    # that the Floor would otherwise matter for quorum math.
    raw_long = _text_similarity(
        "the answer is safe",
        "the answer is not safe",
    )
    composite_long = _textual_agreement(
        "the answer is safe",
        "the answer is not safe",
    )
    assert raw_long > 0.85, (
        f"sanity: longer safe/not-safe phrase should score high; got {raw_long}"
    )
    assert composite_long <= 0.05, (
        f"longer negation-contradicted phrase must also be floored; "
        f"raw={raw_long} composite={composite_long}"
    )


def test_textual_agreement_identical_texts_score_one() -> None:
    """Identical texts must remain at full agreement — the contradiction
    floor must not over-penalize genuine agreement."""
    assert _textual_agreement("approve the design", "approve the design") == 1.0


def test_textual_agreement_unrelated_texts_remain_low() -> None:
    """Two unrelated answers (no shared content, no negation cues) must
    not be lifted into spurious agreement by the heuristic."""
    score = _textual_agreement("the cat is sleeping", "quantum gravity is hard")
    assert score < 0.5


# ── _compute_quorum-level tests ────────────────────────────────────────────────


def _two_muse_lineup(text_a: str, text_b: str) -> dict:
    """Build a 2-muse success dict shaped like ``_compute_quorum`` expects."""
    return {
        "alpha": {"status": "success", "response_text": text_a},
        "beta": {"status": "success", "response_text": text_b},
    }


def test_compute_quorum_safe_vs_not_safe_does_not_reach_quorum() -> None:
    """The exact F11 reproduction.

    Default quorum threshold is 0.66. The pair ``"the answer is safe"``
    vs ``"the answer is not safe"`` scores ~0.9 lexically which would,
    without the fix, union the two muses into a 2-strong component and
    declare quorum reached. The composite textual-agreement score is
    floored to 0.05 by the contradiction modifier, the union-find does
    not connect the two muses, the largest component has 1 muse which
    is below the ``quorum_size = max(2, ceil(0.66*2)) = 2`` requirement,
    so ``quorum_reached`` is False and ``consensus_score`` is well
    below the threshold.

    We test BOTH the canonical short phrase (``"safe"`` vs ``"not safe"``,
    raw score 0.667 — itself below 0.66 so wouldn't quorum even without
    the fix) AND the longer phrase where the raw score IS above 0.66 and
    the contradiction-contradicts-flooring actually changes the verdict."""
    for phrase_a, phrase_b in [
        ("safe", "not safe"),
        ("the answer is safe", "the answer is not safe"),
    ]:
        result = _compute_quorum(
            _two_muse_lineup(phrase_a, phrase_b),
            quorum=0.66,
        )
        assert result["quorum_reached"] is False, (
            f"negation-contradicted pair {phrase_a!r}/{phrase_b!r} must NOT "
            f"reach quorum — got {result!r}"
        )
        assert result["consensus_score"] < 0.66, (
            f"consensus_score for {phrase_a!r}/{phrase_b!r} must be below "
            f"the default quorum threshold, got {result['consensus_score']}"
        )
        pair_score = result["similarity_pairs"]["alpha:beta"]
        assert pair_score <= 0.05, (
            f"per-pair textual-agreement score for {phrase_a!r}/{phrase_b!r} "
            f"must be floored, got {pair_score}"
        )


def test_compute_quorum_safe_vs_not_safe_audits_contradiction_pairs() -> None:
    """The new ``contradiction_pairs`` map surfaces the heuristic's
    positives for operator audit."""
    result = _compute_quorum(
        _two_muse_lineup("safe", "not safe"),
        quorum=0.66,
    )
    assert result["contradiction_pairs"], (
        "contradiction_pairs must include the safe/not-safe hit; got "
        f"{result['contradiction_pairs']!r}"
    )
    assert (
        result["contradiction_pairs"].get("alpha:beta") is True
        or result["contradiction_pairs"].get("beta:alpha") is True
    ), (
        f"contradiction_pairs missing or wrong-direction for safe/not-safe; "
        f"got {result['contradiction_pairs']!r}"
    )


def test_compute_quorum_genuine_agreement_still_reaches_quorum() -> None:
    """Non-contradicting high-overlap answers must STILL reach quorum.

    This is the negative-control half of the fix: we don't want the
    negation-contradiction modifier to accidentally demote genuine
    agreement. Three copies of the same architectural-decision text
    must still register as ``quorum_reached=True`` with a high
    ``consensus_score`` — exactly like the existing
    ``test_majority_mode_reports_quorum_reached`` already proved."""
    lineup = {
        "alpha": {
            "status": "success",
            "response_text": (
                "approve the design because it has a stable cache boundary"
            ),
        },
        "beta": {
            "status": "success",
            "response_text": (
                "approve the design because it has a stable cache boundary"
            ),
        },
        "gamma": {
            "status": "success",
            "response_text": (
                "approve the design because it has a stable cache boundary"
            ),
        },
    }
    result = _compute_quorum(lineup, quorum=0.66)
    assert result["quorum_reached"] is True, (
        f"identical agreeing answers must still reach quorum — got {result!r}"
    )
    assert result["consensus_score"] >= 0.66
    # And the contradiction_pairs audit must be EMPTY — no false positives.
    assert result["contradiction_pairs"] == {}, (
        f"genuine-agreement lineup must not be flagged as contradictions, "
        f"got {result['contradiction_pairs']!r}"
    )


def test_compute_quorum_three_muse_split_no_quorum() -> None:
    """Three muses with three distinct answers ⇒ no quorum.

    Sanity-check the fix doesn't change the
    ``test_majority_mode_reports_quorum_not_reached``-shape behaviour:
    distinct answers don't reach quorum. (We deliberately do NOT
    reuse the F11 case here — two approves + one reject WOULD reach
    quorum because the two approves are genuinely textually agreeing.
    That case is in ``test_compute_quorum_genuine_agreement_still_reaches_quorum``,
    just covering the family-3 shape.)"""
    lineup = {
        "alpha": {"status": "success", "response_text": "approve the design"},
        "beta": {
            "status": "success",
            "response_text": "reject this entire approach",
        },
        "gamma": {
            "status": "success",
            "response_text": "defer this decision to next quarter",
        },
    }
    result = _compute_quorum(lineup, quorum=0.66)
    assert result["quorum_reached"] is False


def test_compute_quorum_handles_single_success_gracefully() -> None:
    """Edge case: a single successful response cannot form a quorum.

    This is the legacy short-circuit path — it must continue to work
    because the negation work added new keys that must always be
    present in the return dict."""
    lineup = {"alpha": {"status": "success", "response_text": "approved"}}
    result = _compute_quorum(lineup, quorum=0.66)
    assert result["quorum_reached"] is False
    assert result["quorum_muses"] == []
    assert result["consensus_score"] == 0.0
    # F11: the new audit field must always be present, even on the
    # short-circuit branch.
    assert "contradiction_pairs" in result
    assert result["contradiction_pairs"] == {}
