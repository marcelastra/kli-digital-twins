"""Tests for the canonical traversal and the round-trip guarantee.

ORIGINAL WORK (Kravis Leadership Institute).

These are the tests that matter most in this repository. The failure they guard
against -- prompt construction and response parsing disagreeing about survey
structure -- produces output that looks completely normal while every answer
after the point of divergence is attached to the wrong question.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest

from kli.survey_tree import (
    RoundtripError,
    count_questions,
    elements_of,
    fingerprint,
    verify_roundtrip,
    walk_questions,
)


def _mc(qid: str, n_options: int = 3) -> dict:
    return {
        "QuestionID": qid,
        "QuestionType": "MC",
        "QuestionText": f"Question {qid}",
        "Options": [f"Option {i}" for i in range(1, n_options + 1)],
        "Settings": {"Selector": "SAVR"},
    }


def _db(qid: str) -> dict:
    return {
        "QuestionID": qid,
        "QuestionType": "DB",
        "QuestionText": "Some instructions.",
    }


def _block(name: str, questions: list[dict]) -> dict:
    return {
        "ElementType": "Block",
        "BlockName": name,
        "BlockID": f"BL_{name}",
        "Questions": questions,
    }


FLAT_DOC = {
    "Metadata": {},
    "Elements": [_block("A", [_mc("QID1"), _mc("QID2")])],
    "EmbeddedData": {},
}

NESTED_DOC = {
    "Metadata": {},
    "Elements": [
        _block("Intro", [_db("QID0")]),
        {
            "ElementType": "Randomizer",
            "FlowID": "FL_1",
            "SelectElements": 1,
            "Elements": [
                {
                    "ElementType": "Group",
                    "FlowID": "FL_2",
                    "Description": "Arm A",
                    "Elements": [_block("A", [_mc("QID1")])],
                },
                {
                    "ElementType": "Branch",
                    "FlowID": "FL_3",
                    "BranchLogic": {},
                    "Elements": [_block("B", [_mc("QID2"), _db("QID9")])],
                },
            ],
        },
        _block("Outro", [_mc("QID3")]),
    ],
    "EmbeddedData": {},
}


class TestSchemaAdapter:
    """The dict-vs-list normalization that joins the two upstream repos."""

    def test_accepts_dict_with_elements(self) -> None:
        assert len(elements_of(FLAT_DOC)) == 1

    def test_accepts_bare_list(self) -> None:
        """The shape the vendored demo code expects."""
        assert len(elements_of(FLAT_DOC["Elements"])) == 1

    def test_accepts_json_string(self) -> None:
        import json

        assert len(elements_of(json.dumps(FLAT_DOC))) == 1

    def test_accepts_double_encoded_string(self) -> None:
        """Some HuggingFace persona columns are stored double-encoded."""
        import json

        inner = json.dumps(FLAT_DOC)
        assert len(elements_of(json.dumps(inner))) == 1

    def test_dict_and_list_give_identical_fingerprints(self) -> None:
        assert fingerprint(NESTED_DOC) == fingerprint(NESTED_DOC["Elements"])

    def test_rejects_dict_without_elements(self) -> None:
        with pytest.raises(TypeError, match="without an 'Elements' key"):
            elements_of({"Metadata": {}})

    def test_rejects_wrong_type(self) -> None:
        with pytest.raises(TypeError):
            elements_of(42)


class TestTraversal:
    """Ordering, nesting and ordinal assignment."""

    def test_walks_nested_containers(self) -> None:
        """Randomizer > Group/Branch > Block must all be traversed."""
        ids = [r.question_id for r in walk_questions(NESTED_DOC)]
        assert ids == ["QID0", "QID1", "QID2", "QID9", "QID3"]

    def test_ordinals_are_contiguous_from_one(self) -> None:
        refs = list(walk_questions(NESTED_DOC, include_descriptive=False))
        assert [r.ordinal for r in refs] == [1, 2, 3]

    def test_descriptive_elements_do_not_consume_ordinals(self) -> None:
        """Adding instructions must not renumber the answer slots."""
        before = fingerprint(NESTED_DOC)
        import copy

        doc = copy.deepcopy(NESTED_DOC)
        doc["Elements"][0]["Questions"].append(_db("QID_EXTRA"))
        assert fingerprint(doc) == before

    def test_labels_match_ordinals(self) -> None:
        refs = list(walk_questions(NESTED_DOC, include_descriptive=False))
        assert [r.label for r in refs] == ["Q1", "Q2", "Q3"]

    def test_deterministic_across_calls(self) -> None:
        assert fingerprint(NESTED_DOC) == fingerprint(NESTED_DOC)

    def test_count_matches_walk(self) -> None:
        assert count_questions(NESTED_DOC) == 3

    def test_skips_instrumentation_types(self) -> None:
        doc = {
            "Elements": [
                _block(
                    "A",
                    [
                        _mc("QID1"),
                        {"QuestionID": "QIDT", "QuestionType": "Timing"},
                        {"QuestionID": "QIDM", "QuestionType": "Meta"},
                    ],
                )
            ]
        }
        assert [r.question_id for r in walk_questions(doc)] == ["QID1"]

    def test_traverses_unknown_container_rather_than_dropping_it(self) -> None:
        """Upstream drops unknown flow types, losing every question beneath.

        Dropping them changes the ordinal count, which is exactly the class of
        failure this module exists to prevent.
        """
        doc = {
            "Elements": [
                {
                    "ElementType": "SomeFutureQualtricsThing",
                    "FlowID": "FL_X",
                    "Elements": [_block("Hidden", [_mc("QID7")])],
                }
            ]
        }
        assert [r.question_id for r in walk_questions(doc)] == ["QID7"]

    def test_path_records_structure(self) -> None:
        ref = next(
            r for r in walk_questions(NESTED_DOC) if r.question_id == "QID2"
        )
        assert ref.path == ("Randomizer:FL_1", "Branch:FL_3", "Block:B", "QID2")

    def test_question_dict_is_live(self) -> None:
        """Mutating a ref must mutate the document, for write-back."""
        import copy

        doc = copy.deepcopy(FLAT_DOC)
        ref = next(iter(walk_questions(doc)))
        ref.question["Answers"] = {"SelectedByPosition": 2}
        assert doc["Elements"][0]["Questions"][0]["Answers"] == {
            "SelectedByPosition": 2
        }


class TestRoundtrip:
    """The guarantee that an answer belongs to the question it answers."""

    def test_accepts_matching_id(self) -> None:
        ref = next(iter(walk_questions(FLAT_DOC)))
        verify_roundtrip(ref, {"QuestionID": "QID1", "Answers": {}})

    def test_rejects_mismatched_id(self) -> None:
        """The misassignment that upstream performs silently."""
        ref = next(iter(walk_questions(FLAT_DOC)))
        with pytest.raises(RoundtripError, match="asked about"):
            verify_roundtrip(ref, {"QuestionID": "QID_WRONG", "Answers": {}})

    def test_rejects_missing_id_when_strict(self) -> None:
        ref = next(iter(walk_questions(FLAT_DOC)))
        with pytest.raises(RoundtripError, match="omitted"):
            verify_roundtrip(ref, {"Answers": {}})

    def test_tolerates_missing_id_when_not_strict(self) -> None:
        ref = next(iter(walk_questions(FLAT_DOC)))
        verify_roundtrip(ref, {"Answers": {}}, strict=False)

    def test_still_rejects_wrong_id_when_not_strict(self) -> None:
        """Lenient about omission, never about contradiction."""
        ref = next(iter(walk_questions(FLAT_DOC)))
        with pytest.raises(RoundtripError):
            verify_roundtrip(ref, {"QuestionID": "QID_WRONG"}, strict=False)

    def test_tolerates_whitespace_and_case_of_key(self) -> None:
        ref = next(iter(walk_questions(FLAT_DOC)))
        verify_roundtrip(ref, {"question_id": " QID1 "})
