"""Tests for rendering, validation, the no-persona control, and the egress guard.

ORIGINAL WORK (Kravis Leadership Institute).

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import copy

import pytest

from kli.assemble_prompts import (
    DEFAULT_SEPARATOR,
    assemble_prompt,
    check_arm_consistency,
    extract_pid,
    resolve_arm_wording,
)
from kli.parse_responses import ValidationFailure, apply_responses, validate_answers
from kli.providers.local import (
    LocalModelConfig,
    RemoteEndpointBlocked,
    assert_local_endpoint,
)
from kli.render_questions import (
    UnsupportedQuestionType,
    build_question_prompt,
    chunk_refs,
    clean_text,
)
from kli.responses_to_csv import extract_columns
from kli.survey_tree import walk_questions


def _doc(questions: list[dict]) -> dict:
    return {
        "Metadata": {},
        "Elements": [
            {
                "ElementType": "Block",
                "BlockName": "A",
                "BlockID": "BL_A",
                "Questions": questions,
            }
        ],
        "EmbeddedData": {},
    }


def _refs(questions: list[dict]) -> list:
    return list(walk_questions(_doc(questions), include_descriptive=False))


MC = {
    "QuestionID": "QID1",
    "QuestionType": "MC",
    "QuestionText": "Pick one",
    "Options": ["Disagree", "Neutral", "Agree"],
    "Settings": {"Selector": "SAVR"},
}
MULTI = {
    "QuestionID": "QID2",
    "QuestionType": "MC",
    "QuestionText": "Pick all",
    "Options": ["A", "B", "C"],
    "Settings": {"Selector": "MAVR"},
}
MATRIX = {
    "QuestionID": "QID3",
    "QuestionType": "Matrix",
    "QuestionText": "Rate each",
    "Rows": ["Item 1", "Item 2"],
    "Columns": ["Low", "Mid", "High"],
    "Settings": {"Selector": "Likert"},
}
SLIDER = {
    "QuestionID": "QID4",
    "QuestionType": "Slider",
    "QuestionText": "Slide",
    "Statements": ["Only"],
    "Range": {"Min": 0, "Max": 100},
    "Settings": {"Selector": "HSLIDER"},
}
CS = {
    "QuestionID": "QID5",
    "QuestionType": "CS",
    "QuestionText": "Allocate 100",
    "Rows": ["X", "Y"],
    "Settings": {"Selector": "VRTL", "ChoiceTotal": "100"},
}
TE = {
    "QuestionID": "QID6",
    "QuestionType": "TE",
    "QuestionText": "Explain",
    "Settings": {"Selector": "SL"},
}
NUMERIC = {
    "QuestionID": "QID7",
    "QuestionType": "TE",
    "QuestionText": "How many?",
    "Settings": {"Selector": "ESTB"},
}


class TestTextCleaning:
    def test_unescapes_entities(self) -> None:
        """Upstream leaks &rsquo; and friends into prompts verbatim."""
        assert clean_text("it&rsquo;s") == "it’s"
        assert clean_text("A &amp; B") == "A & B"

    def test_separates_adjacent_inline_tags(self) -> None:
        """Upstream produces 'BBCNews' from this input."""
        assert clean_text("<b>BBC</b><i>News</i>") == "BBC News"

    def test_preserves_paragraph_breaks(self) -> None:
        assert clean_text("<p>One</p><p>Two</p>").count("\n") >= 1

    def test_handles_none_and_non_strings(self) -> None:
        assert clean_text(None) == ""
        assert clean_text(7) == "7"


class TestRendering:
    def test_renders_constant_sum(self) -> None:
        """The upstream renderer raises ValueError on CS."""
        prompt, skeleton = build_question_prompt(_refs([CS]))
        assert "Constant sum" in prompt
        assert "sum to exactly 100" in prompt
        assert skeleton["Q1"]["Answers"]["Values"] == [None, None]

    def test_skeleton_carries_question_id(self) -> None:
        _, skeleton = build_question_prompt(_refs([MC]))
        assert skeleton["Q1"]["QuestionID"] == "QID1"

    def test_multi_select_differs_from_single(self) -> None:
        _, single = build_question_prompt(_refs([MC]))
        _, multi = build_question_prompt(_refs([MULTI]))
        assert single["Q1"]["Answers"]["SelectedByPosition"] is None
        assert multi["Q1"]["Answers"]["SelectedByPosition"] == []

    def test_matrix_skeleton_length_matches_rows(self) -> None:
        _, skeleton = build_question_prompt(_refs([MATRIX]))
        assert len(skeleton["Q1"]["Answers"]["SelectedByPosition"]) == 2

    def test_reasoning_adds_field(self) -> None:
        _, skeleton = build_question_prompt(_refs([MC]), include_reasoning=True)
        assert "Reasoning" in skeleton["Q1"]

    def test_unsupported_type_raises_with_guidance(self) -> None:
        bad = {"QuestionID": "QIDX", "QuestionType": "RO", "QuestionText": "Rank"}
        with pytest.raises(UnsupportedQuestionType, match="exclude_questions"):
            build_question_prompt(_refs([bad]))


class TestChunking:
    def test_none_means_one_chunk(self) -> None:
        refs = _refs([MC, MULTI, MATRIX])
        assert len(chunk_refs(refs, None)) == 1

    def test_splits_without_losing_questions(self) -> None:
        refs = _refs([MC, MULTI, MATRIX, SLIDER, CS])
        chunks = chunk_refs(refs, 2)
        assert [len(c) for c in chunks] == [2, 2, 1]
        flat = [r.question_id for c in chunks for r in c]
        assert flat == [r.question_id for r in refs]

    def test_labels_stay_globally_unique_across_chunks(self) -> None:
        """Chunk-local renumbering would collide when results are merged."""
        refs = _refs([MC, MULTI, MATRIX, SLIDER])
        labels: list[str] = []
        for chunk in chunk_refs(refs, 2):
            _, skeleton = build_question_prompt(chunk)
            labels.extend(skeleton)
        assert len(set(labels)) == len(labels) == 4


class TestValidation:
    def test_accepts_valid_single_choice(self) -> None:
        ref = _refs([MC])[0]
        out = validate_answers(ref, {"SelectedByPosition": 2, "SelectedText": "x"})
        assert out["SelectedByPosition"] == 2
        assert out["SelectedText"] == "Neutral", "label re-derived from instrument"

    def test_rejects_out_of_range_option(self) -> None:
        ref = _refs([MC])[0]
        with pytest.raises(ValidationFailure, match="outside 1..3"):
            validate_answers(ref, {"SelectedByPosition": 9})

    def test_rejects_multiple_selections_for_single_choice(self) -> None:
        ref = _refs([MC])[0]
        with pytest.raises(ValidationFailure, match="single-choice"):
            validate_answers(ref, {"SelectedByPosition": [1, 2]})

    def test_rejects_wrong_matrix_length(self) -> None:
        """A model dropping a matrix row must not pass silently."""
        ref = _refs([MATRIX])[0]
        with pytest.raises(ValidationFailure, match="expected 2 row answers"):
            validate_answers(ref, {"SelectedByPosition": [1]})

    def test_rejects_slider_out_of_bounds(self) -> None:
        ref = _refs([SLIDER])[0]
        with pytest.raises(ValidationFailure, match="outside"):
            validate_answers(ref, {"Values": [500]})

    def test_rejects_constant_sum_that_does_not_sum(self) -> None:
        ref = _refs([CS])[0]
        with pytest.raises(ValidationFailure, match="total"):
            validate_answers(ref, {"Values": [10, 10]})

    def test_accepts_constant_sum_within_rounding_tolerance(self) -> None:
        ref = _refs([CS])[0]
        assert validate_answers(ref, {"Values": [50, 50]})["Values"] == ["50", "50"]

    def test_rejects_non_numeric_for_numeric_entry(self) -> None:
        ref = _refs([NUMERIC])[0]
        with pytest.raises(ValidationFailure, match="non-numeric"):
            validate_answers(ref, {"Text": "quite a lot"})

    def test_rejects_empty_text(self) -> None:
        ref = _refs([TE])[0]
        with pytest.raises(ValidationFailure, match="empty"):
            validate_answers(ref, {"Text": "   "})


class TestApplyResponses:
    def _apply(self, response: dict, **kwargs) -> tuple[dict, object]:
        doc = _doc([copy.deepcopy(MC)])
        refs = list(walk_questions(doc, include_descriptive=False))
        report = apply_responses(refs, response, participant_id="1", **kwargs)
        return doc, report

    def test_successful_imputation_is_flagged(self) -> None:
        doc, report = self._apply(
            {"Q1": {"QuestionID": "QID1", "Answers": {"SelectedByPosition": 3}}}
        )
        question = doc["Elements"][0]["Questions"][0]
        assert question["LLM_Imputed"] is True
        assert question["LLM_Status"] == "ok"
        assert report.complete

    def test_human_answer_is_preserved_separately(self) -> None:
        doc = _doc([{**copy.deepcopy(MC), "Answers": {"SelectedByPosition": 1}}])
        refs = list(walk_questions(doc, include_descriptive=False))
        apply_responses(
            refs,
            {"Q1": {"QuestionID": "QID1", "Answers": {"SelectedByPosition": 3}}},
            participant_id="1",
        )
        question = doc["Elements"][0]["Questions"][0]
        assert question["Original_Answers"] == {"SelectedByPosition": 1}
        assert question["Answers"]["SelectedByPosition"] == 3

    def test_failure_blanks_the_answer_by_default(self) -> None:
        """Upstream leaves the human's answer in a file labelled model output."""
        doc = _doc([{**copy.deepcopy(MC), "Answers": {"SelectedByPosition": 1}}])
        refs = list(walk_questions(doc, include_descriptive=False))
        report = apply_responses(refs, {}, participant_id="1")
        question = doc["Elements"][0]["Questions"][0]
        assert question["Answers"] is None, "no ground-truth leakage"
        assert question["LLM_Imputed"] is False
        assert question["Original_Answers"] == {"SelectedByPosition": 1}
        assert report.n_failed == 1

    def test_keep_original_is_available_but_still_flagged(self) -> None:
        doc = _doc([{**copy.deepcopy(MC), "Answers": {"SelectedByPosition": 1}}])
        refs = list(walk_questions(doc, include_descriptive=False))
        apply_responses(refs, {}, participant_id="1", on_failure="keep_original")
        question = doc["Elements"][0]["Questions"][0]
        assert question["Answers"] == {"SelectedByPosition": 1}
        assert question["LLM_Imputed"] is False

    def test_failure_reason_is_recorded_on_the_question(self) -> None:
        doc, _ = self._apply(
            {"Q1": {"QuestionID": "QID1", "Answers": {"SelectedByPosition": 99}}}
        )
        status = doc["Elements"][0]["Questions"][0]["LLM_Status"]
        assert status.startswith("invalid:")


class TestNoPersonaControl:
    def test_control_omits_persona_and_header(self) -> None:
        prompt = assemble_prompt(
            question_prompt="QUESTIONS",
            persona_text="SECRET PERSONA",
            empty_persona=True,
        )
        assert "SECRET PERSONA" not in prompt
        assert "Persona Profile" not in prompt
        assert "QUESTIONS" in prompt

    def test_treatment_includes_persona(self) -> None:
        prompt = assemble_prompt(
            question_prompt="QUESTIONS", persona_text="MY PERSONA"
        )
        assert "MY PERSONA" in prompt

    def test_question_section_is_identical_across_arms(self) -> None:
        """The property that makes the two arms comparable."""
        kwargs = {"question_prompt": "QUESTIONS", "separator": DEFAULT_SEPARATOR}
        treatment = assemble_prompt(persona_text="P", **kwargs)
        control = assemble_prompt(
            persona_text="P", empty_persona=True, **kwargs
        )
        marker = DEFAULT_SEPARATOR
        assert treatment[treatment.index(marker) :] == control[control.index(marker) :]

    def test_persona_arm_refuses_to_run_without_a_persona(self) -> None:
        """Silently dropping the persona would turn treatment into control."""
        with pytest.raises(ValueError, match="control observation"):
            assemble_prompt(question_prompt="Q", persona_text=None)

    def test_no_warning_under_upstream_wording(self) -> None:
        """Upstream wording is the deliberate default, not a mistake.

        Peng et al. kept the persona-arm strings in their control arm -- so a
        persona-referencing instruction there is the intended design, covered by
        the printed note rather than a warning.
        """
        assert not check_arm_consistency(
            empty_persona=True,
            system_instruction="Answer as that person would.",
            control_wording="upstream",
        )

    def test_warns_when_neutral_is_defeated_by_an_override(self) -> None:
        """Asking for a neutral control and getting upstream wording anyway."""
        warnings = check_arm_consistency(
            empty_persona=True,
            system_instruction="Answer as that person would.",
            control_wording="neutral",
        )
        assert warnings and "that person" in warnings[0]

    def test_no_warning_for_genuinely_neutral_wording(self) -> None:
        assert not check_arm_consistency(
            empty_persona=True,
            system_instruction="Answer the survey questions.",
            control_wording="neutral",
        )

    def test_no_warning_for_the_persona_arm(self) -> None:
        assert not check_arm_consistency(
            empty_persona=False, system_instruction="Answer as the persona."
        )


class TestControlWordingModes:
    """The upstream-vs-neutral choice, and that the default reproduces upstream."""

    def test_default_is_upstream(self) -> None:
        resolved = resolve_arm_wording(empty_persona=True)
        assert resolved["control_wording"] == "upstream"

    def test_upstream_keeps_persona_arm_strings(self) -> None:
        """Exactly what Peng et al. did: only the persona flag changes."""
        treatment = resolve_arm_wording(empty_persona=False)
        control = resolve_arm_wording(
            empty_persona=True, control_wording="upstream"
        )
        assert control["separator"] == treatment["separator"]
        assert control["system_instruction"] == treatment["system_instruction"]
        assert control["persona_header"] == treatment["persona_header"]
        assert control["persona_wording"] is True

    def test_neutral_replaces_persona_referencing_strings(self) -> None:
        treatment = resolve_arm_wording(empty_persona=False)
        control = resolve_arm_wording(empty_persona=True, control_wording="neutral")
        assert control["separator"] != treatment["separator"]
        assert control["system_instruction"] != treatment["system_instruction"]
        assert control["persona_wording"] is False
        assert "persona" not in control["system_instruction"].lower()

    def test_upstream_mode_prints_a_note(self) -> None:
        """The design choice should not be discoverable only by reading code."""
        resolved = resolve_arm_wording(empty_persona=True)
        assert resolved["note"] and "Peng et al." in resolved["note"]

    def test_neutral_mode_prints_no_note(self) -> None:
        assert resolve_arm_wording(
            empty_persona=True, control_wording="neutral"
        )["note"] is None

    def test_persona_arm_prints_no_note(self) -> None:
        resolved = resolve_arm_wording(empty_persona=False)
        assert resolved["note"] is None
        assert resolved["control_wording"] is None

    def test_explicit_override_wins_in_either_mode(self) -> None:
        for mode in ("upstream", "neutral"):
            resolved = resolve_arm_wording(
                empty_persona=True,
                control_wording=mode,
                system_instruction="MY INSTRUCTION",
            )
            assert resolved["system_instruction"] == "MY INSTRUCTION"

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="upstream' or 'neutral"):
            resolve_arm_wording(empty_persona=True, control_wording="whatever")

    def test_questions_identical_under_both_wordings(self) -> None:
        """Only the opening sentence may differ; the questions never do."""
        refs = _refs([MC, MATRIX])
        with_persona, skel_a = build_question_prompt(refs, persona_wording=True)
        without, skel_b = build_question_prompt(refs, persona_wording=False)
        assert skel_a == skel_b
        marker = "Q1. [QuestionID: QID1]"
        assert with_persona[with_persona.index(marker) :] == without[
            without.index(marker) :
        ]


class TestEgressGuard:
    @pytest.mark.parametrize(
        "url", ["http://localhost:11434", "http://127.0.0.1:8080", "http://[::1]:1234"]
    )
    def test_allows_loopback(self, url: str) -> None:
        assert_local_endpoint(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com",
            "http://8.8.8.8:11434",
            "http://example.com",
        ],
    )
    def test_blocks_non_loopback(self, url: str) -> None:
        with pytest.raises(RemoteEndpointBlocked):
            assert_local_endpoint(url)

    def test_blocks_unresolvable_host(self) -> None:
        """Unverifiable is treated as remote, not as local."""
        with pytest.raises(RemoteEndpointBlocked, match="Cannot resolve"):
            assert_local_endpoint("http://this-host-does-not-exist.invalid")

    def test_explicit_override_allows_remote(self) -> None:
        assert_local_endpoint("https://api.openai.com", allow_remote=True)

    def test_config_enforces_guard_at_construction(self) -> None:
        with pytest.raises(RemoteEndpointBlocked):
            LocalModelConfig(base_url="https://api.openai.com")

    def test_manifest_records_the_guard_state(self) -> None:
        manifest = LocalModelConfig().manifest()
        assert manifest["endpoint_is_loopback"] is True
        assert manifest["allow_remote_endpoint"] is False

    def test_rejects_unknown_api_style(self) -> None:
        with pytest.raises(ValueError, match="api_style"):
            LocalModelConfig(api_style="telepathy")


class TestCsvExtraction:
    def test_single_choice_emits_position(self) -> None:
        doc = _doc([{**copy.deepcopy(MC), "Answers": {"SelectedByPosition": 2}}])
        values, _ = extract_columns(doc)
        assert values["QID1"] == 2

    def test_multi_select_emits_one_column_per_selection(self) -> None:
        """json2csv.py reads 'Selected' here and so emits nothing at all."""
        doc = _doc(
            [
                {
                    **copy.deepcopy(MULTI),
                    "Answers": {
                        "SelectedByPosition": [1, 3],
                        "SelectedText": ["A", "C"],
                    },
                }
            ]
        )
        values, _ = extract_columns(doc)
        assert values["QID2_1"] == 1
        assert values["QID2_3"] == 1
        assert "QID2_2" not in values

    def test_tolerates_the_upstream_selected_field_name(self) -> None:
        doc = _doc([{**copy.deepcopy(MULTI), "Answers": {"Selected": [2]}}])
        values, _ = extract_columns(doc)
        assert values["QID2_2"] == 1

    def test_matrix_emits_one_column_per_row(self) -> None:
        doc = _doc(
            [{**copy.deepcopy(MATRIX), "Answers": {"SelectedByPosition": [1, 3]}}]
        )
        values, _ = extract_columns(doc)
        assert values["QID3_1"] == 1
        assert values["QID3_2"] == 3

    def test_text_entry_uses_text_suffix(self) -> None:
        doc = _doc([{**copy.deepcopy(TE), "Answers": {"Text": "hello"}}])
        values, _ = extract_columns(doc)
        assert values["QID6_TEXT"] == "hello"

    def test_labels_mode_emits_text(self) -> None:
        doc = _doc(
            [
                {
                    **copy.deepcopy(MC),
                    "Answers": {"SelectedByPosition": 3, "SelectedText": "Agree"},
                }
            ]
        )
        labels, _ = extract_columns(doc, labels=True)
        assert labels["QID1"] == "Agree"

    def test_imputation_flags_are_reported(self) -> None:
        doc = _doc(
            [
                {
                    **copy.deepcopy(MC),
                    "Answers": {"SelectedByPosition": 1},
                    "LLM_Imputed": True,
                }
            ]
        )
        _, flags = extract_columns(doc)
        assert flags["QID1"] is True

    def test_blanked_answers_emit_no_column(self) -> None:
        doc = _doc([{**copy.deepcopy(MC), "Answers": None, "LLM_Imputed": False}])
        values, _ = extract_columns(doc)
        assert "QID1" not in values


class TestPidExtraction:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("pid_1030_response.json", "1030"),
            ("pid_1030_persona.txt", "1030"),
            ("pid_ABC123_response.json", "ABC123"),
            ("pid_A-1_response.json", "A-1"),
            ("no_pid_here.json", None),
        ],
    )
    def test_extracts_numeric_and_non_numeric_ids(
        self, filename: str, expected: str | None
    ) -> None:
        """Upstream hardcodes pid_\\d+ in three places, excluding non-numeric IDs."""
        assert extract_pid(filename) == expected
