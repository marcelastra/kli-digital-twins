"""Tests for combining the three waves and for the SJT-inclusion arm.

ORIGINAL WORK (Kravis Leadership Institute).

The fixtures mirror the real KLI surveys: a StudentID question (QID1001) in
every wave, judgment tasks under a Randomizer, and ten SJTs in Wave 3 as 6-row,
7-point Likert matrices (QID1101-QID1110).

Run with:  python -m pytest tests/test_sjt_arm.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from kli.combine_waves import (
    add_twin_id,
    combine_documents,
    combine_wave_dirs,
    combined_dataframe,
)
from kli.run_simulation import run
from kli.sjt_split import (
    KLI_SJT_QUESTION_IDS,
    SjtLeakError,
    check_question_set,
    draw_split,
    prepare,
    render_training_sjts,
    split_document,
)
from kli.survey_tree import walk_questions

PIDS = ("101", "102", "103")


def _mc(qid: str, tag: str, answer: int = 1) -> dict:
    return {
        "QuestionID": qid,
        "QuestionName": tag,
        "QuestionType": "MC",
        "Settings": {"Selector": "SAVR"},
        "QuestionText": f"Question {tag}",
        "Options": ["A", "B", "C"],
        "Answers": {"SelectedByPosition": answer, "SelectedText": "ABC"[answer - 1]},
    }


def _student_id(pid: str) -> dict:
    return {
        "QuestionID": "QID1001",
        "QuestionName": "StudentID",
        "QuestionType": "TE",
        "Settings": {"Selector": "SL"},
        "QuestionText": "Please enter your school ID number.",
        "Answers": {"Text": pid},
    }


def _sjt(n: int, ratings: list[int]) -> dict:
    return {
        "QuestionID": f"QID{1100 + n}",
        "QuestionName": f"SJT{n}",
        "QuestionType": "Matrix",
        "Settings": {"Selector": "Likert", "SubSelector": "SingleAnswer"},
        "QuestionText": f"Situation {n}. How likely are you to take each action?",
        "Rows": [f"Action {n}.{i}" for i in range(1, 7)],
        "Columns": [str(i) for i in range(1, 8)],
        "Answers": {
            "SelectedByPosition": ratings,
            "SelectedText": [str(r) for r in ratings],
        },
    }


def _block(name: str, questions: list[dict]) -> dict:
    return {"ElementType": "Block", "BlockName": name, "BlockID": f"BL_{name}", "Questions": questions}


def _doc(elements: list[dict]) -> dict:
    return {"Metadata": {}, "Elements": elements, "EmbeddedData": {}}


def _waves(pid: str) -> dict[str, dict]:
    seed = int(pid)
    return {
        "wave1": _doc(
            [
                _block("Welcome", [_student_id(pid)]),
                {
                    "ElementType": "Randomizer",
                    "FlowID": "FL_6",
                    "Elements": [_block("Sunk cost - yes", [_mc("QID201", "sunk", 2)])],
                },
            ]
        ),
        "wave2": _doc([_block("Welcome", [_student_id(pid)]), _block("Linda", [_mc("QID301", "linda")])]),
        "wave3": _doc(
            [
                _block("Welcome", [_student_id(pid)]),
                _block(
                    "Leadership development",
                    [_sjt(n, [(seed + n + i) % 7 + 1 for i in range(6)]) for n in range(1, 11)],
                ),
            ]
        ),
    }


def _write_wave_dirs(root: Path, pids=PIDS) -> dict[str, Path]:
    dirs = {w: root / w for w in ("wave1", "wave2", "wave3")}
    for pid in pids:
        for wave, document in _waves(pid).items():
            dirs[wave].mkdir(parents=True, exist_ok=True)
            (dirs[wave] / f"pid_{pid}_response.json").write_text(json.dumps(document))
    return dirs


def _qids(document: dict) -> list[str]:
    return [r.question_id for r in walk_questions(document, include_descriptive=False)]


# --------------------------------------------------------------------------- #
# combining waves
# --------------------------------------------------------------------------- #


class TestCombineWaves:
    def test_flattens_blocks_and_tags_wave(self) -> None:
        combined = combine_documents(_waves("101"))
        assert all(e["ElementType"] == "Block" for e in combined["Elements"])
        assert [e["Wave"] for e in combined["Elements"]] == ["wave1", "wave2", "wave3"]

    def test_randomizer_blocks_survive(self) -> None:
        # The vendored persona renderer skips anything under a Randomizer, so
        # the combined document must not leave any block nested.
        assert "QID201" in _qids(combine_documents(_waves("101")))

    def test_id_question_is_dropped(self) -> None:
        assert "QID1001" not in _qids(combine_documents(_waves("101")))

    def test_duplicate_qid_across_waves_is_refused(self) -> None:
        with pytest.raises(ValueError, match="QID1001 appears in both"):
            combine_documents(_waves("101"), exclude_question_ids=())

    def test_order_is_wave_order(self) -> None:
        qids = _qids(combine_documents(_waves("101")))
        assert qids[:2] == ["QID201", "QID301"]
        assert qids[2:] == list(KLI_SJT_QUESTION_IDS)

    def test_participant_missing_a_wave_is_reported_not_combined(self, tmp_path: Path) -> None:
        dirs = _write_wave_dirs(tmp_path / "in")
        (dirs["wave2"] / "pid_103_response.json").unlink()
        report = combine_wave_dirs(dirs, tmp_path / "out")
        assert report.combined == 2
        assert report.incomplete == {"103": ["wave2"]}
        assert not (tmp_path / "out" / "pid_103_response.json").exists()

    def test_final_dataframe_one_row_per_participant(self, tmp_path: Path) -> None:
        combine_wave_dirs(_write_wave_dirs(tmp_path / "in"), tmp_path / "out")
        frame = combined_dataframe(tmp_path / "out")
        assert list(frame.index) == list(PIDS)
        assert "SJT1_1" in frame.columns and "SJT10_6" in frame.columns
        assert "sunk" in frame.columns and "linda" in frame.columns
        assert frame.loc["101", "sunk"] == 2

    def test_add_twin_id(self, tmp_path: Path) -> None:
        raw = tmp_path / "wave1.csv"
        pd.DataFrame(
            {
                "StudentID": ["Please enter...", '{"ImportId":"QID1001_TEXT"}', "101", "102", "102", ""],
                "Q1": ["text", '{"ImportId":"QID201"}', "1", "2", "3", "1"],
            }
        ).to_csv(raw, index=False)
        report = add_twin_id(raw, tmp_path / "out.csv")
        out = pd.read_csv(tmp_path / "out.csv", dtype=str, keep_default_na=False)
        assert list(out.columns[:2]) == ["TWIN_ID", "StudentID"]
        assert list(out["TWIN_ID"][2:4]) == ["101", "102"]
        assert report["duplicates"] == ["102"]
        assert report["blank"] == [5]

    def test_add_twin_id_never_overwrites_raw(self, tmp_path: Path) -> None:
        raw = tmp_path / "wave1.csv"
        raw.write_text("StudentID\nx\ny\n1\n")
        with pytest.raises(ValueError, match="overwrite"):
            add_twin_id(raw, raw)


# --------------------------------------------------------------------------- #
# the split
# --------------------------------------------------------------------------- #


class TestDrawSplit:
    def test_two_holdouts_eight_training(self) -> None:
        split = draw_split(seed=7)
        assert len(split.holdout_question_ids) == 2
        assert len(split.training_question_ids) == 8
        assert set(split.holdout_question_ids).isdisjoint(split.training_question_ids)
        assert set(split.holdout_question_ids) | set(split.training_question_ids) == set(
            KLI_SJT_QUESTION_IDS
        )

    def test_same_seed_same_draw(self) -> None:
        assert draw_split(seed=7) == draw_split(seed=7)

    def test_draw_is_actually_random(self) -> None:
        draws = {draw_split(seed=s).holdout_question_ids for s in range(30)}
        assert len(draws) > 5

    def test_seed_is_required(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            draw_split()

    def test_explicit_override(self) -> None:
        split = draw_split(holdout_question_ids=["QID1110", "QID1103"])
        assert split.holdout_question_ids == ("QID1103", "QID1110")
        assert split.method == "explicit"

    def test_unknown_holdout_rejected(self) -> None:
        with pytest.raises(ValueError, match="not among"):
            draw_split(holdout_question_ids=["QID9999"])


class TestSplitDocument:
    def test_three_parts(self) -> None:
        split = draw_split(seed=7)
        parts = split_document(combine_documents(_waves("101")), split)
        assert _qids(parts["holdout"]) == list(split.holdout_question_ids)
        assert _qids(parts["training"]) == list(split.training_question_ids)
        assert not set(_qids(parts["persona"])) & set(KLI_SJT_QUESTION_IDS)
        assert {"QID201", "QID301"} <= set(_qids(parts["persona"]))

    def test_missing_sjt_refused(self) -> None:
        document = combine_documents(_waves("101"))
        document["Elements"][-1]["Questions"].pop()
        with pytest.raises(SjtLeakError, match="QID1110"):
            split_document(document, draw_split(seed=7))

    def test_render_shows_own_ratings(self) -> None:
        split = draw_split(holdout_question_ids=["QID1109", "QID1110"])
        training = split_document(combine_documents(_waves("101")), split)["training"]
        text = render_training_sjts(training)
        assert text.count("Scenario ") == 8
        assert "Situation 9" not in text
        # pid 101, SJT1, action 1: (101 + 1 + 0) % 7 + 1 = 5
        assert "[5] Action 1.1" in text
        assert "Scale: 1 to 7" in text

    def test_render_uses_scale_labels_when_present(self) -> None:
        question = _sjt(1, [1, 2, 3, 4, 5, 6])
        question["Columns"] = ["Very unlikely", "2", "3", "4", "5", "6", "Very likely"]
        text = render_training_sjts(_doc([_block("L", [question])]))
        assert "1 = Very unlikely" in text and "7 = Very likely" in text

    def test_check_question_set(self) -> None:
        split = draw_split(seed=7)
        parts = split_document(combine_documents(_waves("101")), split)
        check_question_set(parts["holdout"], split)
        with pytest.raises(SjtLeakError, match="Training SJTs"):
            check_question_set(parts["training"], split)
        with pytest.raises(SjtLeakError, match="does not match"):
            check_question_set(parts["persona"], split)


# --------------------------------------------------------------------------- #
# the arm, end to end through run() in stub mode
# --------------------------------------------------------------------------- #


@pytest.fixture()
def study(tmp_path: Path) -> dict:
    combine_wave_dirs(_write_wave_dirs(tmp_path / "waves"), tmp_path / "combined")
    split = draw_split(seed=7)
    prepare(combined_json_dir=tmp_path / "combined", output_dir=tmp_path / "split", split=split)
    personas = tmp_path / "persona_text"
    personas.mkdir()
    for pid in PIDS:
        (personas / f"pid_{pid}_response.txt").write_text(f"PERSONA OF {pid}")
    return {"root": tmp_path, "split": split, "personas": personas}


def _config(study: dict, *, include: bool, empty: bool = False, scramble: bool = False) -> Path:
    root: Path = study["root"]
    arm = f"{'empty' if empty else 'scrambled' if scramble else 'matched'}_{'on' if include else 'off'}"
    out = root / "runs" / arm
    config = {
        "text_simulation": {
            "questions_to_texts": {"include_reasoning": False},
            "create_text_simulation_input": {
                "persona_text_dir": str(study["personas"]),
                "output_combined_prompts_dir": str(out / "prompts"),
                "empty_persona_prompt": empty,
                "scramble_personas": scramble,
                "scramble_seed": 42,
            },
            "sjt_split": {
                "include_training_sjts_in_persona": include,
                "split_file": str(root / "split" / "sjt_split.json"),
                "training_json_dir": str(root / "split" / "training_json"),
            },
            "LLM_simulation": {
                "question_json_base_dir": str(root / "split" / "holdout_json"),
                "output_updated_questions_dir": str(out / "json"),
                "LLM_config": {"provider": "local", "base_url": "http://localhost:11434"},
                "force_regenerate": True,
            },
        }
    }
    path = root / f"{arm}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _prompt(study: dict, arm: str, pid: str) -> str:
    return (study["root"] / "runs" / arm / "prompts" / f"pid_{pid}_prompt.txt").read_text()


class TestArmEndToEnd:
    @pytest.mark.parametrize("empty,scramble", [(False, False), (False, True), (True, False)])
    @pytest.mark.parametrize("include", [False, True])
    def test_all_six_cells_run(self, study: dict, include: bool, empty: bool, scramble: bool) -> None:
        manifest = run(_config(study, include=include, empty=empty, scramble=scramble), stub=True)
        assert manifest["participants_failed"] == 0, manifest["errors"]
        assert manifest["imputation"]["imputation_rate"] == 1.0
        assert manifest["sjt_split"]["include_training_sjts_in_persona"] is include
        assert manifest["sjt_split"]["holdout_question_ids"] == list(
            study["split"].holdout_question_ids
        )

    def test_switch_adds_only_the_training_block(self, study: dict) -> None:
        run(_config(study, include=False), dry_run=True)
        run(_config(study, include=True), dry_run=True)
        off, on = _prompt(study, "matched_off", "101"), _prompt(study, "matched_on", "101")
        assert "PERSONA OF 101" in off and "PERSONA OF 101" in on
        assert "workplace scenarios" not in off
        assert on.count("Scenario ") == 8
        # Everything in the "off" prompt appears unchanged in the "on" prompt.
        head, _, tail = off.partition("PERSONA OF 101")
        assert on.startswith(head + "PERSONA OF 101") and on.endswith(tail)

    def test_holdout_scenarios_never_in_profile(self, study: dict) -> None:
        run(_config(study, include=True), dry_run=True)
        prompt = _prompt(study, "matched_on", "101")
        for qid in study["split"].holdout_question_ids:
            n = int(qid[3:]) - 1100
            # Exactly once: as a question, never as a rated training scenario.
            assert prompt.count(f"Situation {n}.") == 1

    def test_scrambled_training_sjts_come_from_the_persona_donor(self, study: dict) -> None:
        run(_config(study, include=True, scramble=True), dry_run=True)
        for pid in PIDS:
            prompt = _prompt(study, "scrambled_on", pid)
            donor = prompt.split("PERSONA OF ")[1].split()[0]
            assert donor != pid
            first = study["split"].training_question_ids[0]
            n = int(first[3:]) - 1100
            assert f"[{(int(donor) + n) % 7 + 1}] Action {n}.1" in prompt

    def test_sjt_only_twin_has_profile_but_no_persona(self, study: dict) -> None:
        manifest = run(_config(study, include=True, empty=True), dry_run=True)
        prompt = _prompt(study, "empty_on", "101")
        assert "PERSONA OF" not in prompt
        assert "workplace scenarios" in prompt
        assert manifest["persona_header"] is not None

    def test_training_sjt_in_questions_is_refused(self, study: dict) -> None:
        config = _config(study, include=False)
        data = yaml.safe_load(config.read_text())
        data["text_simulation"]["LLM_simulation"]["question_json_base_dir"] = str(
            study["root"] / "split" / "training_json"
        )
        config.write_text(yaml.safe_dump(data))
        with pytest.raises(SjtLeakError, match="Training SJTs"):
            run(config, stub=True)

    def test_switch_must_be_explicit(self, study: dict) -> None:
        config = _config(study, include=False)
        data = yaml.safe_load(config.read_text())
        del data["text_simulation"]["sjt_split"]["include_training_sjts_in_persona"]
        config.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match="include_training_sjts_in_persona"):
            run(config, stub=True)


def test_num_ctx_is_read_from_config(study: dict) -> None:
    # The persona built from three waves is far larger than the 8192 default,
    # so the configured window must actually reach the model client.
    from kli.config import load_config
    from kli.run_simulation import RunSettings

    config = load_config(_config(study, include=True))
    config["text_simulation"]["LLM_simulation"]["LLM_config"]["num_ctx"] = 32768
    assert RunSettings.from_config(config).model.num_ctx == 32768
