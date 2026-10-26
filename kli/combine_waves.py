"""Combine the three survey waves into one record per participant.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

-------------------------------------------------------------------------------
WHY THIS EXISTS
-------------------------------------------------------------------------------

The KLI study was fielded as three Qualtrics surveys (Wave 1, 2, 3), each
answered by the same people. Ingestion (``parse_qsf.py`` then
``get_responses.py``) runs once per wave and yields one response JSON per
participant *per wave*. Neither upstream repository has a step that joins
waves: the Mega-Study treats waves 1-3 as one pre-merged persona.

This module provides that step, in two outputs:

``<output_dir>/pid_<id>_response.json``
    One response document per participant holding all three waves as a *flat*
    list of Blocks, in wave order, each Block tagged with a ``Wave`` key.

    Flat on purpose. ``get_responses.py`` keeps ``Branch`` and ``Randomizer``
    containers (already resolved to the blocks this participant saw, in the
    order shown), but the vendored persona renderer,
    ``text_simulation/convert_persona_to_text.py:176``, descends only into
    ``Branch``. Every block under a ``Randomizer`` or ``Group`` is silently left
    out of the persona. All three KLI waves put their judgment tasks under
    BlockRandomizers, so nesting here would quietly strip them from every twin.
    Flattening loses nothing, because the randomizer has already been resolved.

``combined_human_values.csv`` / ``combined_human_labels.csv``
    The "final df": one row per participant, one column per answer slot across
    all waves. Columns are named by Qualtrics export tag (``SJT1_3``) where the
    tag is unique, otherwise by ImportId (``QID1101_3``).

-------------------------------------------------------------------------------
THE PARTICIPANT ID
-------------------------------------------------------------------------------

The wave surveys identify people by a typed school ID (``StudentID``,
``QID1001``) rather than an embedded ``TWIN_ID``. ``get_responses.py`` hardcodes
``TWIN_ID`` (line 37) and silently produces an empty result without it, so
:func:`add_twin_id` copies ``StudentID`` into a ``TWIN_ID`` column of each
export before ingestion. Because the ID is *typed*, a typo in one wave puts the
same person under two IDs. That surfaces here as a participant missing from
some waves, which is why the report lists those IDs rather than dropping them
quietly.

The ID question itself is asked in all three waves under the same QID, so it is
excluded from the combined document by default: it is not respondent content,
and a QID appearing three times would break the one-answer-per-QID assumption.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from kli.assemble_prompts import extract_pid
from kli.config import get, load_config
from kli.responses_to_csv import extract_columns
from kli.survey_tree import elements_of, walk_questions

__all__ = [
    "DEFAULT_ID_QUESTION_IDS",
    "add_twin_id",
    "filter_document",
    "combine_documents",
    "combine_wave_dirs",
    "combined_dataframe",
    "CombineReport",
]

#: The ``StudentID`` question, present in every wave under this QID.
DEFAULT_ID_QUESTION_IDS = ("QID1001",)


def add_twin_id(
    csv_in: str | Path,
    csv_out: str | Path,
    *,
    id_column: str = "StudentID",
) -> dict[str, Any]:
    """Copy a Qualtrics export, adding the ``TWIN_ID`` column ingestion needs.

    Qualtrics exports have three header rows (names, question text, ImportId
    JSON); the new column gets matching entries in all three.

    Args:
        csv_in: The raw Qualtrics export for one wave.
        csv_out: Where to write the copy. Must differ from ``csv_in``.
        id_column: The column holding the participant ID.

    Returns:
        ``{"rows", "duplicates", "blank", "non_numeric"}``. Each list names the
        offending IDs (or row numbers, for blanks).

    Raises:
        ValueError: If the ID column is missing, a ``TWIN_ID`` column already
            exists, or ``csv_out`` would overwrite ``csv_in``.
    """
    csv_in, csv_out = Path(csv_in), Path(csv_out)
    if csv_in.resolve() == csv_out.resolve():
        raise ValueError("add_twin_id() will not overwrite the raw export.")

    df = pd.read_csv(csv_in, dtype=str, keep_default_na=False)
    if id_column not in df.columns:
        raise ValueError(
            f"{csv_in}: no {id_column!r} column. Columns start: "
            f"{list(df.columns)[:12]}"
        )
    if "TWIN_ID" in df.columns:
        raise ValueError(f"{csv_in} already has a TWIN_ID column.")

    ids = df[id_column].str.strip()
    ids.iloc[0] = "TWIN_ID"
    ids.iloc[1] = '{"ImportId":"TWIN_ID"}'
    df.insert(0, "TWIN_ID", ids)

    body = ids.iloc[2:]
    counts = Counter(v for v in body if v)
    report = {
        "rows": len(body),
        "duplicates": sorted(v for v, n in counts.items() if n > 1),
        "blank": [int(i) for i, v in body.items() if not v],
        "non_numeric": sorted({v for v in body if v and not v.isdigit()}),
    }

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    return report


def filter_document(document: Any, keep_question: Any) -> dict[str, Any]:
    """Deep-copy a response document, keeping only some questions.

    Structure is preserved (containers, block names, order); blocks and
    containers left empty are dropped. Used both here and by
    :mod:`kli.sjt_split`.

    Args:
        document: A response document.
        keep_question: ``callable(question_dict) -> bool``.

    Returns:
        A new document in ``{"Metadata", "Elements", "EmbeddedData"}`` form.
    """

    def _filter(elements: list[Any]) -> list[Any]:
        out: list[Any] = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            if element.get("ElementType") == "Block":
                questions = [
                    q
                    for q in element.get("Questions", [])
                    if isinstance(q, dict) and keep_question(q)
                ]
                if questions:
                    block = {k: v for k, v in element.items() if k != "Questions"}
                    block["Questions"] = copy.deepcopy(questions)
                    out.append(block)
            elif "Elements" in element:
                children = _filter(element["Elements"])
                if children:
                    container = {k: v for k, v in element.items() if k != "Elements"}
                    container["Elements"] = children
                    out.append(container)
        return out

    base = document if isinstance(document, dict) else {}
    return {
        "Metadata": copy.deepcopy(base.get("Metadata", {})),
        "Elements": _filter(elements_of(document)),
        "EmbeddedData": copy.deepcopy(base.get("EmbeddedData", {})),
    }


def combine_documents(
    documents: dict[str, Any],
    *,
    exclude_question_ids: Iterable[str] = DEFAULT_ID_QUESTION_IDS,
) -> dict[str, Any]:
    """Join one participant's per-wave documents into a single document.

    Args:
        documents: Wave name to response document, in wave order.
        exclude_question_ids: QIDs dropped from every wave (the ID question).

    Returns:
        A document whose ``Elements`` are every Block from every wave, flat and
        in order, each carrying ``"Wave": <wave name>``.

    Raises:
        ValueError: If a QID appears in more than one wave after exclusions.
            Two answers under one QID would make columns and round-trip checks
            ambiguous, so this is refused rather than resolved silently.
    """
    excluded = set(exclude_question_ids)
    elements: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    embedded: dict[str, Any] = {}
    seen: dict[str, str] = {}

    for wave, document in documents.items():
        filtered = filter_document(
            document, lambda q: q.get("QuestionID") not in excluded
        )
        for ref in walk_questions(filtered):
            if ref.question_id in seen:
                raise ValueError(
                    f"{ref.question_id} appears in both {seen[ref.question_id]} "
                    f"and {wave}. Add it to exclude_question_ids if it is "
                    "administrative, or rename it in one survey."
                )
            seen[ref.question_id] = wave
        for block in _blocks(filtered["Elements"]):
            elements.append({**block, "Wave": wave})
        metadata[wave] = filtered["Metadata"]
        embedded[wave] = filtered["EmbeddedData"]

    return {"Metadata": {"Waves": metadata}, "Elements": elements, "EmbeddedData": embedded}


def _blocks(elements: list[Any]) -> Iterable[dict[str, Any]]:
    """Yield every Block beneath ``elements``, depth-first, in order."""
    for element in elements:
        if element.get("ElementType") == "Block":
            yield element
        elif "Elements" in element:
            yield from _blocks(element["Elements"])


@dataclass
class CombineReport:
    """What :func:`combine_wave_dirs` found. Written as ``combine_report.json``."""

    waves: list[str]
    participants_per_wave: dict[str, int]
    combined: int = 0
    #: Participant ID to the waves they are missing from. A typed-ID typo shows
    #: up here as two half-present participants.
    incomplete: dict[str, list[str]] = field(default_factory=dict)
    require_all_waves: bool = True


def combine_wave_dirs(
    wave_dirs: dict[str, str | Path],
    output_dir: str | Path,
    *,
    exclude_question_ids: Iterable[str] = DEFAULT_ID_QUESTION_IDS,
    require_all_waves: bool = True,
) -> CombineReport:
    """Combine per-wave response JSON directories into one per participant.

    Args:
        wave_dirs: Wave name to that wave's ``response_json`` directory, in
            wave order (YAML mappings keep their order).
        output_dir: Where combined ``pid_<id>_response.json`` files go.
        exclude_question_ids: QIDs dropped from every wave.
        require_all_waves: If True (default), participants missing any wave are
            left out and listed in the report. If False they are combined from
            the waves they have.

    Returns:
        A :class:`CombineReport`, also written to ``combine_report.json``.

    Raises:
        FileNotFoundError: If a wave directory holds no response JSONs.
    """
    output_dir = Path(output_dir)
    files: dict[str, dict[str, Path]] = {}
    for wave, directory in wave_dirs.items():
        found = {
            pid: path
            for path in sorted(Path(directory).glob("*.json"))
            if (pid := extract_pid(path)) is not None
        }
        if not found:
            raise FileNotFoundError(f"{wave}: no pid_*.json files in {directory}")
        files[wave] = found

    waves = list(wave_dirs)
    report = CombineReport(
        waves=waves,
        participants_per_wave={w: len(files[w]) for w in waves},
        require_all_waves=require_all_waves,
    )
    everyone = sorted(set().union(*(files[w] for w in waves)))
    output_dir.mkdir(parents=True, exist_ok=True)

    for pid in everyone:
        missing = [w for w in waves if pid not in files[w]]
        if missing:
            report.incomplete[pid] = missing
            if require_all_waves:
                continue
        documents = {
            w: json.loads(files[w][pid].read_text(encoding="utf-8"))
            for w in waves
            if pid in files[w]
        }
        combined = combine_documents(
            documents, exclude_question_ids=exclude_question_ids
        )
        (output_dir / f"pid_{pid}_response.json").write_text(
            json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report.combined += 1

    (output_dir / "combine_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    return report


def _column_names(document: Any) -> dict[str, str]:
    """Map each QID to its export tag, where that tag is unique."""
    tags = {
        ref.question_id: str(ref.question.get("QuestionName") or "")
        for ref in walk_questions(document, include_descriptive=False)
    }
    counts = Counter(t for t in tags.values() if t)
    return {qid: tag for qid, tag in tags.items() if tag and counts[tag] == 1}


def _rename(column: str, names: dict[str, str]) -> str:
    qid, sep, rest = column.partition("_")
    return names[qid] + sep + rest if qid in names else column


def combined_dataframe(
    combined_dir: str | Path, *, labels: bool = False, use_export_tags: bool = True
) -> pd.DataFrame:
    """Build the final one-row-per-participant table from combined documents.

    Args:
        combined_dir: Output of :func:`combine_wave_dirs`.
        labels: Emit selected label text instead of numeric positions.
        use_export_tags: Name columns ``SJT1_3`` rather than ``QID1101_3``.

    Returns:
        DataFrame indexed by ``TWIN_ID``, columns in survey order.
    """
    rows: dict[str, dict[str, Any]] = {}
    order: dict[str, None] = {}
    for path in sorted(Path(combined_dir).glob("pid_*.json")):
        pid = extract_pid(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        values, _ = extract_columns(document, labels=labels)
        if use_export_tags:
            names = _column_names(document)
            values = {_rename(c, names): v for c, v in values.items()}
        rows[str(pid)] = values
        order.update(dict.fromkeys(values))
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(order))
    frame.index.name = "TWIN_ID"
    return frame


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", help="Study YAML config.")
    parser.add_argument(
        "--add-twin-id",
        nargs=2,
        metavar=("RAW_CSV", "OUT_CSV"),
        help="Copy one wave's Qualtrics export, adding TWIN_ID from StudentID, "
        "then exit. Run once per wave before get_responses.py.",
    )
    args = parser.parse_args(argv)

    if args.add_twin_id:
        report = add_twin_id(*args.add_twin_id)
        print(json.dumps(report, indent=2))
        problems = report["duplicates"] or report["blank"] or report["non_numeric"]
        if problems:
            print(
                "WARNING: fix these IDs before ingestion. get_responses.py keeps "
                "only the first row of a duplicate ID and needs numeric IDs."
            )
        return 1 if problems else 0
    if not args.config:
        parser.error("--config is required unless --add-twin-id is given")

    config = load_config(args.config)
    block = get(config, "combine_waves")
    output_dir = Path(get(config, "combine_waves.output_dir"))
    report = combine_wave_dirs(
        dict(get(config, "combine_waves.waves")),
        output_dir,
        exclude_question_ids=block.get(
            "exclude_question_ids", list(DEFAULT_ID_QUESTION_IDS)
        ),
        require_all_waves=bool(block.get("require_all_waves", True)),
    )
    print(
        f"combined {report.combined} participant(s) across {len(report.waves)} "
        f"waves {report.participants_per_wave}"
    )
    if report.incomplete:
        print(
            f"WARNING: {len(report.incomplete)} participant(s) missing from at "
            "least one wave (a mistyped StudentID looks like this): "
            f"{report.incomplete}"
        )

    for labels, name in ((False, "values"), (True, "labels")):
        frame = combined_dataframe(output_dir, labels=labels)
        target = Path(block.get("csv_output_dir", output_dir)) / f"combined_human_{name}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target)
        print(f"wrote {target} ({frame.shape[0]} rows x {frame.shape[1]} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
