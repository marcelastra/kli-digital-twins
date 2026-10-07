"""Flatten response JSONs into CSVs aligned to the human Qualtrics export.

ORIGINAL WORK (Kravis Leadership Institute). See ATTRIBUTION.md.

-------------------------------------------------------------------------------
THE STAGE THAT EXISTS IN NEITHER UPSTREAM REPOSITORY
-------------------------------------------------------------------------------

The Mega-Study ``Snakefile`` defines a ``convert_to_csv`` rule invoking
``text_simulation/convert_responses_to_csv.py``, and all 479 released study
configs carry the ``text_simulation.convert_to_csv`` block. The script was
never published, and the companion repo has no equivalent. Without it you have
per-participant imputed JSON and no way to score it.

**Why this does not reuse ``evaluation/json2csv.py``.** That module is bound to
the Twin-2K-500 wave-4 instrument: it maps columns through
``evaluation/column_mapping.csv``, 42 rows specific to that survey. It also has
a genuine bug for our purposes -- ``_process_mc_multiple_numeric`` reads
``answer_data["Selected"]``, but ``processing_qualtrics_csv/`` writes
``SelectedByPosition``/``SelectedText`` for multi-select. Routed through it,
**every multi-select question yields no columns at all**, silently. This module
derives its schema from *your* reference CSV and reads both field names.

-------------------------------------------------------------------------------
HOW THE SCHEMA IS DERIVED
-------------------------------------------------------------------------------

A Qualtrics export has three header rows::

    row 0: Q1, Q2_1, Q2_2, ...                       machine column names
    row 1: "In your opinion, is the headline..."      question text
    row 2: {"ImportId":"QID1529"}, ...                JSON metadata

Row 2 gives ``{ImportId -> column_name}``. Qualtrics names sub-items
``QID123_4``, which is exactly how :func:`extract_columns` names them, so the
join is structural rather than a hand-maintained mapping.

-------------------------------------------------------------------------------
OUTPUTS
-------------------------------------------------------------------------------

``consolidated_llm_values.csv``
    Numeric/positional values, two header rows (names, then question text), to
    match the shape ``mega_study_evaluation/common/data_loader.py`` reads with
    ``skiprows=[1, 2]``.

``consolidated_llm_labels.csv``
    The selected label text, same layout.

``imputation_report.csv``
    Per participant: questions total, imputed, failed, and the failure reasons.
    Upstream has no equivalent, and without it an incomplete run is
    indistinguishable from a complete one.

``conversion_report.json``
    Schema reconciliation: which ImportIds were matched, which reference
    columns were never filled, which extracted columns had no home.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from kli.assemble_prompts import extract_pid
from kli.survey_tree import walk_questions

__all__ = [
    "read_reference_schema",
    "extract_columns",
    "convert_directory",
    "ConversionReport",
]

#: Qualtrics export columns that are metadata rather than responses. Used only
#: to keep the "never filled" diagnostic meaningful -- a twin is not expected to
#: produce a browser string or a page timer, so listing those as unfilled would
#: bury the columns that genuinely are missing.
_METADATA_HINTS = (
    "StartDate",
    "EndDate",
    "Status",
    "IPAddress",
    "Progress",
    "Duration",
    "Finished",
    "RecordedDate",
    "ResponseId",
    "RecipientLastName",
    "RecipientFirstName",
    "RecipientEmail",
    "ExternalReference",
    "LocationLatitude",
    "LocationLongitude",
    "DistributionChannel",
    "UserLanguage",
    "Browser",
    "Version",
    "Operating System",
    "Resolution",
    "Q_Recaptcha",
    "Q_Language",
    "Q_Ballot",
    "Consent",
    "PROLIFIC",
    "random",
    "Random",
    "att_check",
)

#: Column-name fragments marking Qualtrics instrumentation: display order
#: (``_DO_``), click timing, and page timers.
_METADATA_FRAGMENTS = (
    "_DO_",
    "_DO",
    "_RT_",
    "First Click",
    "Last Click",
    "Page Submit",
    "Click Count",
    "Timing",
)


def _is_metadata_column(column: str) -> bool:
    """True if a reference column is instrumentation rather than a response."""
    lowered = column.lower()
    if any(hint.lower() in lowered for hint in _METADATA_HINTS):
        return True
    return any(fragment.lower() in lowered for fragment in _METADATA_FRAGMENTS)


class ConversionReport(dict):
    """Schema reconciliation results, serializable to JSON."""


def read_reference_schema(
    reference_csv_path: str | Path,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Read the human export's column schema.

    Args:
        reference_csv_path: The human Qualtrics response CSV.

    Returns:
        ``(columns, importid_to_column, column_to_description)``.

    Raises:
        ValueError: If no ImportId metadata row can be located. Rather than
            guess, this fails -- a wrong schema would silently misalign every
            column.
    """
    reference_csv_path = Path(reference_csv_path)
    head = pd.read_csv(
        reference_csv_path, nrows=6, header=None, dtype=str, low_memory=False
    )
    columns = [str(c) for c in head.iloc[0].tolist()]

    import_row_index: int | None = None
    for row_index in range(1, min(len(head), 6)):
        row = head.iloc[row_index].astype(str)
        looks_like_json = row.str.strip().str.startswith("{") & row.str.contains(
            "ImportId", na=False
        )
        if looks_like_json.mean() > 0.3:
            import_row_index = row_index
            break

    if import_row_index is None:
        raise ValueError(
            f"Could not find the ImportId metadata row in {reference_csv_path}. "
            "This module needs the standard Qualtrics export with header rows "
            "(Data & Analysis -> Export -> CSV, 'Download all fields' with "
            "headers). A re-saved or single-header CSV will not work: aligning "
            "columns without ImportIds would be guesswork."
        )

    importid_to_column: dict[str, str] = {}
    for column, cell in zip(columns, head.iloc[import_row_index].tolist()):
        if not isinstance(cell, str) or "ImportId" not in cell:
            continue
        try:
            meta = json.loads(cell)
        except json.JSONDecodeError:
            continue
        import_id = meta.get("ImportId")
        if not import_id:
            continue
        choice_id = meta.get("choiceId")
        key = f"{import_id}_{choice_id}" if choice_id else str(import_id)
        importid_to_column[key] = column

    description_row = import_row_index - 1
    descriptions: dict[str, str] = {}
    if description_row >= 1:
        for column, cell in zip(columns, head.iloc[description_row].tolist()):
            descriptions[column] = "" if pd.isna(cell) else str(cell)

    return columns, importid_to_column, descriptions


def _item_ids(question: dict[str, Any], count: int) -> list[str]:
    """Resolve per-item IDs, falling back to 1-based indices.

    The QSF parser omits ``RowsID``/``StatementsID`` when they equal the trivial
    order, so both shapes must be handled.
    """
    for key in ("StatementsID", "RowsID"):
        ids = question.get(key)
        if isinstance(ids, list) and len(ids) == count:
            return [str(i) for i in ids]
    return [str(i) for i in range(1, count + 1)]


def extract_columns(
    document: Any, *, labels: bool = False
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Flatten one response document into ``{column: value}``.

    Column naming matches Qualtrics' own ImportId convention, which is what
    makes the reference-schema join work:

    ==================  ==========================  ========================
    Question            Column(s)                   Value
    ==================  ==========================  ========================
    MC single           ``{QID}``                   1-based option position
    MC multi            ``{QID}_{position}``        1 per selected option
    Matrix              ``{QID}_{item_id}``         1-based column position
    Slider (1 item)     ``{QID}``                   number
    Slider (n items)    ``{QID}_{item_id}``         number
    Constant sum        ``{QID}_{item_id}``         number
    Text entry          ``{QID}_TEXT``              string
    Text entry (form)   ``{QID}_{item_id}``         string
    ==================  ==========================  ========================

    Args:
        document: A response document (human or imputed).
        labels: If True, emit selected label text instead of positions.

    Returns:
        ``(values, imputed_flags)`` where ``imputed_flags`` maps each emitted
        column to whether the model produced it.
    """
    values: dict[str, Any] = {}
    flags: dict[str, bool] = {}

    for ref in walk_questions(document, include_descriptive=False):
        question = ref.question
        answers = question.get("Answers")
        imputed = bool(question.get("LLM_Imputed", False))
        if not isinstance(answers, dict):
            continue

        qid = ref.question_id
        qtype = ref.question_type
        selector = str(question.get("Settings", {}).get("Selector", "") or "").upper()

        def emit(column: str, value: Any) -> None:
            values[column] = value
            flags[column] = imputed

        if qtype == "MC":
            # Read both field names: processing_qualtrics_csv writes
            # SelectedByPosition, while evaluation/json2csv.py expects
            # "Selected". Tolerating both keeps human and twin files readable
            # by the same code.
            positions = answers.get("SelectedByPosition")
            if positions is None:
                positions = answers.get("Selected")
            texts = answers.get("SelectedText")

            if selector in {"MAVR", "MAHR", "MACOL"} or isinstance(positions, list):
                position_list = positions if isinstance(positions, list) else []
                text_list = texts if isinstance(texts, list) else []
                for index, position in enumerate(position_list):
                    if position is None:
                        continue
                    if labels:
                        text = text_list[index] if index < len(text_list) else None
                        emit(f"{qid}_{position}", text)
                    else:
                        emit(f"{qid}_{position}", 1)
            elif positions is not None:
                emit(qid, texts if labels else positions)

        elif qtype == "Matrix":
            positions = answers.get("SelectedByPosition") or []
            texts = answers.get("SelectedText") or []
            if not isinstance(positions, list):
                positions = [positions]
            ids = _item_ids(question, len(positions))
            for index, position in enumerate(positions):
                if position is None:
                    continue
                if labels:
                    text = texts[index] if index < len(texts) else None
                    emit(f"{qid}_{ids[index]}", text)
                else:
                    emit(f"{qid}_{ids[index]}", position)

        elif qtype in {"Slider", "CS"}:
            raw_values = answers.get("Values") or []
            if not isinstance(raw_values, list):
                raw_values = [raw_values]
            if qtype == "Slider" and len(raw_values) == 1:
                statements = question.get("Statements") or []
                if len(statements) <= 1:
                    emit(qid, raw_values[0])
                    continue
            ids = _item_ids(question, len(raw_values))
            for index, value in enumerate(raw_values):
                if value is None:
                    continue
                emit(f"{qid}_{ids[index]}", value)

        elif qtype == "TE":
            text = answers.get("Text")
            if selector == "FORM" and isinstance(text, list):
                ids = _item_ids(question, len(text))
                for index, item in enumerate(text):
                    if item is None:
                        continue
                    emit(f"{qid}_{ids[index]}", item)
            elif text is not None:
                emit(f"{qid}_TEXT", text)

    return values, flags


def _build_frame(
    records: list[dict[str, Any]],
    reference_columns: list[str],
    importid_to_column: dict[str, str],
    pid_column: str,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Map extracted columns onto the reference schema."""
    matched: set[str] = set()
    unmapped: set[str] = set()
    rows: list[dict[str, Any]] = []

    for record in records:
        pid = record.pop("__pid__")
        row: dict[str, Any] = {pid_column: pid}
        for column, value in record.items():
            target = importid_to_column.get(column)
            if target is not None:
                row[target] = value
                matched.add(target)
            else:
                # Keep it under its extraction name rather than dropping it.
                # A silently discarded DV is the worst possible failure here.
                row[column] = value
                unmapped.add(column)
        rows.append(row)

    ordered = [c for c in reference_columns if c in matched or c == pid_column]
    if pid_column not in ordered:
        ordered.insert(0, pid_column)
    extras = sorted(unmapped)
    frame = pd.DataFrame(rows)
    for column in ordered + extras:
        if column not in frame.columns:
            frame[column] = None
    return frame[ordered + extras], matched, unmapped


def _write_with_headers(
    frame: pd.DataFrame,
    descriptions: dict[str, str],
    path: Path,
) -> None:
    """Write a CSV with the two header rows the evaluation layer expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    description_row = pd.DataFrame(
        [{c: descriptions.get(c, "") for c in frame.columns}]
    )
    importid_row = pd.DataFrame([{c: "" for c in frame.columns}])
    combined = pd.concat(
        [description_row, importid_row, frame], ignore_index=True
    )
    combined.to_csv(path, index=False, na_rep="")


def convert_directory(
    *,
    input_json_dir: str | Path,
    reference_csv_path: str | Path,
    output_dir: str | Path,
    pid_column: str = "TWIN_ID",
    filename_glob: str = "*.json",
) -> ConversionReport:
    """Convert a directory of response JSONs into aligned CSVs.

    Args:
        input_json_dir: Directory of imputed response JSONs.
        reference_csv_path: The human Qualtrics export, used as target schema.
        output_dir: Where to write the CSVs and reports.
        pid_column: Participant ID column name, matching the reference export.
        filename_glob: Which files to read.

    Returns:
        A :class:`ConversionReport`, also written as ``conversion_report.json``.

    Raises:
        FileNotFoundError: If no input files match.
    """
    input_json_dir = Path(input_json_dir)
    output_dir = Path(output_dir)
    paths = sorted(input_json_dir.glob(filename_glob))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {filename_glob!r} in {input_json_dir}"
        )

    reference_columns, importid_to_column, descriptions = read_reference_schema(
        reference_csv_path
    )

    value_records: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    imputation_rows: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    skipped: list[dict[str, str]] = []

    for path in paths:
        pid = extract_pid(path)
        if pid is None:
            skipped.append({"file": path.name, "reason": "no pid in filename"})
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            skipped.append({"file": path.name, "reason": f"unreadable: {exc}"})
            continue

        values, flags = extract_columns(document, labels=False)
        labels, _ = extract_columns(document, labels=True)
        values["__pid__"] = pid
        labels["__pid__"] = pid
        value_records.append(values)
        label_records.append(labels)

        total = imputed = 0
        for ref in walk_questions(document, include_descriptive=False):
            total += 1
            if ref.question.get("LLM_Imputed"):
                imputed += 1
            else:
                status = str(ref.question.get("LLM_Status", "not imputed"))
                reason_counter[status.split(":")[0].strip()] += 1
        imputation_rows.append(
            {
                pid_column: pid,
                "questions_total": total,
                "questions_imputed": imputed,
                "questions_failed": total - imputed,
                "complete": total > 0 and imputed == total,
            }
        )

    if not value_records:
        raise FileNotFoundError(
            f"No usable response files in {input_json_dir}; {len(skipped)} skipped."
        )

    values_frame, matched, unmapped = _build_frame(
        value_records, reference_columns, importid_to_column, pid_column
    )
    labels_frame, _, _ = _build_frame(
        label_records, reference_columns, importid_to_column, pid_column
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_with_headers(
        values_frame, descriptions, output_dir / "consolidated_llm_values.csv"
    )
    _write_with_headers(
        labels_frame, descriptions, output_dir / "consolidated_llm_labels.csv"
    )

    imputation_frame = pd.DataFrame(imputation_rows)
    imputation_frame.to_csv(output_dir / "imputation_report.csv", index=False)

    response_columns = {
        c
        for c in reference_columns
        if not _is_metadata_column(c) and c != pid_column
    }
    never_filled = sorted(response_columns - matched)

    report = ConversionReport(
        {
            "input_dir": str(input_json_dir),
            "reference_csv": str(reference_csv_path),
            "participants_written": int(len(values_frame)),
            "files_skipped": skipped,
            "reference_columns_total": len(reference_columns),
            "reference_columns_matched": len(matched),
            "reference_response_columns_never_filled": never_filled,
            "extracted_columns_without_reference_match": sorted(unmapped),
            "imputation": {
                "questions_total": int(imputation_frame["questions_total"].sum()),
                "questions_imputed": int(imputation_frame["questions_imputed"].sum()),
                "participants_complete": int(imputation_frame["complete"].sum()),
                "failure_reasons": dict(reason_counter.most_common()),
            },
        }
    )
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def summarize(report: ConversionReport) -> str:
    """Render a conversion report as a short human-readable summary."""
    lines = [
        f"participants written      : {report['participants_written']}",
        f"reference columns matched : {report['reference_columns_matched']}"
        f" / {report['reference_columns_total']}",
    ]
    never = report["reference_response_columns_never_filled"]
    if never:
        lines.append(
            f"reference response columns never filled : {len(never)} "
            f"(first few: {', '.join(never[:5])})"
        )
    extras = report["extracted_columns_without_reference_match"]
    if extras:
        lines.append(
            f"extracted columns with no reference match : {len(extras)} "
            f"(kept under their own names; first few: {', '.join(extras[:5])})"
        )
    imputation = report["imputation"]
    if imputation["questions_total"]:
        rate = imputation["questions_imputed"] / imputation["questions_total"]
        lines.append(
            f"imputation rate           : {rate:.1%} "
            f"({imputation['questions_imputed']}/{imputation['questions_total']})"
        )
    if imputation["failure_reasons"]:
        lines.append(f"failure reasons           : {imputation['failure_reasons']}")
    return "\n".join(lines)


def _skipped_note(skipped: Iterable[dict[str, str]]) -> str:
    items = list(skipped)
    return f"{len(items)} file(s) skipped" if items else "no files skipped"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Reads ``text_simulation.convert_to_csv`` from the study config, so the same
    config drives this stage as every other.
    """
    import argparse

    from kli.config import get, load_config

    parser = argparse.ArgumentParser(
        description="Flatten imputed response JSONs into CSVs aligned to the "
        "human Qualtrics export."
    )
    parser.add_argument("--config", required=True, help="Study YAML config.")
    parser.add_argument(
        "--pid-column",
        default="TWIN_ID",
        help="Participant ID column name in the reference export "
        "(default: TWIN_ID, matching upstream).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    base = "text_simulation.convert_to_csv"
    report = convert_directory(
        input_json_dir=get(config, f"{base}.input_json_dir"),
        reference_csv_path=get(config, f"{base}.reference_csv_path"),
        output_dir=get(config, f"{base}.output_dir"),
        pid_column=args.pid_column,
        filename_glob="pid_*.json",
    )
    print(summarize(report))
    print(f"\nwritten to {get(config, f'{base}.output_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
