"""Flatten LLM-imputed response JSONs into a wide CSV aligned to the human response CSV.

=============================================================================
STATUS: NOT IMPLEMENTED — THIS IS A SPECIFIED STUB.
=============================================================================

This is the one pipeline stage that exists in NEITHER upstream repository.

The Twin-2K-500 Mega-Study `Snakefile` defines a `convert_to_csv` rule that
invokes `text_simulation/convert_responses_to_csv.py`, and all 479 study
configs carry the `text_simulation.convert_to_csv` config block that this
script is supposed to read. The script itself was never published, and the
companion `digital-twin-simulation` repo has no equivalent.

Until this is written, the pipeline stops after `postprocess_responses.py`:
you will have per-participant imputed JSON, but nothing that the evaluation
layer can join against human data.

-----------------------------------------------------------------------------
REQUIRED BEHAVIOR
-----------------------------------------------------------------------------

Read the config block (see `configs/example_study.yaml`):

    text_simulation:
      convert_to_csv:
        input_json_dir: <dir of LLM-imputed response JSONs>
        reference_csv_path: <the HUMAN Qualtrics response CSV>
        output_dir: <where to write the twin CSV>

Then:

1. Derive the target column schema from `reference_csv_path`, NOT from a
   fixed mapping file. The reference CSV is a standard Qualtrics export with
   three header rows:
       row 0 — machine column names
       row 1 — human-readable question text
       row 2 — JSON metadata, e.g. {"ImportId":"QID1529"}
   Build {ImportId -> column_name} from row 2, exactly as
   `processing_qualtrics_csv/get_responses.py::get_import_ids_from_csv` does
   (note the `choiceId` suffix convention: {"ImportId":"QID140","choiceId":"3"}
   becomes the key `QID140_3`).

   Do NOT reuse `evaluation/column_mapping.csv` for this. That file is 42 rows
   hardcoded to the Twin-2K-500 wave-4 instrument and is meaningless for any
   other survey.

2. Walk each imputed response JSON and emit one row per participant.
   `evaluation/json2csv.py::AnswerExtractor` already implements this walk and
   the per-question-type column naming — reuse it rather than reimplementing.
   Its naming contract:
       MC single          -> {QID}                 = SelectedByPosition (1-based)
       MC multi           -> {QID}_{choice_id}     = 1 for selected only
       Matrix             -> {QID}_{item_id}       = SelectedByPosition[i]
       Slider (multi)     -> {QID}_{item_id}       = float(Values[i])
       Slider (single)    -> {QID}                 = float(Values[0])
       Constant sum       -> {QID}_{item_id}       = float(Values[i])
       Text entry single  -> {QID}_TEXT            = str
       Text entry form    -> {QID}_{item_id}       = str

   IMPORTANT: `AnswerExtractor` must be given the SAME traversal order used to
   build the prompt. See the warning about positional binding in the README.

3. Map each extracted column through {ImportId -> reference column name} and
   write the result in the reference CSV's own column order. Columns present
   in the extraction but absent from the reference schema must be preserved
   (append them, sorted) rather than silently dropped — losing a DV here is
   invisible downstream.

4. Emit the participant ID column under the same name the reference CSV uses
   (upstream this is hardcoded as `TWIN_ID`; see README "Participant ID").

5. Write two files to `output_dir`:
       consolidated_llm_values.csv   — numeric/positional values
       consolidated_llm_labels.csv   — the selected label text
   These filenames are what `mega_study_evaluation/common/data_loader.py`
   expects. The human-side counterpart it joins against is
   `consolidated_original_answers_values.csv`.

-----------------------------------------------------------------------------
VALIDATION THIS SHOULD PERFORM (upstream performs none)
-----------------------------------------------------------------------------

* Assert that every ImportId in the reference CSV is either written or
  explicitly accounted for, and report the ones that are not.
* Report, per participant, how many questions produced no value and why.
* Refuse to write a row whose participant ID is missing or non-unique.
* Carry through the per-question imputation flag described in the README, so
  that questions where the LLM failed and the original human answer was
  retained are identifiable in the output. Upstream does not do this, and the
  result is silent ground-truth leakage.
"""

import argparse
import sys


def convert_responses_to_csv(config_path: str) -> None:
    """Convert LLM-imputed response JSONs to a reference-aligned CSV.

    Args:
        config_path: Path to the study YAML config.

    Raises:
        NotImplementedError: Always. See the module docstring for the spec.
    """
    raise NotImplementedError(
        "convert_responses_to_csv is not implemented. This stage exists in "
        "neither upstream repository and must be written. See the module "
        "docstring in this file for the full specification, and the README "
        "section 'What you still have to write'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten LLM-imputed response JSONs into a reference-aligned CSV."
    )
    parser.add_argument("--config", required=True, help="Path to the study YAML config.")
    args = parser.parse_args()

    print(
        "ERROR: text_simulation/convert_responses_to_csv.py is a specified stub, "
        "not a working script.\n"
        "       Read the module docstring for what it must do, then implement it.\n",
        file=sys.stderr,
    )
    convert_responses_to_csv(args.config)
