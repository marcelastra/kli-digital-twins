# Study configuration contract

One YAML file per study per specification arm. Generated from
`configs/study_template.yaml.j2` by `scripts/setup_study.py`, or written by hand.

A worked baseline is in `configs/example_study.yaml` (vendored from the
mega-study's `accuracy_nudges.yaml`). Note that its internal paths still read
`data/accuracy_nudges/...` and must be edited for your study.

> This file is newly written. The upstream mega-study repo's `configs/README.md`
> is not documentation — despite the `.md` extension it is a misfiled YAML
> fragment for the `story_beliefs` study. It is not vendored here.

## Which keys are actually read

The vendored Qualtrics stages read their config blocks as documented below.
**The `text_simulation` blocks are a different matter:** the vendored
`text_simulation/` scripts come from the demo repo, which uses flat top-level
keys and individual CLI flags rather than this nested tree. The nested schema
below is the mega-study's contract — the target to adapt the scripts toward,
not what they read today. See the README's "What you still have to write".

| Block | Read by | Status |
|---|---|---|
| `qsf_to_json` | `processing_qualtrics_qsf/parse_qsf.py` | **Works as documented** |
| `csv_to_fill_answers` | `processing_qualtrics_csv/get_responses.py` | **Works as documented** |
| `text_simulation.personas_to_texts` | `text_simulation/batch_convert_personas.py` | CLI flags only |
| `text_simulation.questions_to_texts` | `text_simulation/convert_question_json_to_text.py` | CLI flags only |
| `text_simulation.create_text_simulation_input` | `text_simulation/create_text_simulation_input.py` | CLI flags only; prompts hardcoded |
| `text_simulation.LLM_simulation` | `text_simulation/run_LLM_simulations.py` | Reads `--config` but **flat** keys |
| `text_simulation.convert_to_csv` | `text_simulation/convert_responses_to_csv.py` | **Not implemented** |

---

## `qsf_to_json`

Converts the Qualtrics `.qsf` survey definition into a structured survey
template JSON.

| Key | Required | Meaning |
|---|---|---|
| `input_file` | yes | Path to the `.qsf` |
| `output_file` | yes | Survey template JSON to write |
| `exclude_blocks` | no | List of block **names** (Descriptions) to drop — consent, intros, attention checks. `null` for none |
| `exclude_questions` | no | List of QuestionIDs to drop. **This is your escape hatch for unsupported question types** (see README) |
| `remove_question_text` | no | `{QID: substring}` — removes that substring from the raw HTML before stripping |
| `rename_blocks` | no | `{old_name: new_name}` |
| `revise_questions` | no | `{QID: {payload_key: value}}` — overrides arbitrary payload keys. `value: null` **deletes** the key. Upstream uses this to null out `DisplayLogic` and to replace image-based question text with prose transcriptions |
| `exclude_flow_elements` | no | List of `FlowID`s; drops the whole subtree |

`revise_questions` is undocumented upstream but heavily used. Two mega-study
configs (`junk_fees`, `promiscuous_donors`) use it to work around the display-logic
defect described in the README.

## `csv_to_fill_answers`

Fills the survey template with each participant's answers from the Qualtrics
CSV export, producing one JSON per participant.

| Key | Required | Meaning |
|---|---|---|
| `input_survey_template` | yes | The `qsf_to_json.output_file` |
| `input_csv_file` | yes | Raw Qualtrics response export (the 3-header-row "Download Data → CSV" format) |
| `output_dir` | yes | Where to write `pid_<id>_response.json` |
| `limit` | no | Max participants. **Use `-1` only** — any other value is broken upstream (see README) |
| `load_embedded_data` | no | `[{csv_field: {rename: new_name}}, ...]` — pre-seed embedded data from CSV columns |
| `keep_questions_when_answer_missing` | no | List of QIDs to retain even with no matching answer. Needed wherever display logic has been nulled out |

## `text_simulation.personas_to_texts`

| Key | Meaning |
|---|---|
| `persona_json_dir` | Directory of persona JSONs |
| `output_text_dir` | Where to write rendered persona `.txt` files |
| `persona_variant` | `full` in all 479 upstream configs. The vendored script accepts `full`, `demographic`, `summary`, `summary+text`, `combine` |

## `text_simulation.questions_to_texts`

| Key | Meaning |
|---|---|
| `input_path` | Directory of per-participant response JSONs |
| `output_dir` | Where to write question prompt `.txt` files |
| `include_reasoning` | `true` asks the model for a rationale alongside each answer. This is a distinct experimental arm |

## `text_simulation.create_text_simulation_input`

| Key | Meaning |
|---|---|
| `persona_text_dir` | **The persona-richness arm selector.** Point at the full text, the summary text, or a demographics-only directory |
| `question_prompts_dir` | Output of the previous stage |
| `output_combined_prompts_dir` | Where to write assembled prompts |
| `persona_prompt_header` | Text preceding the persona |
| `persona_question_prompt_separator` | Text between persona and question |
| `empty_persona_prompt` | `true` = **no-persona control arm**; the persona body is omitted entirely |
| `combine_prompt_end` | Optional trailing text. Inert upstream — resolves to `null` in all 114 configs that set it |

## `text_simulation.LLM_simulation`

| Key | Meaning |
|---|---|
| `input_dir` | Assembled prompts |
| `output_dir` | Raw LLM responses |
| `question_json_base_dir` | The response JSONs to fill in |
| `output_updated_questions_dir` | Imputed question JSONs — what the evaluation layer consumes |
| `system_instruction` | The persona-adoption system prompt |
| `LLM_config.provider` | `openai`, `openrouter`, `own_server` |
| `LLM_config.model_name` | e.g. `gpt-4.1`, or a fine-tuned checkpoint ID |
| `LLM_config.temperature` | `0.0` for the deterministic arm, `0.7` baseline |
| `LLM_config.max_tokens` | Upstream uses `16384` |
| `LLM_config.max_retries` | Upstream uses `10` |
| `num_workers` | Parallel request workers (upstream: `300`) |
| `force_regenerate` | `false` reuses existing outputs |
| `max_personas` | `-1` for all |

## `text_simulation.convert_to_csv`

| Key | Meaning |
|---|---|
| `input_json_dir` | The imputed question JSONs |
| `reference_csv_path` | The **human** response CSV, used as the target column schema |
| `output_dir` | Where to write the aligned twin CSVs |

**No implementation exists.** See `text_simulation/convert_responses_to_csv.py`
for the full specification.

---

## Specification arms

Upstream encodes experimental arms as a config-filename suffix, with
dimensions at their default value **omitted from the name**. So
`temperature_zero` alone means *full persona, no reasoning, T=0, gpt-4.1*.
Results directories are named `<spec_type>_<YYYY-MM-DD>`.

The arms differ from the baseline in exactly one substantive line each:

| Arm | The one line that changes |
|---|---|
| `empty_persona` | `empty_persona_prompt: true` |
| `persona_summary` | `persona_text_dir: data/mega_persona_summary_text` |
| `demographics_only` | `persona_text_dir: data/personas_demographics_only_text` |
| `with_reasoning` | `include_reasoning: true` |
| `temperature_zero` | `LLM_config.temperature: 0.0` |
| model arms | `LLM_config.provider` + `model_name` |

Everything else in each diff is path re-rooting so that arms do not overwrite
one another. Keep that convention: it is the cleanest part of the upstream
design, and it is what makes an individuation claim falsifiable.
