# kli-digital-twins

A working skeleton for building and evaluating LLM digital twins from Qualtrics
survey data, assembled from the two public Twin-2K-500 repositories.

**Neither upstream repository contains a complete pipeline.** Their gaps are
complementary, and this repository joins them: Qualtrics ingestion and
evaluation from the [Mega-Study repo](https://github.com/TianyiPeng/Twin-2K-500-Mega-Study),
twin construction from the [digital-twin-simulation repo](https://github.com/tianyipeng-lab/digital-twin-simulation).
See [`ATTRIBUTION.md`](ATTRIBUTION.md) for file-by-file provenance and commits.

> **Read this before running anything.** Parts of this pipeline work out of the
> box, parts need a known and bounded amount of work, and parts fail *silently*
> on data they weren't designed for. The sections "What you still have to
> write" and "Inherited defects" are not optional reading — the silent failures
> are the ones that will corrupt a study without raising an error.

---

## Status at a glance

| Stage | Source | Status |
|---|---|---|
| 1. QSF → survey template JSON | mega-study | ✅ Works |
| 2. Qualtrics CSV → per-participant response JSON | mega-study | ✅ Works, with caveats |
| 3. Persona JSON → persona text | demo | ⚠️ Works on HF-format input; needs an adapter for ours |
| 4. Question JSON → prompt | demo | ⚠️ **Needs the schema adapter** |
| 5. Persona + question → final prompt | demo | ⚠️ Works; no empty-persona arm |
| 6. LLM execution | demo | ⚠️ Works for OpenAI; config schema differs |
| 7. LLM output → response JSON | demo | ⚠️ **Needs the schema adapter** |
| 8. Response JSON → analysis CSV | *neither* | ❌ **Not implemented** |
| 9. Metrics and meta-analysis | mega-study | ✅ Works, ~40 lines per study |

---

## Context: what the upstream work found

The mega study reports that digital twins correlate with their human
counterparts at an **average of r = 0.20** across 164 outcomes, and are *"only
modestly more accurate than those of a homogeneous base LLM."* It names five
systematic distortions: insufficient individuation, stereotyping,
representation bias, ideological bias, and hyper-rationality.

Those twins were built on ~500 questions per person across four waves. If your
persona substrate is thinner than that, treat r = 0.20 as a **ceiling
estimate, not a floor**. This pipeline is a benchmark harness, not a method
known to work well — and the control arms below are what make that testable.

---

## Pipeline

```
  your_survey.qsf ──[1: parse_qsf.py]──────────────► survey_flow.json
                                                            │
  your_responses.csv ─[2: get_responses.py]◄────────────────┘
                           │
                           ▼
                 pid_<id>_response.json          ← per participant
                           │
   persona JSON ─[3: batch_convert_personas.py]──► persona .txt
                           │
                           ├─[4: convert_question_json_to_text.py]──► question prompts
                           │
                           └─[5: create_text_simulation_input.py]───► combined prompts
                                                                            │
                                         [6: run_LLM_simulations.py]────────┘
                                                      │
                                         [7: postprocess_responses.py]
                                                      │
                                                      ▼
                                      response_json_llm_imputed/
                                                      │
                                    [8: convert_responses_to_csv.py]   ← NOT IMPLEMENTED
                                                      │
                                                      ▼
                                  [9: mega_study_evaluation/]  →  metrics
```

A persona is structurally just a filled-in survey response file — the same
block/question/answer schema as stage 2's output. So your personas come from
running your own Qualtrics data through stages 1–2 and nothing else.

---

## Setup

```bash
git clone <this repo> && cd kli-digital-twins
poetry install

# API key
echo "OPENAI_API_KEY=sk-..." > .env

# Optional: the upstream Twin-2K-500 dataset, for calibration against
# published numbers. Not needed to run on your own data.
poetry run python download_dataset.py
```

Scaffold a study:

```bash
poetry run python scripts/setup_study.py \
    --study my_study \
    --qsf  path/to/your_survey.qsf \
    --csv  path/to/your_responses.csv \
    --dry_run     # prints the config it would write; drop this to commit it
```

Then run the two stages that work today:

```bash
poetry run python processing_qualtrics_qsf/parse_qsf.py  --config configs/my_study/my_study.yaml
poetry run python processing_qualtrics_csv/get_responses.py --config configs/my_study/my_study.yaml
```

Inspect the output before going further. See "Validate your ingestion" below —
this is the step most likely to fail quietly.

---

## What you still have to write

Four items, in dependency order. Together they are a few days of work, not a
research project.

### 1. The schema adapter (highest priority, highest risk)

The seam between the two upstream repos does not currently join.

- **What stage 2 produces:** `pid_<id>_response.json` — a **dict** of
  `{Metadata, Elements, EmbeddedData}`, where `Elements` nests `Block`,
  `Branch`, `Randomizer`, and `Group` containers.
- **What stages 4 and 7 expect:** a flat **top-level list of Blocks**. See
  `text_simulation/convert_question_json_to_text.py:281`
  (`for element in data: ## data is a list of blocks`) and
  `text_simulation/postprocess_responses.py:235`.

Handed your files, those two scripts iterate dict keys and raise
`AttributeError`.

The recursion you need **already exists in this repo**:
`text_simulation/convert_persona_to_text.py:233–238` handles the
`{"Elements": ...}` dict form, and lines 187–188 recurse into `Branch`. It was
simply never propagated to the question path.

> **⚠️ Do this as ONE shared traversal function, used by both the prompt
> builder and the postprocessor.**
>
> The binding between prompt and response is **purely positional**. The prompt
> builder numbers questions `Q1..Qn` as it walks the JSON; the postprocessor
> re-walks the same structure with its own counter and looks up
> `response["Q{count}"]`. There is no question-ID round trip. If the two
> traversals ever diverge — a Randomizer handled differently, a skipped
> descriptive block — then **every answer after the divergence is silently
> misassigned to the wrong question.** No error is raised. This is the single
> most dangerous property of the vendored code.
>
> Fixing it properly means emitting the question ID alongside each `Qn` and
> asserting the round-trip on parse. Do that and you remove a whole class of
> silent corruption.

Also parameterize the hardcoded filename patterns while you are in there:
`*_wave4_Q_wave4_A.json` at `convert_question_json_to_text.py:316` and
`postprocess_responses.py:278`, and the `pid_\d+` regexes at
`run_LLM_simulations.py:121`, `create_text_simulation_input.py:27`, and
`postprocess_responses.py:321` (which require purely numeric participant IDs).

### 2. `text_simulation/convert_responses_to_csv.py`

Exists in neither upstream repo. The full specification is in that file's
docstring. In brief: read the imputed response JSONs, derive the target column
schema from your *human* response CSV's ImportId header row, and emit
`consolidated_llm_values.csv` and `consolidated_llm_labels.csv` — the filenames
`mega_study_evaluation/common/data_loader.py` expects.

Do **not** reuse `evaluation/column_mapping.csv` for this. Those 42 rows are
specific to the upstream wave-4 instrument and are meaningless for your survey.

### 3. Question-type coverage

`text_simulation/convert_persona_to_text.py::format_question_text` raises
`ValueError` on anything outside `MC`, `Matrix`, `TE`, `Slider`, `DB`. The
Qualtrics parser in stage 1 also emits `CS` (constant sum), and will crash on
rank-order items for an unrelated reason (see below).

Each new type needs three things, and the third is easy to forget:

1. a `_format_question_text_<Type>` formatter in `convert_persona_to_text.py`
2. a validator in `postprocess_responses.py::validation_functions`
3. a response-schema block in **both** copies of
   `convert_question_json_to_text.py::format_instructions` — the
   `include_reasoning=True` and `False` variants are a copy-paste fork, not a
   parameterization, so edits must be made twice

Note an existing instance of exactly this bug: the prompt schema documents four
types but the formatter emits five labels, and `"Multiple Choice"` (multi-select)
has **no schema and no validator**. Multi-select questions therefore fail
validation 100% of the time and burn every retry. Fix that first if your
instrument has any.

This is the natural point to refactor into the `question_formatters/` package
the `Snakefile` expects.

### 4. The empty-persona control arm

`empty_persona_prompt` has no code path anywhere in the vendored code.
`create_text_simulation_input.py:57` requires a matching persona file per
participant ID, so there is no way to emit a persona-free prompt.

This is ~15 lines, and **it is the most important 15 lines in the project.**
Without the empty-persona and demographics-only controls you cannot distinguish
"the twin captures this individual" from "the LLM captures this population" —
which, given the upstream finding that twins barely beat a persona-free
baseline, is the claim a reviewer will press hardest. Write it before you run
anything at scale.

---

## Inherited defects

Verified in the vendored code. The silent ones matter most.

### Silent — no error, wrong data

**Question-based display logic always evaluates to `False`.**
`processing_qualtrics_csv/get_responses.py:139` calls
`eval_display_logic(question["DisplayLogic"], embedded_data_dict)` with two
arguments. The `response_data` parameter defaults to `None`, and
`util/logic_eval.py` returns `False` for every `LogicType == "Question"`
condition when `response_data is None`. **Every conditionally-displayed
question disappears from every participant's file**, whether or not they saw
and answered it. In the upstream corpus that is ~543 of 649 display-logic
conditions, and two of their own study configs carry hand-written workarounds
for it (nulling `DisplayLogic` via `revise_questions` plus a
`keep_questions_when_answer_missing` list). **If your instrument branches at
all, this will hollow out your data.**

**Question-based branch logic resolves against a key that never matches.** The
QSF stores `LeftOperand` as a locator URI (`q://QID62/SelectableChoice/5`)
while the response dict is keyed `QID62`, `QID62_5`, `QID62_TEXT`. The lookup
returns `None` and the branch is taken or not by operator accident —
`Selected` against a null right operand evaluates `None == None → True`, so
the branch is always taken.

**Four operators Qualtrics emits are unsupported** and degrade silently to
`==`: `NotEmpty` (the code implements `IsNotEmpty`), `NotDisplayed`,
`MatchesRegex`, `Is`.

**Partial Matrix and Slider responses produce misaligned arrays.** Constant-sum
and text-form extractors pad missing cells with `None` to preserve positional
alignment; Matrix and Slider do not. On a partial response `Answers.Values` is
shorter than `Statements`, with no index to recover the alignment.

**Failed questions retain the human answer, unflagged.**
`postprocess_responses.py` writes the output file *before* returning failure,
and a question that failed parsing or validation keeps its original `Answers`
value. On held-out data that is ground-truth leakage into a file labeled as
model output. Add an explicit per-question imputation flag.

**Text cleaning loses content.** HTML entities are not unescaped on the main
path (`&rsquo;`, `&amp;`, `&mdash;` leak through), adjacent inline tags join
without spaces (`"Source: BBCNews"`), and the table handler keeps only `<th>`
headers plus the **first** `<tr>` of the body — discarding all other rows and
any prose outside the table.

**`Metadata` is always `{}`.** The producer writes the key `"metadata"`; the
consumer tests for `"survey_metadata"` (`parse_qsf.py:185` vs `:282`).

**Randomized loop order is invented, not recovered.** `looping.py` calls bare,
unseeded `random.shuffle()` when loop randomization is enabled, then discards
the invented order. Dormant unless your survey uses randomized loop-and-merge.

### Loud — will stop you immediately

**The participant ID column is hardcoded.** `PID_COLUMN = "TWIN_ID"` at
`processing_qualtrics_csv/get_responses.py:37` is a module constant, not a
config value. A CSV without a literal `TWIN_ID` column raises `KeyError`.
Qualtrics' own `ResponseId` and `PROLIFIC_PID` are not accepted. Either add a
`TWIN_ID` column to your export or change that line. IDs must also be
numeric — `load_csv` normalizes via `str(int(x))` and raises otherwise.

**Rank-order questions crash the parser.** `parse_qsf.py:239` dispatches on
`QuestionType == "Rank"`, but Qualtrics emits `"RO"`. Unknown types `raise
ValueError` rather than warn-and-skip, so one rank-order item aborts the whole
run. Workaround: list the QID under `exclude_questions`. Side-by-side (`SBS`)
items have a guaranteed `KeyError`; column-selector variants `SACOL`/`MACOL`
also raise.

**`limit` cannot be used for subsampling.** `get_responses.py` resets the
dataframe index, sorts without resetting again, then breaks on the *original*
index label. `limit: 5` yields one file. **Use `-1` only.**

**Gemini is broken.** `text_simulation/llm_helper.py` calls `genai.Client()`
and `types.FinishReason`, which belong to the `google-genai` SDK, but
`pyproject.toml` pins `google-generativeai`. Worse, `provider` **defaults to
`'gemini'`** (`run_LLM_simulations.py:215`), so omitting it yields an
`AttributeError` swallowed into an error dict. Set `provider` explicitly.

**Only `openai` and `gemini` providers exist.** `llm_helper.py:130–136` raises
on anything else. No OpenRouter, no `own_server`. The `llm-batch-helper` PyPI
package adds OpenRouter and Together but **also** lacks `own_server`, which 95
upstream configs require — implying the mega study ran a forked, vendored copy
of that library that was never published.

**`max_tokens` vs `max_completion_tokens`.** `llm_helper.py:63` passes
`max_tokens`, which gpt-5 and the o-series reject. Those models also reject
`temperature` values other than 1.

**JSON parsing is one `split("```json")`.** No `json_repair`, no brace
matching, no trailing-comma tolerance — and the Slider example in the prompt
schema itself contains a trailing comma, i.e. invalid JSON offered to the model
as an exemplar. Harden this early; retries cost real money.

**In-place config mutation.** The upstream `run_pipeline.sh` edited
`text_simulation/configs/openai_config.yaml` with a BSD-only `sed -i ''`. Not
vendored here, but the notebooks in the demo repo do the same.

---

## Validate your ingestion

Upstream has **no test coverage** for either Qualtrics subsystem: no `tests/`,
no schema validation of outputs, no reconciliation that every CSV ImportId was
consumed, no report of dropped questions. The only test is a six-case
self-check inside `logic_eval.py`, and all six exercise `EmbeddedField` —
which is exactly why the display-logic defect above went unnoticed.

**Write a reconciliation step before trusting any output.** Roughly thirty
lines, reporting per participant:

- how many questions were dropped, and why — display-logic-false vs. no
  matching answer vs. unsupported type
- which ImportIds in the CSV were never consumed
- which questions have `Answers` arrays whose length disagrees with their
  `Rows` / `Statements` / `Options`

This single check surfaces most of the silent failures listed above. Do it
first, and cite it in your methods appendix.

---

## Evaluation

`mega_study_evaluation/common/` is fully study-agnostic and reusable as-is.
`mega_study_evaluation/_template_study/` is the cleanest per-study module
(141 lines, vendored from their `accuracy_nudges`). Copy it and edit six things:
`study_name`; the parallel `DV_vars` / `DV_vars_min` / `DV_vars_max` lists; the
seven domain flag lists; `condition_vars`; optionally a composite-DV
constructor; optionally the input filenames.

There is **no base class and no enforced interface** — the contract is
convention, and violating it causes your study to be *silently dropped* from
aggregation. Hard requirements:

1. Write `meta analysis.csv` — the orchestrator checks this exact filename.
2. Write `meta analysis individual level.csv` with exactly the columns
   `TWIN_ID, variable_name, value, respondent_type, study_name, specification_name`,
   `respondent_type ∈ {human, twin}`.
3. Your `study_name` must match the **display** name used in your DV range
   table, or range lookups miss and accuracy silently becomes `NaN`.

In return you get correlation with Fisher-z CIs, MAD accuracy, paired *t*,
variance *F*, Cohen's *d* per arm, plus `post_metric_calculation`'s
tail-conditioned accuracies, variance ratio, Glass's Δ, and a seeded random
benchmark.

### Metric definitions, as implemented

Accuracy is a range-normalized MAD complement, with **table-driven** scale
bounds (not data-derived):

```
accuracy = 1 − mean|human − twin| / (scale_max − scale_min)
```

Tail-conditioned accuracy is the individuation diagnostic: respondents are
sorted by their *human* value and accuracy is recomputed on the bottom 5%, top
5%, and middle 90%, always with the full-scale denominator. Variance
compression is `std(twin)/std(human)`. Mean shift is Glass's Δ, stored as an
absolute value — direction is lost in the output file.

Correlations are Fisher-z averaged weighted by `n_points`; everything else is
arithmetically averaged.

### Statistical caveats to carry into your appendix

- **No multiple-comparison correction exists anywhere** in the upstream stack.
  With many DVs × many specifications, the per-DV *p*-values are raw.
- **The variance test is the independent-samples *F*-test applied to paired
  data.** It ignores the human–twin correlation and is anticonservative. Use a
  Pitman–Morgan test if variance compression is a claim you make.
- **No bootstrapping.** All intervals are parametric: Student-*t* on accuracy,
  Fisher-z on correlations, Wilson on proportions.

---

## Design decisions worth preserving

Three things upstream got right, independent of their code quality:

**The holdout with a reliability ceiling.** Personas are waves 1–3 with every
item that reappears in wave 4 removed; ground truth is the wave-4 question set
filled with the respondent's *earlier* answers; and the benchmark is the same
person's *actual* later answers to those items. The headline number is twin
accuracy **relative to human test–retest reliability** — how much of the
achievable signal was captured. A twin-accuracy figure without a within-person
reliability ceiling is uninterpretable.

**The factorial arm matrix.** Full persona / summary / demographics-only /
empty persona × reasoning × temperature × model. Arms differ by exactly one
substantive config line each; everything else is path re-rooting so arms never
overwrite one another. See `configs/README.md`.

**Shipping the config with the results.** Every upstream results directory
contains the exact YAML that produced it. Keep this habit — it is why their
unpublished simulation layer is reconstructible at all.

---

## Reproducibility statement for the paper

Suggested language, adjust as needed:

> Our pipeline adapts two public implementations released alongside the
> Twin-2K-500 dataset (Toubia et al., arXiv:2505.17479) and the Twin-2K-500
> Mega-Study (Peng et al., arXiv:2509.19088). Qualtrics survey ingestion and
> the evaluation stack derive from the Mega-Study repository; persona
> rendering, prompt assembly, and LLM execution derive from the
> digital-twin-simulation repository. The Mega-Study repository's
> `text_simulation/` module — the twin-construction layer invoked by its
> workflow definition — is not included in its public release; we therefore
> use the earlier published implementation of that layer and reconstruct the
> remaining components from the model and prompt parameters recorded in the
> Mega-Study's released configuration files. Commit hashes for both upstream
> sources, and a file-by-file statement of provenance and modification, are
> recorded in our repository. We note that no multiple-comparison correction is
> applied in the upstream metric stack, and that upstream reports an average
> human–twin correlation of r = 0.20 on a persona substrate of ~500 items per
> participant.

---

## Suggested first week

1. Add a `TWIN_ID` column to your Qualtrics export. Run stages 1–2.
2. Write the reconciliation check. **Look at what it reports** — especially
   dropped conditional questions.
3. Write the shared traversal adapter. Confirm prompt and parse agree on a
   handful of participants by hand.
4. Add the `empty_persona_prompt` branch.
5. Run stages 3–7 on ~20 participants with a cheap model. Read the actual
   prompts and the actual model output as text before scaling.
6. Write `convert_responses_to_csv.py`.
7. Copy `_template_study/`, define your DVs, and get one metric end to end.

Scale only after step 5 has been eyeballed by a human. The positional-binding
defect and the display-logic defect both produce output that looks completely
normal.

---

## License

Apache 2.0, inherited from both upstream repositories. See [`LICENSE`](LICENSE),
[`NOTICE`](NOTICE), and [`ATTRIBUTION.md`](ATTRIBUTION.md).

No participant data and no dataset files are included in this repository.
`.gitignore` excludes `data/`, `**/raw_data/`, and pipeline outputs — keep it
that way.
