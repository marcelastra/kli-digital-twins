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

| Stage | Implementation | Source | Status |
|---|---|---|---|
| 1. QSF to survey template JSON | `processing_qualtrics_qsf/` | mega-study | Vendored, works |
| 1b. **Validate ingestion** | `kli/reconcile.py` | **KLI** | **Original** |
| 2. Qualtrics CSV to response JSON | `processing_qualtrics_csv/` | mega-study | Vendored, works |
| 3. Persona JSON to persona text | `text_simulation/` | demo | Vendored |
| 4. Question JSON to prompt | `kli/render_questions.py` | **KLI** | **Original** |
| 5. Persona + question to prompt | `kli/assemble_prompts.py` | **KLI** | **Original**, incl. control arm |
| 6. Local model execution | `kli/providers/local.py` | **KLI** | **Original** |
| 7. Model output to response JSON | `kli/parse_responses.py` | **KLI** | **Original** |
| 8. Response JSON to analysis CSV | `kli/responses_to_csv.py` | **KLI** | **Original** |
| 9. Metrics and meta-analysis | `mega_study_evaluation/` | mega-study | Vendored, ~40 lines per study |

**Everything original is under `kli/` and `tests/`.** Nothing in `kli/` imports
from `text_simulation/`, so the authorship boundary is enforced by the code's own
structure rather than only by documentation. Every original file carries an
`ORIGINAL WORK (Kravis Leadership Institute)` line in its module docstring.
[`ATTRIBUTION.md`](ATTRIBUTION.md) has the file-by-file table for your appendix.

### What we had to write, and why

Four gaps in the upstream releases, in the order they bite.

**The schema adapter** (`kli/survey_tree.py`). The two upstream repos describe a
survey differently: the Qualtrics stage emits a dict whose `Elements` nest
`Branch`, `Randomizer` and `Group` containers, while the twin-construction code
expects a flat top-level list of blocks. More seriously, upstream numbers
questions `Q1..Qn` in one walk and then re-walks independently to parse answers,
with no question-ID round trip, so any divergence silently attaches every later
answer to the wrong question. Here there is exactly one traversal, both sides use
it, and every answer's echoed `QuestionID` is verified against the question
actually asked.

**The no-persona control** (`kli/assemble_prompts.py`). Upstream's configs all
carry `empty_persona_prompt`, but no code reads it, and
`create_text_simulation_input.py:57` requires a matching persona file per
participant, which makes a persona-free prompt unreachable by construction.
Without this arm you cannot separate "the model captured this individual" from
"the model captured the population average". Our implementation defaults to
reproducing Peng et al.'s wording exactly, and makes the alternative a one-line
config change — see "Control-arm wording" below, which is also where an
unreported and cheap result is described.

**The analysis CSV** (`kli/responses_to_csv.py`). Exists in neither repository.

**A local model provider** (`kli/providers/local.py`). Upstream is OpenAI-first.
Participant survey responses stay on this machine.

### Verified against real data

Developed and tested against genuine Qualtrics output, not synthetic fixtures.
Running stages 1-2 over three upstream studies produced **1,654 real
per-participant response files** (`story_beliefs` 251, `accuracy_nudges` 1,003,
`junk_fees` 400), spanning flat documents and documents nested three levels deep
through Randomizer, Group and Branch containers, and covering all six question
types the parser emits. The traversal's invariants hold across all three, the
renderer handles every question in all three, and a stub-model run reaches 100%
imputation end to end through to the analysis CSV. **81 tests pass**
(`python -m pytest tests/ -v`); upstream has no test coverage for any of these
stages.

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
  your_survey.qsf --[1: parse_qsf.py]-------------> survey_flow.json
                                                          |
  your_responses.csv -[2: get_responses.py]<--------------+
                           |
                           +--[1b: kli.reconcile]--> what got dropped, and why
                           v
                 pid_<id>_response.json            <- per participant
                           |
   persona .txt -[3: batch_convert_personas.py]----> persona text
                           |
                           +--[4: kli.render_questions]--> prompts + JSON skeleton
                           |
                           +--[5: kli.assemble_prompts]--> final prompts
                                                            (or no-persona control)
                                        |
                       [6: kli.providers.local]  <-- Mistral, on this machine
                                        |
                       [7: kli.parse_responses]  <-- verified, flagged
                                        v
                        response_json_llm_imputed/
                                        |
                       [8: kli.responses_to_csv]
                                        v
                      consolidated_llm_values.csv  ->  [9: mega_study_evaluation/]
```

Stages 4 through 8 run from one command (`kli.run_simulation`), then stage 8
separately. A persona is structurally just a filled-in survey response file, so
your personas come from running your own Qualtrics data through stages 1-2.

---

## Setup

```bash
git clone <this repo> && cd kli-digital-twins
poetry install
python -m pytest tests/ -q          # 81 tests, no model or network needed
```

### The local model

```bash
# install Ollama from https://ollama.com, then:
ollama pull mistral        # ~4.1 GB, Mistral 7B Instruct
ollama serve               # usually already running

# confirm this pipeline can see it:
python -m kli.run_simulation --config configs/example_study_local.yaml --health-check
```

Nothing leaves your machine. `kli/providers/local.py` resolves the configured
endpoint and refuses to send anything unless every resolved address is loopback.
Lifting that requires `allow_remote_endpoint: true` in the config, which is
recorded in every run manifest. Note the scope of the claim: it constrains this
pipeline's model calls, not your whole system, and it cannot stop a local server
that has itself been configured to proxy elsewhere. For an auditable guarantee,
block outbound traffic at the host level too.

### Run a study

```bash
# 1-2. ingest your Qualtrics export
python processing_qualtrics_qsf/parse_qsf.py     --config configs/my_study.yaml
python processing_qualtrics_csv/get_responses.py --config configs/my_study.yaml

# 1b. ALWAYS DO THIS. Reports questions silently dropped during ingestion.
#     Exits 1 if it finds suspected display-logic loss, so it can gate a script.
python -m kli.reconcile --config configs/my_study.yaml

# 3. render personas (vendored; variant: full | summary | demographic)
python text_simulation/batch_convert_personas.py \
    --persona_json_dir data/my_study/response_json \
    --output_text_dir  data/my_study/persona_text \
    --variant full

# 4-7. build prompts, call the local model, write imputed JSON.
#      Work up in this order: look at the prompts, then prove the plumbing
#      offline, then run small, then run everything.
python -m kli.run_simulation --config configs/my_study.yaml --dry-run --limit 3
python -m kli.run_simulation --config configs/my_study.yaml --stub-model --limit 3
python -m kli.run_simulation --config configs/my_study.yaml --limit 20
python -m kli.run_simulation --config configs/my_study.yaml

# 8. analysis CSV, aligned to your human export
python -m kli.responses_to_csv --config configs/my_study.yaml
```

`--stub-model` fills the response skeleton from the instrument itself, with no
model and no network. A stub run should reach **100% imputation**; anything less
is a bug in this pipeline rather than a limitation of the model. It is a test of
plumbing, never of accuracy.

### Run both arms

```bash
# treatment: empty_persona_prompt: false
python -m kli.run_simulation --config configs/my_study.yaml

# control:   empty_persona_prompt: true, outputs re-rooted elsewhere
python -m kli.run_simulation --config configs/my_study_control.yaml
```

Same participants, same questions, same model, persona withheld. The question
section of the prompt is byte-identical between arms.

#### Control-arm wording, and an open question worth a paragraph in the paper

`control_wording` has two settings. The default, `upstream`, reproduces what
Peng et al. actually did.

**What they did.** Verified across all 76 of their released `empty_persona`
configs: the only substantive difference from the matching persona config is
`empty_persona_prompt: false -> true`. Every prompt string is left in place.
Their control arm ran with the system instruction *"answer the 'New Survey
Question' as if you are the person described in the 'Persona Profile' (which
consists of their past survey responses)"*, the separator *"Please respond as
the persona described above"*, and `persona_prompt_header` still supplied. Their
control was told to imitate a person described in a profile that was not there.

One detail is not recoverable: whether that header line was actually *printed*
when the body was empty. The config supplies the string; the code that decided
is the unpublished file. Worth asking the authors.

**Why `upstream` is our default.** Exactly one variable differs between arms, so
any outcome difference is attributable to the persona alone, and your control
numbers stay comparable to the published ones.

**Why it is arguably a confound.** A model instructed to imitate someone it was
never shown is handicapped, not merely uninformed. If that depresses control
performance, the twin-minus-control gap is inflated.

**Where that cuts is the interesting part.** Peng et al. report that twins are
*"only modestly more accurate than those of a homogeneous base LLM."* If their
control was handicapped by incoherent instructions, their headline is
**conservative** — the true twin advantage over a fairly-prompted baseline would
be smaller still. Their negative finding survives the criticism and is
strengthened by it.

**So the difference between the two wordings is itself measurable.** Running the
control arm both ways tells you how much of the "base LLM" benchmark is prompt
artifact rather than model capability. As far as we can tell nobody has reported
this. It is one extra arm on data you already have:

```bash
# control, upstream wording (default) — comparable to the paper
python -m kli.run_simulation --config configs/my_study_control.yaml

# control, neutral wording — add `control_wording: neutral` under LLM_simulation
python -m kli.run_simulation --config configs/my_study_control_neutral.yaml
```

Under `upstream` the run prints a note recording the choice, so it cannot be
discovered only by reading source six months later. Under `neutral`, an explicit
`system_instruction` in the config that still mentions a persona raises a warning
— that combination means the label and the behavior have come apart. Either way
the exact strings are written into `run_manifest.json`.

The full argument lives in `kli/assemble_prompts.py` under "control-arm
wording".

### What each run leaves behind

`run_manifest.json`, beside the imputed JSON: model, temperature, arm, chunk
size, whether the endpoint was loopback, the full system instruction and
separator, the imputation rate, and per-participant errors. Cite it rather than
reconstructing settings from memory.

`imputation_report.csv` and `conversion_report.json`, from stage 8: per
participant, how many questions the model actually answered; and which reference
columns were never filled.

---

## What you still have to write

The four gaps are closed. What remains is study-specific.

**Question types beyond the six.** `kli/render_questions.py` covers `MC`
(single and multi), `Matrix`, `Slider`, `TE` (SL/ML/ESTB/FORM), `CS`, and `DB`
as context. Rank-order (`RO`) and side-by-side (`SBS`) are not supported, and
they crash the *vendored QSF parser* before they ever reach the renderer. If
your instrument has them, either list the QIDs under `exclude_questions` or add
support in three places: a branch in `describe_question`, a branch in
`answer_skeleton`, and a validator in `kli/parse_responses.py::validate_answers`.
`UnsupportedQuestionType` names all three when it fires.

**Your DVs for the evaluation layer.** Copy `mega_study_evaluation/_template_study/`
and edit six things — see the Evaluation section.

**A DV range table** if you want `post_metric_calculation` metrics: rows of
`(study_name, DV_variable, DV_min, DV_max)`, joined on the *display* study name.

**Prompt wording for your population.** The defaults in
`kli/assemble_prompts.py` are reasonable but generic. If you change them, change
them for both arms and record it.

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

**Multi-select questions produce no columns at all through `json2csv.py`.**
`evaluation/json2csv.py::_process_mc_multiple_numeric` reads
`answer_data["Selected"]`, but `processing_qualtrics_csv/question_types/mc_question_extractor.py`
writes `SelectedByPosition` and `SelectedText`. The key never matches, so every
multi-select answer is dropped, silently, for humans and twins alike. This is
why `kli/responses_to_csv.py` does its own extraction and reads both field
names. If you use `json2csv.py` directly for anything, check whether your
instrument has multi-selects first.

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

Neither upstream repo has test coverage for the Qualtrics subsystems: no
`tests/`, no schema validation of outputs, no reconciliation that every CSV
column was consumed, no report of dropped questions. `save_output` is a bare
`json.dump`. The only test for that layer is a six-case self-check inside
`logic_eval.py`, and all six exercise `EmbeddedField` -- which is precisely why
the display-logic defect went unnoticed.

`kli/reconcile.py` closes that gap. Run it after stage 2, every time:

```bash
python -m kli.reconcile --config configs/my_study.yaml
```

It compares the parsed survey template against the per-participant files and
reports how many participants retained each question. A question that is in the
template, carries display logic, and survives for *nobody* is the signature of
the display-logic defect. It exits 1 when it finds that pattern, so you can gate
a pipeline script on it.

**How bad this actually is.** Running it against upstream's own `junk_fees`
study with their hand-written workaround removed: **49 of 117 questions -- 42%
of the instrument -- disappear for all 400 participants**, with no error and no
warning. That is what the defect costs when nobody is checking. The workaround
is per-question and goes in the config:

```yaml
qsf_to_json:
  revise_questions:
    QID123: {DisplayLogic: null}
csv_to_fill_answers:
  keep_questions_when_answer_missing: ["QID123"]
```

`kli.reconcile` prints exactly this snippet, pre-filled with your QIDs.

Note what the workaround does and does not do: it stops the question being
deleted, but it does not reconstruct who should have seen it. If a question was
genuinely only shown to some participants, you still need to decide what a
missing answer means for that subgroup, and say so in your methods.

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

Suggested language. Adjust, but keep the four disclosures: what was vendored,
what was written here, that the models ran locally, and the upstream benchmark.

> Our pipeline combines two public implementations released alongside the
> Twin-2K-500 dataset (Toubia et al., arXiv:2505.17479) and the Twin-2K-500
> Mega-Study (Peng et al., arXiv:2509.19088). Qualtrics survey ingestion and
> the evaluation stack derive from the Mega-Study repository; persona rendering
> derives from the digital-twin-simulation repository. The Mega-Study
> repository's `text_simulation/` module -- the twin-construction layer invoked
> by its workflow definition -- is not included in its public release.
>
> We therefore implemented the following components ourselves: a canonical
> survey traversal shared by prompt construction and response parsing, with
> question-identifier verification on every answer; question rendering with an
> explicit response schema, extended to constant-sum items; prompt assembly
> including a no-persona control condition; a local model provider; response
> parsing with per-question validation and explicit imputation flags; and the
> conversion of simulated responses to an analysis dataset, a stage absent from
> both public releases. These components, and the tests covering them, are
> identified file by file in our repository.
>
> All model inference was performed locally (Mistral 7B Instruct via Ollama) on
> institutional hardware. No participant survey responses were transmitted to
> any third-party service at any stage. Our implementation refuses non-loopback
> model endpoints by default, and each run records whether that constraint was
> in force.
>
> Two methodological notes. First, no multiple-comparison correction is applied
> in the upstream metric stack, and we report [corrected / descriptive]
> statistics accordingly. Second, upstream reports an average human-twin
> correlation of r = 0.20 on a persona substrate of roughly 500 items per
> participant, which we treat as an upper reference point rather than a
> target.

If you used the display-logic workaround, add a sentence naming the affected
questions and what a missing answer means for them. If you set
`allow_remote_endpoint: true` at any point, the no-egress sentence above is not
accurate as written and must be revised.

---

## Suggested first week

1. Add a `TWIN_ID` column to your Qualtrics export. Run stages 1-2.
2. **Run `kli.reconcile` and read the output.** If it reports suspected
   display-logic loss, fix that before anything else -- everything downstream
   inherits it.
3. `ollama pull mistral`, then `--health-check`.
4. `--dry-run --limit 3` and **read an actual prompt end to end.** Does the
   persona read like a person? Is the question legible? Is the JSON skeleton
   right? This is the cheapest bug-catching in the whole project.
5. `--stub-model --limit 3`. Expect 100% imputation. Anything less is a bug
   here, not a model limitation.
6. `--limit 20` against the real model. Read the model's actual answers as text.
   Check the imputation rate in the manifest: if a question type is failing
   systematically, tune `questions_per_call` down before scaling.
7. Run both arms at `--limit 20` and compare. If the twin arm does not visibly
   differ from the no-persona arm on your DVs at small N, find out why now
   rather than after 2,000 participants.
8. Stage 8, then copy `_template_study/`, define your DVs, and get one metric
   end to end.

Scale only after step 6 has been read by a human. The positional-binding hazard
and the display-logic defect both produce output that looks entirely normal --
the first is now structurally prevented and the second is now detected, but your
own additions will not be.

---

## License

Apache 2.0, inherited from both upstream repositories. See [`LICENSE`](LICENSE),
[`NOTICE`](NOTICE), and [`ATTRIBUTION.md`](ATTRIBUTION.md).

No participant data and no dataset files are included in this repository.
`.gitignore` excludes `data/`, `**/raw_data/`, and pipeline outputs — keep it
that way.
