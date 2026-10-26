# Attribution and provenance

Every file in this repository is either copied verbatim from one of two
upstream repositories, or newly written here. This file records which.

Both upstream repositories are licensed **Apache License 2.0**, which permits
redistribution and modification provided the license, attribution, and a
statement of changes are preserved. See `LICENSE` and `NOTICE`.

## Upstream sources

| Short name | Repository | Commit | Date | Associated paper |
|---|---|---|---|---|
| **mega-study** | [`TianyiPeng/Twin-2K-500-Mega-Study`](https://github.com/TianyiPeng/Twin-2K-500-Mega-Study) | `afe2bb933fce377ed196f441a4c12962cb55a53a` | 2026-06-08 | Peng, Gui, Brucks, Merlau, Fan, Ben Sliman, Johnson et al., *"Digital Twins as Funhouse Mirrors: Five Key Distortions"*, [arXiv:2509.19088](https://arxiv.org/abs/2509.19088) |
| **demo** | [`tianyipeng-lab/digital-twin-simulation`](https://github.com/tianyipeng-lab/digital-twin-simulation) | `f1eed510c9a4fb47aaa9bb46e178aaa2d9224623` | 2025-08-04 | Toubia, Gui, Peng, Merlau, Li & Chen, *"Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions"*, [arXiv:2505.17479](https://arxiv.org/abs/2505.17479) |

The associated dataset is [`LLM-Digital-Twin/Twin-2K-500`](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), licensed CC BY 4.0. No dataset files are vendored here.

## Why two sources

Neither upstream repository contains a complete pipeline, and the gaps are
complementary:

| | Qualtrics ingestion | Twin construction | Evaluation | Study configs |
|---|---|---|---|---|
| mega-study | yes | **absent** | yes | yes |
| demo | **absent** | yes | partial | no |

The mega-study repo's `text_simulation/` directory is referenced by its
`Snakefile` and documented in its README but was never committed — verified
against all branches, tags, and full history, and it is not covered by that
repo's `.gitignore`. The demo repo contains a working version of that layer,
but has no Qualtrics handling at all (zero occurrences of "qualtrics" or
".qsf" anywhere in it).

## File-by-file provenance

### Copied verbatim from **mega-study**

```
processing_qualtrics_qsf/            # QSF -> survey template JSON
  parse_qsf.py
  question_types/{base,constant_sum,matrix,multiple_choice,rank,side_by_side,slider,text_entry}.py
  flow_elements_types/{parse_branch_element,parse_embedded_data_element,parse_group_element,parse_randomizer_element}.py

processing_qualtrics_csv/            # CSV + template -> per-participant response JSON
  get_responses.py
  question_types/{constant_sum,looping,matrix_question_extractor,mc_question_extractor,slider_question_extractor,te_question_extractor}.py
  util/logic_eval.py

evaluation/
  json2csv.py                        # AnswerExtractor: response JSON -> flat dataframe
  evaluation_basic.yaml              # NOTE: wave globs are Twin-2K-500-specific
  column_mapping.csv                 # NOTE: 42 rows specific to their wave-4 instrument

mega_study_evaluation/
  __init__.py
  parse_specification.py
  common/{__init__,args_parser,data_loader,results_processor,stats_analysis,variable_mapper}.py
  _template_study/                   # copied from their accuracy_nudges/ (cleanest template)

post_metric_calculation/
  compute_vector_metrics.py

configs/
  study_template.yaml.j2
  example_study.yaml                 # copied from their configs/accuracy_nudges/accuracy_nudges.yaml

scripts/setup_study.py
Snakefile
download_dataset.py
LICENSE
```

### Copied verbatim from **demo**

```
text_simulation/
  convert_persona_to_text.py         # persona JSON -> persona text (5 variants)
  batch_convert_personas.py          # batch driver
  convert_question_json_to_text.py   # question JSON -> prompt + JSON response schema
  create_text_simulation_input.py    # persona + separator + question -> final prompt
  run_LLM_simulations.py             # async batch execution
  llm_helper.py                      # provider clients, retry, concurrency
  postprocess_responses.py           # LLM output -> response JSON
  configs/openai_config.yaml
```

### Original work — written at KLI, not derived from either upstream

**This is the section to cite in the paper's appendix.** Every file listed here
is original. Each carries an `ORIGINAL WORK (Kravis Leadership Institute)` line
in its module docstring, so authorship is checkable file by file without
consulting this document.

Everything original lives under `kli/` and `tests/`. Nothing in `kli/` imports
from `text_simulation/`, so the boundary is enforced by the code's own
structure, not only by documentation.

| File | Lines | What it contributes |
|---|---:|---|
| `kli/survey_tree.py` | 343 | **The schema adapter**, and the structural fix for upstream's silent answer-misassignment hazard. One canonical traversal of the nested per-participant response document, shared by prompt construction and response parsing, with a QuestionID round-trip check. |
| `kli/render_questions.py` | 431 | Renders questions into an answerable prompt with an explicit JSON skeleton. Adds constant-sum support, which the upstream renderer cannot handle at all. Adds chunking for small local models. |
| `kli/assemble_prompts.py` | ~400 | Persona and question assembly, including **the no-persona control arm** that upstream specifies in every config but never implements. Defaults to reproducing Peng et al.'s control wording exactly (verified against all 76 of their `empty_persona` configs); `control_wording: neutral` offers an alternative baseline, and the module docstring documents the trade-off and the unreported result that running both would produce. |
| `kli/providers/local.py` | 447 | **Local model provider** (Mistral via Ollama, or any OpenAI-compatible local server) with a hard no-egress guard that refuses non-loopback endpoints before any request is made. |
| `kli/parse_responses.py` | 436 | Model output back into the response-JSON schema, with per-question validation against the instrument, the round-trip check, and explicit imputation flags that prevent ground-truth leakage. |
| `kli/responses_to_csv.py` | 593 | **The analysis CSV.** This pipeline stage exists in neither upstream repository. Derives its column schema from the study's own reference export. |
| `kli/run_simulation.py` | 777 | Pipeline driver reading the nested config schema, with dry-run and offline stub modes and a per-run manifest. Applies the SJT-inclusion switch across all three persona arms. |
| `kli/reconcile.py` | 290 | Post-ingestion validation. Detects the questions upstream silently drops to the display-logic defect. |
| `kli/pilot_metrics.py` | ~430 | **Pilot-scale metrics**: per-item human-twin agreement and run-to-run stability across replicates. Original; needed because the correspondence metrics are computed across participants and are uninterpretable at n = 3. |
| `kli/combine_waves.py` | 426 | Joins the three KLI survey waves into one document per participant and one participant-by-answer table. Flattens Randomizer containers, which the vendored persona renderer otherwise skips silently. Adds the `TWIN_ID` column the vendored ingestion requires. |
| `kli/sjt_split.py` | 362 | **The SJT-inclusion arm.** Seeded study-wide draw of two holdout SJTs; splits each participant into persona base, holdout and training SJTs; renders training SJTs with the participant's own ratings; refuses runs whose question set is not exactly the holdouts. |
| `kli/config.py` | 136 | Nested config reader; fails loudly on missing keys rather than silently defaulting. |
| `kli/__init__.py`, `kli/providers/__init__.py` | 67 | Package docs and exports. |
| `tests/test_survey_tree.py` | 234 | Traversal and round-trip invariants. |
| `tests/test_pipeline.py` | ~544 | Rendering, validation, control arm, egress guard, CSV extraction. |
| `tests/test_sjt_arm.py` | 399 | Wave combination, the SJT split and leak checks, and all six arm combinations end to end in stub mode. |
| `tests/test_security_and_pilot.py` | ~330 | No-egress guarantee (including a regression test for the proxy-bypass hole found in external review), context-window handling, and pilot metrics. |

**Totals: ~5,600 lines of pipeline code and ~1,500 lines of tests, 159 tests
passing.** Upstream has no test coverage for any of these stages.

The scrambled-persona control (`scramble_persona_assignment` in
`kli/assemble_prompts.py`) is also original, and is the pipeline's analogue of
the random-trait control in Stojanov (2026).

Also newly written:

```
README.md                            # setup, pipeline, gaps, inherited defects
ATTRIBUTION.md                       # this file
NOTICE                               # Apache 2.0 §4(d) attribution notice
.gitignore
pyproject.toml                       # merged dependencies, plus local-model deps
configs/README.md                    # real config documentation
configs/example_study_local.yaml     # worked config for the local-model pipeline
docs/choosing_metrics.md             # which metrics apply at which sample size
docs/data_locality.md                # the no-egress guarantee, and its limits
configs/pilot/                       # three-participant pilot configs
text_simulation/convert_responses_to_csv.py   # shim forwarding to kli.responses_to_csv
```

### Verification against real data

The original components were developed and tested against genuine Qualtrics
output, not synthetic fixtures. Running the vendored stages 1–2 over three of
the upstream studies produced 1,654 real per-participant response files
(`story_beliefs` 251, `accuracy_nudges` 1,003, `junk_fees` 400), spanning flat
documents and documents nested three levels deep through Randomizer, Group and
Branch containers, and covering all six question types the parser emits (MC,
Matrix, Slider, TE, CS, DB).

The traversal's invariants hold across all three; the renderer handles every
question in all three; and a full stub-model run reaches 100% imputation end to
end through to the analysis CSV. The data itself is not redistributed here.

## Statement of changes (Apache 2.0 §4(b))

As of the initial commit, **no upstream source file has been modified.** Every
vendored `.py`, `.yaml`, `.csv`, and `.j2` file is byte-identical to its
upstream original at the commit listed above. Changes made relative to the
upstreams are confined to:

1. **Selection** — only a subset of each upstream repo is included.
2. **Relocation** — `mega_study_evaluation/accuracy_nudges/` is vendored as
   `mega_study_evaluation/_template_study/`; `configs/accuracy_nudges/accuracy_nudges.yaml`
   is vendored as `configs/example_study.yaml`. File contents are unchanged,
   which means the paths *inside* `configs/example_study.yaml` still read
   `data/accuracy_nudges/...` and must be edited for your study.
3. **Addition** — the newly written files listed above.
4. **Omission** — compiled artifacts (`__pycache__`, `*.pyc`), the upstream
   `results/` tree (3,411 gzipped CSVs), the upstream `.dat/` raw study data,
   and upstream components not needed here (per-study evaluation modules other
   than the template, the specialized evaluation suite, ML baselines, the
   fine-tuning arm, the relevance analysis, and the prediction-comparison
   pipeline).

Any subsequent modification to a vendored file should be recorded below with
the file path, the date, and a one-line reason, so that divergence from
upstream stays auditable.

### Modification log

| Date | File | Change | Reason |
|---|---|---|---|
| — | — | *(none yet)* | — |
