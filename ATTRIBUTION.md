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

### Newly written for this repository

```
README.md                            # setup, pipeline, gaps, inherited defects
ATTRIBUTION.md                       # this file
NOTICE                               # Apache 2.0 §4(d) attribution notice
.gitignore
pyproject.toml                       # merged dependencies of both upstreams
configs/README.md                    # real config documentation
text_simulation/convert_responses_to_csv.py   # SPECIFIED STUB — not implemented
```

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
