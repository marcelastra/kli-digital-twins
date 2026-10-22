# Which metrics hold at which sample size

This repository computes **only** the metrics that are valid at pilot sample
sizes. Metrics computed across participants live in the project's analysis
repository and are applied once the full sample exists.

The split is not arbitrary. It follows from what each metric is averaging over.

---

## The distinction

Every metric aggregates over something. Which something determines how many
participants it needs.

**Within-pair and within-twin metrics** compare a twin to its own human, or a
twin to itself. Each participant contributes an independent observation, and
with many items per participant you accumulate many comparisons even with few
people.

> Three participants answering a 40-item instrument give you 120 human–twin
> item comparisons. That supports a meaningful agreement rate.

**Across-participant metrics** describe the sample as a whole — a correlation
over participants, a standard deviation ratio, a standardized mean difference.
Here the participants *are* the observations, and three of them is three.

> A Pearson correlation on n = 3 can come out at .99 or −.99 by accident. A
> standard deviation estimated from three points is close to noise. Reporting
> either as a finding would be misleading regardless of how carefully the
> pipeline produced the inputs.

Adding items does not rescue an across-participant metric. Forty items give you
forty correlations each computed on three people — forty unreliable numbers, not
one reliable one.

---

## In this repository (`kli/pilot_metrics.py`)

| Metric | What it averages over | Minimum n |
|---|---|---|
| **Per-item agreement** — exact match rate, within-one rate, mean absolute deviation | Participant × item comparisons | Any |
| **Run-to-run stability** — modal-answer rate, fully-consistent rate, SD across runs | Replicates within one twin | Any (needs ≥ 2 replicates) |

Both generalize. Neither becomes redundant once the sample grows, and both
answer questions the across-participant metrics cannot.

Stability in particular is worth reporting at every stage, because it **bounds
everything else**. A twin that gives a different answer each time it is asked
cannot agree with its human more consistently than it agrees with itself. If
stability is poor, a low agreement rate says nothing about the persona — it says
the configuration is too noisy to measure anything. Check stability first.

---

## In the analysis repository

Computed across participants; apply once the full sample exists.

| Metric | What it tells you |
|---|---|
| Glass's Δ | Systematic shift in the twin sample's mean |
| SD ratio, dispersion mismatch | Whether twins reproduce between-person variability |
| Person-level MAD | Distance from each twin to its own human, in human SD units |
| Within-person profile correlation | Whether the twin reproduces the *shape* of a person's profile |
| Outcome-level correlation | Whether twins rank participants as the humans did |
| Standardized bias / RMSE / error SD | Systematic versus random error |

Definitions follow Stojanov (2026), *Personality and Individual Differences*,
261, 113915. The implementation lives with the analysis code rather than here,
so there is one copy rather than two drifting apart.

One caveat carried over: the within-person profile correlation is computed
*across outcomes*, so it needs a reasonable number of them to be stable.
Stojanov used 91. With fewer than about ten it should not be interpreted, and
person-level MAD is the better individual-level report.

---

## What the pilot is actually for

Three participants cannot establish how well twins reproduce human responses.
That is not the pilot's job. The pilot establishes:

1. **That the pipeline runs end to end** on your real instruments — ingestion,
   persona rendering, prompt assembly, local inference, parsing, export.
2. **That ingestion is not silently dropping questions.** Run
   `python -m kli.reconcile` on both instruments and read the output.
3. **That prompts fit the context window.** The pipeline refuses oversized
   prompts rather than letting the server truncate them, so this surfaces as an
   error rather than as quietly persona-free output.
4. **How stable the model is** at your chosen temperature and chunk size. This
   is a property of the model and prompt, not of the sample, so three
   participants estimate it about as well as three hundred.
5. **How long a full run will take.** Multiply by participants and replicates.

Agreement rates from the pilot are worth looking at, but as a sanity check — is
it near chance? is it suspiciously perfect? — rather than as a result.

---

## If you run the full metrics on pilot data anyway

Reasonable as a plumbing check: it confirms the export aligns column-for-column
with the human data and that the analysis code runs. If you do, label the output
clearly as a pipeline test rather than a finding, and do not put the numbers in
a slide. A correlation computed on three people has a way of being quoted later
without its caveat.
