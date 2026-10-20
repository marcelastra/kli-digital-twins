# Measuring how well digital twins reproduce human responses

Methodological rationale for the correspondence metrics in `kli/correspondence.py`.
Written to be quotable in a methods section.

**Primary reference.** Stojanov, A. (2026). Do participant-matched LLM personas
approximate human survey data? *Personality and Individual Differences*, 261,
113915. https://doi.org/10.1016/j.paid.2026.113915

---

## 1. Why not just report accuracy

The Twin-2K-500 evaluation stack reports two things per outcome: a
range-normalized mean absolute deviation ("accuracy") and a correlation. Both
are useful. Neither is sufficient, and together they can make a failed
simulation look successful.

Consider three twin populations, all scored against the same humans:

| | twin behaviour | mean accuracy | correlation |
|---|---|---|---|
| A | reproduces each person exactly, then adds a constant | high | perfect |
| B | reproduces each person's rank but compresses the spread by half | high | perfect |
| C | assigns everyone the sample mean | moderate | zero |

A and B score identically on accuracy and correlation, yet they fail in
opposite ways. A's error is entirely systematic and could be removed by
subtracting a constant; B's responses carry no between-person variance to
speak of, so any analysis depending on individual differences is compromised
even though every twin is "close". C looks mediocre rather than catastrophic,
despite containing no information at all.

The metrics below separate those cases. We adopt them from Stojanov (2026)
rather than inventing our own, for three reasons: they were designed for
exactly this question, they decompose correspondence into components with
different practical implications, and they come with published values that give
our results an external reference point.

---

## 2. The metrics

All are computed on paired data: each twin is matched to the specific human it
was built from. Unmatched records are dropped and reported.

### 2.1 Group level — does the twin sample look like the human sample?

Computed per outcome, across participants.

**Glass's Δ** = (M_twin − M_human) / SD_human

Standardized difference in means. Answers whether twins are systematically
shifted on an outcome, and by how much in human units. We report both the
signed mean (direction of bias across outcomes) and the mean absolute value
(magnitude, which does not cancel).

**SD ratio** = SD_twin / SD_human

Between-person variability. A value near 1 means twins differ from one another
about as much as people do. Values well below 1 indicate the compression that
is the most consistently reported failure of LLM personas: Stojanov found a
median ratio of 0.36 for demographics-only personas, meaning twins spanned
roughly a third of the human spread.

**Dispersion mismatch** = |log(SD_twin / SD_human)|

The SD ratio made symmetric. Zero is perfect agreement. We log the ratio
because on the raw scale over- and under-dispersion are not comparable: ratios
of 0.5 and 2.0 are equally wrong but sit at distances of 0.5 and 1.0 from
perfect agreement. The log fixes that and makes the metric comparable across
outcomes measured on different response formats.

### 2.2 Individual level — does *this* twin resemble *this* person?

Computed per participant, across outcomes.

**Person-level MAD** = mean over outcomes of |human − twin| / SD_human

How far a twin sits from its own human counterpart, in human standard
deviations, averaged across outcomes. This is the most direct answer to "could
this twin stand in for this person".

**Within-person profile correlation**

Both sides are first standardized against the human mean and SD of each
outcome, which removes differences in level and spread. We then correlate each
participant's human profile with their twin's profile across outcomes. This
asks whether the twin reproduces the *shape* of the person — relatively high
here, relatively low there — independent of overall level.

This metric needs a reasonable number of outcomes to be stable. Stojanov used
91. Our implementation warns below 10 and the module documents why; with three
or four outcomes the value should not be interpreted.

### 2.3 Outcome level — does the twin rank people correctly?

**Outcome-level correlation**: for each outcome, Pearson *r* between human and
twin scores across participants. This is the quantity most directly relevant to
whether twins can substitute for participants in individual-differences
research. Correlations are Fisher *z*-transformed before averaging and before
any comparison between arms.

### 2.4 Error decomposition — is the error systematic or random?

Computed per outcome, all standardized by the human SD.

- **Standardized absolute bias** = |mean(twin − human)| / SD_human
- **Standardized RMSE** = RMSE / SD_human
- **Standardized error SD** = SD(twin − human) / SD_human

The distinction carries a practical consequence. Systematic error is a fixed
offset and can in principle be calibrated away, so a twin population with high
bias and low error variability may still be usable after correction. Random
error cannot be removed post hoc. Reporting RMSE alone conflates the two.

---

## 3. Why every metric is scaled by the human standard deviation

Each standardized quantity divides by the **human** SD, never a pooled SD. This
follows Stojanov's use of Glass's Δ rather than Cohen's *d* (Lakens, 2013), and
is a deliberate choice with a substantive justification rather than a
conventional one.

The human distribution is the target the twin is trying to reproduce, so human
units are the natural denominator. More importantly, **twin variance is itself
one of the things being measured.** A pooled denominator would shrink as twins
became more homogeneous, so a method that collapsed everyone onto the sample
mean would be rewarded with a smaller apparent error. Holding the denominator
fixed at the human SD keeps variance compression visible rather than absorbing
it into the yardstick.

Our test suite verifies this directly: increasing compression increases
standardized RMSE, which would not hold under pooled standardization.

A secondary benefit is comparability. Outcomes measured on a 1–7 Likert scale
and a 0–100 slider are not otherwise on the same footing, and averaging raw
deviations across them would weight by scale width rather than by substantive
importance.

---

## 4. The three-arm design

Correspondence metrics describe how closely twins track humans. They cannot, on
their own, say whether that tracking comes from person-specific information.
Three arms are required, differing only in what persona the model receives:

| Arm | Persona supplied | Config |
|---|---|---|
| **Matched** | the participant's own | default |
| **Scrambled** | a real persona belonging to a *different* participant | `scramble_personas: true` |
| **Empty** | none | `empty_persona_prompt: true` |

Two contrasts follow, and they identify different things:

**Matched − Empty** is the overall benefit of supplying a persona. This is the
comparison the Twin-2K-500 mega-study makes, and it is necessary but not
sufficient: a persona supplies individuating information *and* changes how the
model responds in general, and this contrast cannot distinguish them.

**Matched − Scrambled** isolates the person-specific signal. Persona presence,
length, format and richness are held constant; only the pairing between person
and persona is destroyed. Any advantage that survives this contrast is genuine
individuation.

**Scrambled − Empty** is the remainder: the benefit of having *a* persona,
regardless of whose.

This mirrors the control Stojanov added at a reviewer's request, assigning each
persona a Big Five profile drawn from a different participant. It proved
decisive. Random traits *worsened* mean correspondence but modestly *improved*
some error metrics, showing that part of the apparent benefit of real traits
was a generic broadening of response variability rather than person-specific
matching. Without that arm the improvement would have been over-read — which is
precisely the inference risk in any twin study reporting only a persona-versus-
no-persona contrast.

The scrambled assignment is a seeded derangement, so no participant can receive
their own persona by chance and the assignment is reproducible. The seed is
recorded in the run manifest.

---

## 5. How to read the numbers

Correlations in this literature are small, and the appropriate reference points
are not the conventional .10/.30/.50 thresholds.

**Gignac and Szodorai (2016)** give empirical benchmarks for
individual-differences research: *r* = .11 small, .19 typical, .29 large.
Stojanov's best correspondence (*r* = .21) sits at the lower edge of "typical".

**Calamia et al. (2013)**: test-retest reliabilities of psychological measures
typically exceed *r* = .70. This is the relevant bar for substitution. A twin
correlation of .21 is statistically distinguishable from zero in a sample of
177 and still far below the level at which twins could replace participants.

Stojanov's conclusion is worth stating plainly, because it sets the prior for
any new study: model-implied responses are *"statistically plausible at the
group level but psychologically shallow at the individual level"*, and are
*"best suited for aggregate-level applications such as pilot testing, screening
hypotheses, or approximating group-level distributions"* rather than for
substituting individual participants.

`interpret_correlation()` prints the appropriate band automatically, so this
framing is applied consistently rather than reconstructed from memory.

---

## 6. Published reference values

Stojanov (2026), Table 2. GPT-4o, temperature 0.7, *k* = 91 constructs,
*n* = 177 participants. Available in code as `STOJANOV_2026_BENCHMARKS`.

| Metric | Demographics only | Demographics + Big Five | Random Big Five |
|---|---|---|---|
| Mean \|Glass's Δ\| | 0.332 | 0.299 | 0.344 |
| Dispersion mismatch | 1.068 | 0.519 | 1.116 |
| Median SD ratio | 0.355 | 0.613 | 0.345 |
| Person-level MAD | 0.891 | 0.855 | 0.880 |
| Within-person profile *r* | 0.037 | 0.187 | 0.060 |
| Outcome-level *r* | 0.036 | 0.213 | 0.061 |
| Standardized RMSE | 1.119 | 1.117 | 1.108 |
| Standardized error SD | 1.053 | 1.062 | 1.038 |

**These are reference points, not targets.** That study used a different
population (CloudResearch, employed and partnered), a different instrument (91
psychological constructs), and a different model. Cite them to situate a
result, not to claim replication. Note in particular that standardized RMSE was
essentially unchanged across all three conditions — the gains from richer
personas appeared in dispersion and rank correspondence, not in total error.

---

## 7. Statistical comparison between arms

Following Stojanov §2.2.5: paired-samples *t*-tests, Wilcoxon signed-rank tests
as a non-parametric robustness check, and Cohen's *d_z* as the effect size.
Pairing is by outcome for outcome-level metrics and by participant for
individual-level metrics; `compare_arms()` reports which unit each row uses.

The comparison is paired because the arms describe the same outcomes and the
same participants, differing only in persona content. That is what makes the
contrast informative — the instrument and the sample are held fixed.

Correlations are Fisher *z*-transformed before testing.

**One caveat to carry into the write-up.** With many outcomes and several arms,
these tests are numerous and no multiple-comparison correction is applied here
— consistent with the upstream stack, which also applies none. Either apply a
correction post hoc or state that the per-outcome tests are descriptive. The
arm-level summary comparisons are few enough to report individually.

---

## 8. Usage

```bash
# one arm against the human data
python -m kli.correspondence \
    --human runs/my_study/human/consolidated_original_answers_values.csv \
    --twin  runs/my_study/matched/csv_output/consolidated_llm_values.csv \
    --outcomes "dv_engagement,dv_trust,dv_intent" \
    --out runs/my_study/correspondence/matched

# the contrast that isolates person-specific signal
python -m kli.correspondence \
    --human   .../human.csv \
    --twin    .../matched/consolidated_llm_values.csv \
    --control .../scrambled/consolidated_llm_values.csv \
    --out runs/my_study/correspondence/matched_vs_scrambled
```

Naming `--outcomes` explicitly is strongly recommended. Without it, every shared
numeric column is treated as an outcome, which will silently include timing and
metadata columns.

Outputs: `correspondence_by_outcome.csv`, `correspondence_by_participant.csv`,
`correspondence_summary.json`, and `arm_comparison.csv` when a control is given.

---

## 9. Draft methods text

> **Correspondence metrics.** We evaluated human–twin correspondence following
> Stojanov (2026), at three levels of aggregation. At the group level we
> computed, for each outcome, the standardized mean difference between twin and
> human responses (Glass's Δ, using the human standard deviation as the
> denominator per Lakens, 2013), the ratio of twin to human standard deviations,
> and the absolute log of that ratio as a symmetric index of dispersion
> mismatch. At the individual level we computed, for each participant, the mean
> absolute deviation between their own and their twin's responses scaled by the
> human standard deviation of each outcome, and the correlation between their
> human and twin response profiles across outcomes after standardizing both to
> the human reference distribution. At the outcome level we computed the
> correlation between human and twin responses across participants. We further
> decomposed prediction error into standardized absolute bias, standardized
> RMSE, and standardized error variability, in order to distinguish systematic
> from unsystematic error. All standardized metrics use the human standard
> deviation as the denominator, so that compression of twin variance remains
> visible rather than being absorbed into the scaling.
>
> **Control conditions.** We ran three arms differing only in the persona
> supplied: the participant's own persona, a real persona belonging to a
> different participant (a seeded derangement), and no persona. Contrasting the
> matched and no-persona arms estimates the overall benefit of persona
> conditioning; contrasting the matched and scrambled arms isolates
> person-specific signal from the generic effect of supplying a persona of any
> kind. Arms were compared with paired-samples *t*-tests, Wilcoxon signed-rank
> tests, and Cohen's *d_z*, pairing by outcome or by participant as appropriate.
> Correlations were Fisher *z*-transformed before averaging and before
> comparison.

---

## References

Calamia, M., Markon, K., & Tranel, D. (2013). The robust reliability of
neuropsychological measures: Meta-analyses of test–retest correlations.
*The Clinical Neuropsychologist*, 27(7), 1077–1105.

Gignac, G. E., & Szodorai, E. T. (2016). Effect size guidelines for individual
differences researchers. *Personality and Individual Differences*, 102, 74–78.

Lakens, D. (2013). Calculating and reporting effect sizes to facilitate
cumulative science. *Frontiers in Psychology*, 4, 863.

Stojanov, A. (2026). Do participant-matched LLM personas approximate human
survey data? *Personality and Individual Differences*, 261, 113915.
