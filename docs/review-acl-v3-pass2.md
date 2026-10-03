# ARR Review — Unrelated Is Not Opposed: A Signed Scale for Meaning Preservation

**Domain / Track:** evaluation of generation (meaning-preservation metrics for text simplification), with natural language inference as a component
**Review passes:** 2 (comprehension and structure; methodological rigour)
**Version reviewed:** post-revision, 11 pages, body ends p.7

## Compliance pre-check

Long paper. Sections 1–8 end on page 7; Limitations, Ethical Considerations, Availability and references follow and are outside the limit. Limitations is present, unnumbered and placed after the conclusion, which matches ARR practice. Three appendices. No page-limit violation. One anonymity remark below, not a violation.

## Paper Summary

The paper argues that meaning-preservation metrics, which emit a single non-negative score, cannot express opposition: contradicting sentences share subject and lexical material, so a magnitude metric reads that overlap as preserved meaning. The authors measure the failure on a released metric, which ranks agreement above contradiction at 60.8% AUC on a 12,000-pair balanced split and at 48.0% on SICK, while a thresholded token overlap reaches 63.5% on the same split. They propose a signed scale on [-100, 100] obtained by multiplying a magnitude head by (1 - alpha * p_contra), the product form ensuring that unrelated pairs stay near zero, with alpha fixed at 2 by requiring the negative half to be reachable. A 140-run grid (seven encoders x ten seeds x two corpus conditions) on a merged, stratified, leakage-controlled corpus reports a polarity head at 91.1% macro-F1; derived-polarity augmentation leaves task performance unchanged while raising generated sanity-suite accuracy from 79–90% to 98–99%; and composition places 83.7% of SICK contradictions below zero where the magnitude alone places none.

## Summary of Strengths

1. **The diagnosis is measured and the right baseline was run.** Showing that a thresholded token overlap (63.5% AUC) orders polarity better than a metric trained on human preservation judgements (60.8%) is the strongest evidence in the paper, and it is exactly the comparison that makes the argument unavoidable rather than rhetorical.
2. **Statistical hygiene well above the subfield norm.** Ten seeds per cell with standard deviations throughout, Welch's t-test with Cohen's d beside every p-value, Holm-Bonferroni across the test family, and an explicit statement that a 0.1-point difference reaches significance and means nothing.
3. **Design choices are argued, not asserted.** The product form, the rejection of MQM-style typed penalties, the rejection of end-to-end supervision, and the rejection of PAWS each carry a reason tied to a failure the choice avoids.
4. **Bounding results are reported even when they narrow the contribution.** The off-the-shelf model's 96.6% AUC, the augmentation/sanity-suite confound, the single-seed status of the AUC figures, and the correlation the composition costs are all stated by the authors rather than left for a reviewer to find.
5. **Two contributions, clearly scoped.** The revision to two claims removes the common failure of listing a dataset and a grid as contributions.

## Summary of Weaknesses

1. **[MAJOR] The robustness check for alpha is close to a tautology, and the paper says so without drawing the consequence.** Section 7 states that the slope "survives a change of magnitude model", then two sentences later explains that both checkpoints send *the same* 84.6% of contradictions below zero because, with m >= 0, the sign of the product depends on p_contra alone. If the sign criterion is invariant by construction, the only term in the objective that can move alpha is the relatedness correlation, so the check tests far less than the heading claims. **Fix:** state that two of the three criteria are invariant to the magnitude model by construction, and present the check as evidence about the third term only.

2. **[MAJOR] The combined objective is never defined.** Tables 7 and 8 report a "combined objective", Section 7 says alpha is selected "against three requirements multiplied together", and Section 3 says reachability is a constraint rather than "a term in the objective" — but the quantity is never written down. A reader cannot reproduce the selection, and cannot check whether the 14% relative gap in Table 8 is large. **Fix:** give the product explicitly, with the clipping of the correlation term at zero.

3. **[MODERATE] The abstract pairs a ten-seed mean with a single-run figure without distinguishing them.** "A polarity head reaches 91.1% macro-F1 and 98.7% AUC" reads as two measurements of the same kind. The body discloses that the AUC comes from one run; the abstract does not. **Fix:** qualify the AUC in the abstract or drop it there.

4. **[MODERATE] Section 6 refers to Table 6 two pages before it appears**, and the sentence introducing it ("Table 6 scores three released magnitude models") undercounts a table that now carries three blocks including polarity heads and surface baselines. **Fix:** move the baseline discussion next to the table, or split the table.

5. **[MINOR] Cohen's d reported to two decimals at magnitudes near 80.** Table 4 gives d = +81.07. At that magnitude the second decimal is noise and invites the reader to take the precision seriously. **Fix:** one decimal above d = 10, or report the order of magnitude.

6. **[MINOR] "Both are variants of the same metric"** (Section 7) is accurate about the training recipe but may read as "the same architecture", which is false: the two checkpoints are a BERT-base and a DeBERTa-v3-large. **Fix:** say "both trained on the same preservation corpus".

## Comments, Suggestions and Typos

- Figure 1 (the pipeline) is referenced before Table 1 but placed after it in the rendered page; consider moving the float.
- Table 3 caption says "Bold: best per column, higher being better everywhere"; the bold MoNLI value sits on a RAW row while the paper's preferred condition is AUG, which a hurried reader may misread.
- Section 5 says the grid is "roughly 500 GPU-hours" and Appendix C repeats it; one of the two can go.
- "the AUC of 1 - p_contra ranking entailment above contradiction" (Section 5) is the clearest statement of the quantity in the paper and deserves to be where the AUC is first used, in Section 1.

## Questions for the Authors

1. What exactly is the combined objective, and is the correlation term clipped or normalised before the product?
2. Given that the sign of the composed score depends only on p_contra, what does the magnitude head contribute beyond amplitude, and would a weaker magnitude model change any conclusion other than the relatedness correlation?
3. The AUC figures come from one run. What is their seed variance, and does the +2.1-point advantage over the off-the-shelf head survive it?

## Scores

- **Confidence:** 4/5 — the subfield, the corpora, the statistical procedure and the composition arithmetic were checked against the reported tables and the released analysis code. Not 5: the code was not executed.
- **Soundness:** 3.5/5 — main claims supported, protocol careful, significance handled properly. Held below 4 by the undefined objective (weakness 2) and the overstated robustness check (weakness 1), both correctable in text.
- **Excitement:** 4/5 — the finding that a deployed metric orders polarity no better than token overlap is the kind of result that changes how a community reads its own numbers, and the composition is simple enough to be adopted. Not 5: the construction assembles existing components.
- **Overall Assessment:** 3.5/5 (Borderline Conference) — no blocking weakness; a useful diagnosis and a cheap, reproducible solution. The two major weaknesses are textual and could be closed in a rebuttal, which would move this to 4.
- **Reproducibility:** 3.5/5 — hyperparameters, seeds, splits, hardware and budget documented and artefacts promised, but the selection criterion that fixes the paper's one free parameter is not written down.
- **Datasets:** 4/5 — merged, stratified, leakage-controlled polarity corpus with derived-polarity augmentation, loaders released separately.
- **Software:** 4/5 — corpus builder, training, composition, and the analysis that generates every table and figure.

## Best Paper: No

Strong diagnosis and careful engineering, but the contribution extends existing components, and the scale is validated against necessary conditions rather than against human judgement.

## Limitations and Societal Impact

Present, unnumbered, after the conclusion, and substantial. Six named limitations, each carrying a measurement, including two that most authors would omit: that the sign rests entirely on one head, and that the composition correlates worse than the magnitude alone on every human-rated quantity available. The Ethical Considerations section correctly identifies that the failure mode reaches the reader rather than the engineer.

## Ethical Concerns

Needs Ethics Review: **No**. Public corpora, no human subjects, no personal data; bias inheritance acknowledged and cited.

## Knowledge of Author Identity

No educated guess required, but note: the paper extends a named released metric and cites two papers sharing a first author, one of them that metric. Both are cited in the third person, which is permitted. The risk is inherent to the topic rather than a violation.

---

*Decision-support draft. ARR forbids submitting a review written by generative AI and forbids uploading a confidential manuscript to a third-party service. Read, verify, own and reformulate before any real submission; never paste verbatim.*
