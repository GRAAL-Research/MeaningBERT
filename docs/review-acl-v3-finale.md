# ARR Review — Unrelated Is Not Opposed: A Signed Scale for Meaning Preservation

**Domain / Track:** evaluation of generation (meaning-preservation metrics for text simplification), with natural language inference as a component
**Review passes:** 3 (comprehension and structure; methodological rigour; adversarial)
**Version reviewed:** final pre-submission, 12 pages, body ends two thirds down p.8

## Compliance pre-check

Long paper. Sections 1–8 end on page 8; Limitations, Ethical Considerations, Availability and references follow and are outside the limit. Limitations present, unnumbered, after the conclusion. Four appendices. Repository withheld for review. No page-limit violation, and unlike the previous version there is now roughly a third of a page of slack rather than none.

## Paper Summary

Meaning-preservation metrics emit a single non-negative score, which cannot express opposition: contradicting sentences share subject and lexical material, so a magnitude metric reads that overlap as preserved meaning. The authors measure the failure on a released metric, which ranks agreement above contradiction at 60.8% AUC on a 12,000-pair balanced split and 48.0% on SICK, while a thresholded token overlap reaches 63.5%. They propose a signed scale on [-100, 100] obtained by multiplying a magnitude head by (1 - alpha * p_contra), the product form keeping unrelated pairs near zero, with alpha fixed at 2 by requiring the negative half to be reachable. A grid over seven encoders, ten seeds and two corpus conditions reports a polarity head at 91.1% macro-F1 and 98.41% AUC over ten seeds; derived-polarity augmentation leaves task performance unchanged while raising generated sanity-suite accuracy from 79–90% to 98–99%; and composition places 83.7% of SICK contradictions below zero where the magnitude alone places none, cutting accepted contradictions at a fixed cut from 14.10% to 2.13% for 0.72 points of entailment recall.

## Summary of Strengths

1. **The diagnosis is measured, and the right baseline was run.** A thresholded token overlap (63.5% AUC) orders polarity better than a metric trained on human preservation judgements (60.8%). That single comparison makes the argument unavoidable rather than rhetorical.
2. **Every headline number now carries seed variance.** The previous version reported the polarity AUC from one run; it is now 98.41 ± 0.32 over ten retrained seeds, and the advantage over the off-the-shelf head (+1.79) is stated in units of that standard deviation rather than asserted.
3. **The decision-point table answers the question the rest of the paper invites.** Expressing a distinction is not the same as being a better metric; Table 8 measures what each scale lets through, carries a control for a scale that merely shifts everything down, and reports a threshold-free sweep rather than three chosen cuts.
4. **Bounding results are reported even when they narrow the contribution.** The off-the-shelf head's 96.62% AUC, the augmentation/sanity-suite confound, the correlation the composition costs, and the fact that the macro-F1 gap flatters fine-tuning because the published head never saw the label mapping.
5. **The library version is treated as an experimental variable.** Appendix B states that a later major release of `transformers` sends DeBERTaV3-large to a non-finite gradient within a hundred steps and reports the majority-class floor rather than failing. Few papers would disclose this; it is exactly the kind of detail that decides whether a third party reproduces the work.

## Summary of Weaknesses

1. **[MODERATE, FIXED] Table 6 carried a standard deviation on one row and none on the others,** with no explanation. The caption now states that the ten seeds apply to the fine-tuned head and that the fixed checkpoints are evaluated once.

2. **[MODERATE, FIXED] The pooled AUC (98.41%) exceeded both per-corpus means (98.11% SICK, 98.32% VitaminC),** which reads as an arithmetic error. Checking the ten seeds individually, the pooled value lies *between* the two halves on nine of them; the SICK mean is dragged below by a single seed at 94.40 against a 99.26 best. The text now says so.

3. **[MODERATE, FIXED] The ten weight-keeping runs behind the AUC column were undocumented.** Appendix B said the settings were identical for all 140 runs, and the AUC runs are not among those 140: nine of the ten use a micro-batch of 8 on a later card. Appendix B.3 now describes them, with the effective batch unchanged at 32.

4. **[MINOR, REMAINS] The macro-F1 comparison with the off-the-shelf head is not label-matched.** The published head never saw the mapping from VitaminC's FACT3 labels onto the paper's three classes, so part of the 22.7-point gap is convention rather than capability. The paper says this and notes that the AUC gap does not suffer from it. A reviewer may still ask for the off-the-shelf macro-F1 under an oracle label permutation.

5. **[MINOR, REMAINS] Two numbers are mixed across machines.** The AUC mean pools one seed trained on the original hardware with nine trained on a later card at a different micro-batch. The effective batch is held at 32, which is the paper's own stated condition for comparability, but the mixture is only inferable from Appendix B.3 and is not flagged where the number appears.

## Comments, Suggestions and Typos

- Table 6 is first referenced in Section 6, two pages before it appears. The text now tells the reader that the baseline macro-F1 values live in its caption, since the table's own column reports AUC.
- Captions were cut by 17% overall, Table 8's by 42%; the argumentative material they carried duplicated the body.
- Thousands separators follow English usage (1,536) rather than the thin space.
- Table 4 reports Cohen's d to two decimals below 10 and one above, mixed within a column, and labels its unit "pp" while the running text says "points".
- The LexFlip reference (arXiv:2609.05296) is about a month old at submission and unrefereed, yet load-bearing in the introduction and in Section 2.1. Both uses are reported as observations rather than established results, which is the right handling.

## Questions for the Authors

1. Table 8 shows the published polarity head accepting 0.15% contradictions against the fine-tuned head's 2.13%, for 1.7 further points of entailment recall. Why is the fine-tuned configuration the headline, given that contribution 1 is stated in terms of contradictions accepted?
2. What would the off-the-shelf head's three-way macro-F1 be under the best label permutation, so the +22.7 can be read as capability rather than convention?
3. Does the AUC mean change materially if the one seed trained on the original hardware is excluded?

## Scores

- **Confidence:** 4/5 — every number in the text was recomputed against the released result files, the per-seed breakdowns were inspected individually rather than in aggregate, and the captions were checked against the generators that produce them. Not 5: the training code was not executed.
- **Soundness:** 4/5 — claims are supported, the protocol is careful, significance is handled properly, and the headline figures now carry seed variance. Held at 4 by weaknesses 4 and 5.
- **Excitement:** 4/5 — that a deployed metric orders polarity no better than token overlap is a result that changes how a community reads its own numbers, and the composition is cheap enough to be adopted. Not 5: the construction assembles existing components.
- **Overall Assessment:** 4/5 (Conference) — no blocking weakness; a measured diagnosis and a reproducible solution with its costs stated.
- **Reproducibility:** 4.5/5 — hyperparameters, seeds, splits, hardware, library version and budget documented; the analysis script generates every table and figure. Not 5 until the repository is public.
- **Datasets:** 4/5 — merged, stratified, leakage-controlled polarity corpus with derived-polarity augmentation.
- **Software:** 4/5 — corpus builder, training, composition, and the analysis that produces every number in the paper.

## Best Paper: No

Strong diagnosis and careful engineering, but the contribution extends existing components, and the scale is validated against necessary conditions rather than against human judgement.

## Limitations and Societal Impact

Present, unnumbered, after the conclusion, and substantial: seven named limitations, each carrying a measurement, including three most authors would omit — that the sign rests entirely on one head, that the composition correlates worse than the magnitude alone on every human-rated quantity available, and that the property the title names is evidenced on generated pairs because no corpus annotates genuinely unrelated ones.

## Ethical Concerns

Needs Ethics Review: **No**. Public corpora, no human subjects, no personal data; bias inheritance acknowledged and cited.

## Knowledge of Author Identity

No educated guess required. The paper extends a named released metric and cites two papers sharing a first author, one of them that metric. Both are cited in the third person, which is permitted.

---

*Decision-support draft. ARR forbids submitting a review written by generative AI and forbids uploading a confidential manuscript to a third-party service. Read, verify, own and reformulate before any real submission; never paste verbatim.*
