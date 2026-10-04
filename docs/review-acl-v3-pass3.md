# ARR Review — Unrelated Is Not Opposed: A Signed Scale for Meaning Preservation

**Domain / Track:** evaluation of generation (meaning-preservation metrics for text simplification), with natural language inference as a component
**Review passes:** 3 (comprehension and structure; methodological rigour; adversarial)
**Version reviewed:** post-restructure, 12 pages, body ends p.8

## Compliance pre-check

Long paper. Sections 1-8 end on page 8; Limitations, Ethical Considerations, Availability and references follow and sit outside the limit. Limitations is present, unnumbered, after the conclusion. Four appendices. No page-limit violation. The repository link is withheld for review, which is correct. One anonymity remark below, not a violation.

## Paper Summary

Meaning-preservation metrics emit a single non-negative score, which cannot express opposition: contradicting sentences share subject and lexical material, so a magnitude metric reads that overlap as preserved meaning. The authors measure the failure on a released metric, which ranks agreement above contradiction at 60.8% AUC on a 12,000-pair balanced split and 48.0% on SICK, while a thresholded token overlap reaches 63.5% on the same split. They propose a signed scale on [-100, 100] obtained by multiplying a magnitude head by (1 - alpha * p_contra), the product form keeping unrelated pairs near zero, with alpha fixed at 2 by requiring the negative half to be reachable. A grid over seven encoders, ten seeds and two corpus conditions reports a polarity head at 91.1% macro-F1; derived-polarity augmentation leaves task performance unchanged while raising generated sanity-suite accuracy from 79-90% to 98-99%; and composition places 83.7% of SICK contradictions below zero where the magnitude alone places none, cutting accepted contradictions at a fixed cut from 14.10% to 2.13% for 0.72 points of entailment recall.

## Summary of Strengths

1. **The diagnosis is measured, and the right baseline was run.** A thresholded token overlap (63.5% AUC) orders polarity better than a metric trained on human preservation judgements (60.8%). That single comparison makes the argument unavoidable rather than rhetorical.
2. **Statistical hygiene well above the subfield norm.** Ten seeds per cell with standard deviations throughout, Welch's t-test with Cohen's d beside every p-value, Holm-Bonferroni across the test family, and an explicit statement that a 0.1-point difference reaches significance and means nothing.
3. **The decision-point table answers the question the rest of the paper invites.** Expressing a distinction is not the same as being a better metric; Table 8 measures what each scale lets through, carries a control for a scale that merely shifts everything down, and reports a threshold-free sweep rather than three chosen cuts.
4. **Bounding results are reported even when they narrow the contribution.** The off-the-shelf head's 96.6% AUC, the augmentation/sanity-suite confound, the single-seed status of the AUC figures, and the correlation the composition costs are all stated by the authors.
5. **Novelty holds on an external check.** Signed scales exist in simplification evaluation for *simplicity* (relative -2 to +2 judgements), not for meaning preservation. No prior signed meaning-preservation metric was found.

## Summary of Weaknesses

1. **[MAJOR, FIXED IN THIS REVISION] SICK was cited to the wrong paper.** The bibliography pointed at *SemEval-2014 Task 1*, the shared task, while every claim attached to the citation (9,840 pairs, relatedness and inference over the same pairs, the only corpus carrying both) belongs to the LREC resource paper, *A SICK Cure for the Evaluation of Compositional Distributional Semantic Models* (Marelli et al., LREC 2014, 216-223). The citation is load-bearing: SICK is the only corpus on which alpha can be calibrated.

2. **[MAJOR, FIXED] The paper's title claim was attached to the wrong column.** Section 7.1 said the neutral column of Table 7 "is the one the title of this paper is about", while Figure 3's own caption states that SICK neutrals are related captions, not unrelated sentences. The property the title names is evidenced by the product form and by the generated unrelated suite, not by SICK neutrals.

3. **[MAJOR, FIXED] The property the title names is never measured on natural unrelated pairs,** and this was not among the limitations. It now is: no corpus annotates genuinely unrelated pairs for meaning preservation, so this half of the claim rests on the construction and on self-generated evidence.

4. **[MODERATE, FIXED] A headline number was attributed to the wrong quantity.** "Against the grid mean, fine-tuning adds +22.7 points of macro-F1" is 91.10 minus 68.36, which is the best cell against the off-the-shelf head. The grid mean is 87.70 and would give +19.3.

5. **[MODERATE, FIXED] The +22.7 was not checkable from any table,** because the off-the-shelf head's macro-F1 appeared nowhere. It is now in the caption of Table 6.

6. **[MODERATE, FIXED] Table 5 shows three of seven encoders without saying so,** while the same paragraph quoted a range measured over all seven.

7. **[MODERATE, REMAINS] The macro-F1 comparison with the off-the-shelf head is not label-matched.** The published head never saw the mapping from VitaminC's FACT3 labels onto the paper's three classes, so part of the 22.7-point gap is convention rather than capability. The revision now says this and notes that the AUC gap, which is threshold-free and label-order-free, does not suffer from it. A reviewer may still ask for the off-the-shelf head's macro-F1 under an oracle label permutation.

8. **[MINOR, FIXED] Numerical and rounding slips:** the off-the-shelf AUC was 96.60 in Table 6 against 96.62 in the result file; TF-IDF covered 38.9% of the achievable range, not 38.8%; the contradiction share was 84% in the abstract and conclusion against 83.7% everywhere else; mirrored contradictions reach 95.6-96.5%, not "96%"; the abstract and the Availability section still counted "140 runs" after the body had dropped that framing.

## Comments, Suggestions and Typos

- Table 6 is first referenced in Section 6 but placed on page 7. The revision now tells the reader that the three baseline macro-F1 values live in its caption, since the table's own column reports AUC; a reader checking 39.4% against the table would otherwise find 63.50.
- Appendix D was a single sentence hosting a figure and a table that carry an argument. It now states what they show.
- Table 4 reports Cohen's d to two decimals below 10 and one above, mixed within a column, and labels its unit "pp" while the running text says "points".
- The LexFlip reference (arXiv:2609.05296) is about one month old at submission and unrefereed. It is load-bearing in the introduction and in Section 2.1. Both uses are reported as observations rather than as established results, which is the right handling, but a reviewer will notice the weight placed on a preprint.
- The body now ends exactly at the foot of page 8. There is no slack: three added lines push the conclusion over the limit.

## Questions for the Authors

1. Table 8 shows the *published* polarity head accepting 0.15% contradictions against the fine-tuned head's 2.13%, for 1.7 further points of entailment recall. Why is the fine-tuned configuration the headline rather than the published one, given that contribution 1 is stated in terms of contradictions accepted?
2. The AUC figures come from a single run. What is their seed variance, and does the +2.1-point advantage over the off-the-shelf head survive it?
3. What would the off-the-shelf head's three-way macro-F1 be under the best label permutation, so the +22.7 can be read as capability rather than convention?

## Scores

- **Confidence:** 4/5 — the corpora, the statistical procedure, the composition arithmetic and every number quoted in the text were recomputed against the released result files and the generated tables. Not 5: the training code was not executed.
- **Soundness:** 4/5 — claims are supported, the protocol is careful, significance is handled properly, and the two structural confusions found in this pass were in the prose rather than in the measurements. Held at 4 by weakness 7 and by the single-run AUC.
- **Excitement:** 4/5 — that a deployed metric orders polarity no better than token overlap is a result that changes how a community reads its own numbers, and the composition is cheap enough to be adopted. Not 5: the construction assembles existing components.
- **Overall Assessment:** 4/5 (Conference) — no blocking weakness survives this revision; a measured diagnosis and a reproducible solution with its costs stated.
- **Reproducibility:** 4/5 — hyperparameters, seeds, splits, hardware and budget documented, artefacts promised, and the analysis script generates every table and figure. Not 5 until the repository is public.
- **Datasets:** 4/5 — merged, stratified, leakage-controlled polarity corpus with derived-polarity augmentation, loaders released separately.
- **Software:** 4/5 — corpus builder, training, composition, and the analysis that produces every number in the paper.

## Best Paper: No

Strong diagnosis and careful engineering, but the contribution extends existing components, and the scale is validated against necessary conditions rather than against human judgement.

## Limitations and Societal Impact

Present, unnumbered, after the conclusion, and substantial: seven named limitations, each carrying a measurement, including three most authors would omit, namely that the sign rests entirely on one head, that the composition correlates worse than the magnitude alone on every human-rated quantity available, and that the title's property is evidenced on generated pairs. The Ethical Considerations section correctly identifies that the failure mode reaches the reader rather than the engineer.

## Ethical Concerns

Needs Ethics Review: **No**. Public corpora, no human subjects, no personal data; bias inheritance acknowledged and cited.

## Knowledge of Author Identity

No educated guess required. The paper extends a named released metric and cites two papers sharing a first author, one of them that metric. Both are cited in the third person, which is permitted. The risk is inherent to the topic rather than a violation.

---

*Decision-support draft. ARR forbids submitting a review written by generative AI and forbids uploading a confidential manuscript to a third-party service. Read, verify, own and reformulate before any real submission; never paste verbatim.*
