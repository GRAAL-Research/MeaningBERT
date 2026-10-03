# ARR Review — Unrelated Is Not Opposed: A Signed Scale for Meaning Preservation

**Domain / Track:** evaluation of generation (meaning-preservation metrics for text simplification), with natural language inference as a component
**Review passes:** 3

## Compliance pre-check

Long paper. Body (Sections 1–7) ends on page 7; Limitations, Ethical Considerations and references follow and are outside the limit. A Limitations section is present and substantial. Three appendices document corpus construction, training and compute budget. No page-limit violation. One anonymity concern is raised below.

## Paper Summary

The paper argues that meaning-preservation metrics, which emit a single non-negative score, cannot express opposition: contradicting sentences share subject and lexical material, so a magnitude metric reads that overlap as preserved meaning. The authors measure the failure on a released metric, which ranks agreement above contradiction at 60.8% AUC on a 12,000-pair balanced split and at 48.0% on SICK, and show that thresholded token overlap reaches 63.5% on the same split. They propose a signed scale on [-100, 100] obtained by multiplying a magnitude head by (1 - alpha * p_contra) from a three-way polarity head, the product form ensuring that unrelated pairs stay near zero. A 140-run grid (seven encoders x ten seeds x two corpus conditions) on a merged, class- and corpus-stratified, leakage-controlled corpus shows a polarity head reaching 91.1% macro-F1 and 98.7% AUC; derived-polarity augmentation leaves task performance unchanged while raising generated sanity-suite accuracy from 79–90% to 98–99% on all seven encoders; and inference pretraining buys monotonicity reasoning rather than aggregate accuracy at large size. The composition, calibrated on the SICK development half and reported on its test half, selects alpha = 2 and places 83.7% of contradictions below zero where the magnitude alone places none.

## Summary of Strengths

1. **The diagnosis is measured, not asserted, and the measurement is damaging.** Reporting that a thresholded token overlap (63.5% AUC) orders polarity better than a metric trained on human preservation judgements (60.8%) is the single most convincing piece of evidence in the paper, and it is exactly the right baseline to have run. Table 6 is well constructed.
2. **Experimental hygiene is above the norm for this subfield.** Ten seeds per cell with standard deviations reported throughout, Welch's t-test with Cohen's d beside every p-value, Holm-Bonferroni across the test family, class- and corpus-stratified splits, and source-sentence leakage removed in both directions with the counts given in Appendix A. The paper explicitly reports that a 0.1-point difference reaches significance and means nothing, which is a rare and welcome form of self-restraint.
3. **The design rationale is argued rather than assumed.** The product form, the rejection of MQM-style typed penalties, the rejection of end-to-end supervision, and the rejection of PAWS are each given a reason tied to a failure the design is meant to avoid.
4. **Honest negative and bounding results.** The off-the-shelf comparison (96.6% AUC without training) is reported even though it narrows the contribution, the augmentation/sanity-suite confound is disclosed in the body rather than buried, and the MoNLI failure is diagnosed precisely (93% of true entailments predicted neutral, 0.2% predicted contradiction) instead of being reported as a bare low number.
5. **Reproducibility.** Hyperparameters, seeds, splits, hardware and per-run wall-clock are all documented, and every table and figure is stated to be generated from the released result files.

## Summary of Weaknesses

1. **[MAJOR — anonymity]** The paper self-identifies. Section 1 writes "the failure is measurable on the metric we build on" immediately before naming MeaningBERT (Beauchemin et al., 2023), and Table 6 labels two retrained variants "(ours)" while the text says they were trained "on the same corpus with the same recipe". A reviewer can infer with near certainty that the authors are the authors of the cited work. **Fix:** replace "the metric we build on" with a neutral phrase, and relabel the Table 6 rows as retrained variants of the released metric rather than "(ours)".

2. **[MAJOR — unquantified headline]** The 98.7% AUC and the per-corpus breakdown (91.6% SICK vs 90.4% VitaminC) come from a single run, while every other number in the paper is a mean over ten seeds. The paper does not say so. The +2.1 AUC advantage over the off-the-shelf head is therefore reported without any estimate of seed variance, and it is one of the three claims in the contribution list. **Fix:** state explicitly that these are single-seed measurements, or compute them over the ten seeds.

3. **[MAJOR — the composition loses on every human-correlation measurement]** Table 7 shows the magnitude alone at r = 0.852 against 0.795 for the composition; Section 7 reports 0.697 against 0.637 on the simplification corpus. On both human-rated quantities the signed scale correlates *worse* than the metric it extends. The paper presents these as two separate small costs, several pages apart, and never states the combined picture. A reviewer will. The defence available to the authors is that correlation with a non-negative human rating cannot reward a correct negative score, and that defence belongs in the paper. **Fix:** state the tension explicitly in Limitations and give the argument.

4. **[MODERATE — overstated baseline reading]** "Roughly half the task is therefore surface statistics of the corpus" (Section 6). Against a floor of 16.7 and a ceiling of 91.1, TF-IDF at 45.6 covers 38.8% of the achievable range and token overlap 30.5%. "Roughly half" is not what the numbers say. **Fix:** report the range-normalised share or soften to "roughly two fifths".

5. **[MODERATE — dev and test numbers mixed]** Section 7 states that both magnitude models "send the same 84.6% of contradictions below zero" two paragraphs after Table 7 reports 83.7% on test. The 84.6% is a development-split figure; the split is not named. **Fix:** name the split.

6. **[MODERATE — robustness claim is narrow]** "The slope is a property of the scale rather than of the magnitude model it is fitted with" is supported by two checkpoints that are both variants of the same released metric trained on the same corpus. That is a weaker test than the sentence claims. **Fix:** soften the claim, or add a structurally different magnitude estimator such as BERTScore.

7. **[MODERATE — the figure appears to contradict the scale's definition]** Figure 3 shows SICK neutral pairs peaking near +50, while Section 3 defines 0 as "no relation". SICK neutral pairs are related captions rather than unrelated sentences, so the behaviour is correct, but nothing in the figure or its caption says so and the reader is left with an apparent contradiction. **Fix:** one clarifying sentence in the caption.

8. **[MINOR — arithmetic]** "Fine-tuning adds +2.1 points of AUC and +22.8 points of macro-F1". The grid mean gives 91.1 against 68.4, that is +22.7; the single-seed value gives +22.9. Neither is 22.8. **Fix:** pick a basis and state it.

9. **[MINOR]** Table 6 contains a row ("Majority class") with no value in either column. Either give its AUC as undefined with a footnote or drop the row and keep the macro-F1 figures in the caption.

10. **[MINOR]** Section 6 refers to Table 6 before Tables 4 and 5 are discussed, so the reader meets the tables out of order.

## Comments, Suggestions and Typos

- Section 6, "The headline transfers across corpora": "two genres that share little beyond their label set" is a strong claim about genre that is not supported by any measurement; consider "two corpora of different provenance".
- Table 3 caption: "Bold marks the best value in each column, higher being better everywhere" is correct but the bold MoNLI value sits on a RAW row while the paper's preferred condition is AUG; a half-sentence would prevent a misreading.
- Table 8 would read better with the absolute number of contradictions (712) in the row label rather than only in the caption.
- Figure 2: the two panels do not share an x range, which the caption states; consider adding a visual cue, since the eye compares the two panels' positions before reading the axis.
- Appendix C notes that two encoders "wedge" a specific GPU and that this was handled by exclusion rather than diagnosis. This is honest and appropriate, but it should be cross-referenced from the Limitations "Scope" paragraph, which currently mentions only the fp32 constraint.
- The abstract gives 84% for contradictions below zero while Table 7 gives 83.7%; rounding is fine but the body should use one figure consistently.

## Questions for the Authors

1. Are the 98.7% AUC and the per-corpus figures from one of the 140 grid runs, or from a separate run of the same configuration? What is their seed variance?
2. On both human-rated quantities (SICK relatedness, preservation ratings) the composition correlates worse than the magnitude alone. What evidence establishes that the signed scale is a better metric, as opposed to a metric that encodes an additional distinction at a cost?
3. Would a human study on even 200 pairs, asking annotators to rate "how strongly these sentences say the opposite", change the calibration of alpha away from 2?

## Scores

- **Confidence:** 4/5 — the subfield, the corpora and the statistical methodology were checked in detail; the composition's arithmetic was verified against the reported tables. Not 5, since the released code was not executed.
- **Soundness:** 3.5/5 — the main claims are supported, the experimental protocol is careful, and significance is handled properly. Held below 4 by the single-seed headline (weakness 2) and by the unaddressed correlation tension (weakness 3), both of which are corrigible in text or with modest additional computation.
- **Excitement:** 4/5 — the observation that a deployed metric orders polarity no better than token overlap is the kind of result that changes how a community reads its own numbers, and the composition is simple enough to be adopted. Not 5, because the construction is an extension of existing components rather than a new capability.
- **Overall Assessment:** 3.5/5 (Borderline Conference) — no blocking weakness, a genuinely useful diagnosis, and a solution that is cheap and reproducible. The single-seed headline and the correlation tension keep it from a clean 4; both are fixable in a rebuttal.
- **Reproducibility:** 4/5 — hyperparameters, seeds, splits, hardware and budget are documented and the artefacts are promised under an open licence; the single-seed measurements are the one gap.
- **Datasets:** 4/5 — the merged, stratified, leakage-controlled polarity corpus with derived-polarity augmentation is a reusable resource, and the loaders are released separately so it can be rebuilt.
- **Software:** 4/5 — corpus builder, training, composition and the analysis that generates every table and figure.

## Best Paper: No

The diagnosis is excellent and the engineering is careful, but the contribution extends existing components rather than opening a new direction, and the scale is validated against necessary conditions rather than against human judgement.

## Limitations and Societal Impact

Present and substantial. Five named limitations, each with a measurement rather than a gesture, including one (the sign resting entirely on one head) that follows from the paper's own equation and that most authors would have left unsaid. The missing one is weakness 3 above. The Ethical Considerations section correctly identifies that the failure mode reaches the reader rather than the engineer.

## Ethical Concerns

Needs Ethics Review: **No**. Public corpora, no human subjects, no personal data. The bias inheritance from Wikipedia revisions is acknowledged and cited.

## Knowledge of Author Identity

**Anonymity concern.** "The metric we build on" in Section 1, the "(ours)" labels in Table 6, and the statement that variants were trained "on the same corpus with the same recipe" together identify the authors as the authors of Beauchemin et al. (2023). Reported under weakness 1 rather than as a deliberate violation, since no link or acknowledgement reveals identity directly.

---

*This review is a decision-support draft. ARR forbids submitting a review written by generative AI and forbids uploading a confidential manuscript to a third-party service. It must be read, verified, owned and reformulated before any real submission, and never pasted verbatim.*
