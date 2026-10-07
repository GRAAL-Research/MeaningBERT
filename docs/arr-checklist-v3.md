# Responsible NLP Checklist (ARR) — « Unrelated Is Not Opposed »

Réponses à copier dans le formulaire OpenReview au dépôt. Les sections renvoient à `paper/v3/main.tex`.

## A. For every submission

- **A1. Did you describe the limitations of your work?** Yes. Section « Limitations » (seven limitations: no signed gold standard, MoNLI cost of augmentation, single corpus for the composition, correlation cost, single head deciding the sign and lexicon design bias, unrelated pairs only generated, English sentence level).
- **A2. Did you discuss any potential risks of your work?** Yes. Section « Ethical Considerations »: a metric that scores a negation as preserved meaning lets the error reach the reader; inherited corpus biases.
- **A3. Do the abstract and introduction summarize the paper's main claims?** Yes. Abstract and Section 1, contributions list.

## B. Did you use or create scientific artifacts?

Yes.

- **B1. Did you cite the creators of artifacts you used?** Yes. VitaminC, SICK, MoNLI, NaN-NLI, CSMD/MeaningBERT, BERT, RoBERTa, DeBERTaV3 (Sections 4 and 5).
- **B2. Did you discuss the license or terms for use and/or distribution?** Yes. Section « Availability »: the merged corpus inherits the most restrictive source licence and is redistributed under CC BY-NC-SA 4.0; loaders released separately.
- **B3. Did you discuss if your use of existing artifacts was consistent with their intended use?** Yes. All corpora are research NLI or meaning preservation datasets used for research evaluation and training (Section 4).
- **B4. Did you discuss the steps taken to check whether the data contains personally identifying information or offensive content?** N/A in the paper: public research corpora of Wikipedia claims and image captions; no new collection. Answer « No, existing public corpora, no new data collected ».
- **B5. Did you provide documentation of the artifacts?** Yes. Tables 1 and 2, Appendix A (construction, leakage), Section 7.2 (lexical augmentation, lexicon split).
- **B6. Did you report relevant statistics like the number of examples, details of train/test/dev splits?** Yes. Tables 1, 2 and 7; Section 7.2 for LEX (192,060) and AUG+LEX (267,060) training sizes.

## C. Did you run computational experiments?

Yes.

- **C1. Did you report the number of parameters in the models used, the total computational budget, and computing infrastructure used?** Yes. Appendix C (hardware: three Pascal cards for the grid, three RTX 6000 Ada for the lexical conditions; about 500 GPU-hours for the grid, per-run times). Model sizes are those of the public base and large checkpoints.
- **C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values?** Yes. Section 5 and Table 10; one setting for all runs, no per-encoder tuning.
- **C3. Did you report descriptive statistics about your results?** Yes. Mean and standard deviation over ten random seeds, Welch and paired t-tests, Cohen's d, Holm-Bonferroni, bootstrap intervals (Sections 5 to 7).
- **C4. If you used existing packages, did you report the implementation, model, and parameter settings used?** Yes. Hugging Face transformers 4.57 (Appendix C), public checkpoints named in Section 5 and Table 3.

## D. Did you use human annotators or research with human subjects?

No new annotation. Human ratings come from existing corpora (SICK relatedness, CSMD). Answer « No » to D1 to D5.

## E. Did you use AI assistants?

Yes.

- **E1. Did you include information about your use of AI assistants?** Yes. Section « Use of Large Language Models »: Claude assisted with linguistic refinement, typography and LaTeX formatting; authors retained control of scientific content, analysis and conclusions.
