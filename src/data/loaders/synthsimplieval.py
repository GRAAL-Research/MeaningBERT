"""SynthSimpliEval loader for CSMD v2.

Status: BLOCKED, no public data found. See ``RAPPORT.md`` for the full trail.

Source: Liu, Nam, Cui & Swayamdipta, "Evaluation Under Imperfect Benchmarks and Ratings: A
Case Study in Text Simplification", arXiv:2504.09394. The paper describes two disjoint
pools of scored simplifications built on top of the same 260 complex sentences (60 from
SimpEval2022 plus 200 synthetic sentences generated with Qwen 2.5 72B Instruct), simplified
by four Llama 3 models (1B/3B/8B/70B):

1. A **1040-pair, LLM-as-a-jury-rated** pool (all 260 sentences x 4 models), scored by a
   panel of seven LLMs and normalised to 0-1. This is the pool the mission brief describes
   as "environ 1040 simplifications notees". It is not human-judged: it fails
   ``PRODUIT.md``'s locked non-objectif "Pas de LLM-as-judge, ni en baseline ni en
   professeur", so even if it were downloadable it would not qualify as v2 training
   material.
2. An **80-pair, human-judged** pilot (20 of the 260 sentences, 10 from each source, x 4
   models), rated by 3 NLP-expert annotators on a genuine 5-point Likert scale -- a scale
   CONTRACT.md *does* support (``likert5``). This is the pool that would actually qualify.

Neither pool's per-item scores are published:

- The paper's own code repository, https://github.com/jliu7350/text-simplification-benchmark
  (linked from the abstract's first footnote), contains a single one-line README and no
  data files as of 2026-09-19 (checked via the GitHub API tree listing) -- a dead repository
  in the sense of BRIEF.md rule 4.
- The arXiv submission's source tarball (``https://arxiv.org/e-print/2504.09394``) contains
  only LaTeX source and PDF figures, no CSV/JSON data files.
- The paper text itself reports only aggregate statistics (mean scores per model size,
  Spearman/ICC correlations) and a handful of illustrative example sentences with no score
  attached (reproduced in ``tests/fixtures/synthsimplieval/paper_examples.json`` for
  documentation; they carry no ``label`` field because none was ever published).

Per BRIEF.md rule 4, this loader does not fabricate scores for these pairs. ``load()``
raises :class:`DataUnavailableError` unconditionally.
"""

from __future__ import annotations

from datasets import Dataset


class DataUnavailableError(RuntimeError):
    """Raised because no human-judged SynthSimpliEval data could be retrieved publicly."""


_MESSAGE = (
    "SynthSimpliEval: no downloadable per-item score data found. Checked: "
    "https://github.com/jliu7350/text-simplification-benchmark (README-only, no data files "
    "as of 2026-09-19), the arXiv:2504.09394 e-print source tarball (LaTeX and figures only), "
    "and the paper text (aggregate statistics only, no per-pair scores). The one pool "
    "described in the mission brief (~1040 pairs, 0-1 normalised) is LLM-as-a-jury-rated, "
    "not human-judged, and would also violate PRODUIT.md's 'Pas de LLM-as-judge' "
    "non-objectif even if it were retrievable. See RAPPORT.md."
)


def load() -> Dataset:
    """Would return the CONTRACT-compliant SynthSimpliEval dataset; instead raises, by design.

    See the module docstring and ``RAPPORT.md``.

    Returns:
        Never returns; kept for interface parity with the other CSMD v2 loaders.

    Raises:
        DataUnavailableError: Always. No human-judged, per-item SynthSimpliEval data is
            publicly retrievable as of 2026-09-19.
    """
    raise DataUnavailableError(_MESSAGE)
