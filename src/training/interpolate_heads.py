"""Weight-space interpolation between an off-the-shelf NLI head and its v4 fine-tune.

The v4 heads gain the lexical guard rails and lose general inference on DACCORD. WiSE-FT
(Wortsman et al., 2022) trades the two without training: the weights
``(1 - alpha) * base + alpha * tuned`` sweep from the off-the-shelf head (``alpha = 0``) to
the fine-tuned one (``alpha = 1``), and the question is whether some alpha keeps both
DACCORD at the 1a bar and co-hyponyms above zero.

The two endpoints must agree on what each output row means. ``joeddav/xlm-roberta-large-xnli``
puts contradiction first, the v4 heads last, so the base head is permuted into the v4 order
first, with the same plan the training used (``train_polarity.head_reuse_plan``); averaging
unaligned rows would mix contradiction with entailment and still produce a number.

Run::

    PYTHONPATH=src python src/training/interpolate_heads.py \
        --base joeddav/xlm-roberta-large-xnli \
        --tuned results/polarity/xlmr-large-nli-v4en/seed42/model \
        --alpha 0.5 --out results/polarity/wise/v4-wise-xlmr-en-a0.5-seed42/model
"""

from __future__ import annotations

from typing import Any

import click

try:
    from training.train_polarity import CLASS_NAMES, head_reuse_plan, output_layer
except ImportError:  # pragma: no cover - script run from inside ``src/training``.
    from train_polarity import CLASS_NAMES, head_reuse_plan, output_layer  # type: ignore


def interpolate(base: dict[str, Any], tuned: dict[str, Any], alpha: float) -> dict[str, Any]:
    """``(1 - alpha) * base + alpha * tuned`` on every floating tensor.

    Integer buffers (position ids, token types) are copied from *tuned*: they are indices,
    not weights, and an average of two indices is not an index.

    Raises:
        ValueError: If *alpha* is outside ``[0, 1]`` or the two state dicts do not hold the
            same tensors with the same shapes.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha doit etre dans [0, 1], recu {alpha}")
    if base.keys() != tuned.keys():
        missing = sorted(base.keys() ^ tuned.keys())[:5]
        raise ValueError(f"les deux modeles n'ont pas les memes tenseurs : {missing}")
    out: dict[str, Any] = {}
    for name, weight in tuned.items():
        if base[name].shape != weight.shape:
            raise ValueError(f"{name} : {tuple(base[name].shape)} contre {tuple(weight.shape)}")
        out[name] = (1.0 - alpha) * base[name] + alpha * weight if weight.is_floating_point() else weight.clone()
    return out


def aligned_base(checkpoint: str):
    """The off-the-shelf head, its output rows permuted into :data:`CLASS_NAMES` order.

    Raises:
        ValueError: If the checkpoint's head is not an inference head that can be aligned.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    reusable, permutation = head_reuse_plan(getattr(model.config, "id2label", None))
    if not reusable or permutation is None:
        raise ValueError(f"{checkpoint} n'a pas de tete d'inference alignable")
    layer = output_layer(model)
    with torch.no_grad():
        index = torch.tensor(permutation, device=layer.weight.device)
        layer.weight.copy_(layer.weight[index])
        if layer.bias is not None:
            layer.bias.copy_(layer.bias[index])
    model.config.id2label = dict(enumerate(CLASS_NAMES))
    model.config.label2id = {name: index for index, name in enumerate(CLASS_NAMES)}
    return model


@click.command()
@click.option("--base", required=True, help="Off-the-shelf checkpoint the v4 head was trained from.")
@click.option("--tuned", required=True, help="Folder of the fine-tuned v4 head.")
@click.option("--alpha", type=float, required=True, help="Weight of the fine-tuned head, 0 to 1.")
@click.option("--out", required=True)
def main(base: str, tuned: str, alpha: float, out: str) -> None:
    """Ecrit la tete interpolee entre la tete prete a l'emploi et sa version v4."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(tuned)
    reference = aligned_base(base)
    model.load_state_dict(interpolate(reference.state_dict(), model.state_dict(), alpha))
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(tuned).save_pretrained(out)
    click.echo(f"alpha {alpha} : {out}")


if __name__ == "__main__":
    main()
