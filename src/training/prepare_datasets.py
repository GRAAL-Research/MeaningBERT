"""Pre-generate all dataset variants to disk for fast training.

Run once before sweeps:
    python prepare_datasets.py --output_dir ./data

This creates:
    data/
        meaning/                        - base dataset (853 train)
        meaning_with_swap/              - swap augmentation (4267 train)
        meaning_with_back_translation/  - swap + back-translation (~12801 train)
        meaning_holdout_identical/
        meaning_holdout_unrelated/
"""
import argparse
import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import MarianMTModel, MarianTokenizer


def back_translate_batch(
    texts: list[str],
    en_fr: tuple,
    fr_en: tuple,
    batch_size: int = 32,
) -> list[str]:
    """Back-translate a list of English texts via French (EN -> FR -> EN)."""
    en_fr_tokenizer, en_fr_model = en_fr
    fr_en_tokenizer, fr_en_model = fr_en

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # EN -> FR
        inputs = en_fr_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        fr_tokens = en_fr_model.generate(**inputs, max_length=512)
        fr_texts = en_fr_tokenizer.batch_decode(fr_tokens, skip_special_tokens=True)

        # FR -> EN
        inputs = fr_en_tokenizer(fr_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        en_tokens = fr_en_model.generate(**inputs, max_length=512)
        en_texts = fr_en_tokenizer.batch_decode(en_tokens, skip_special_tokens=True)

        results.extend(en_texts)
        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

    return results


def create_back_translated_split(dataset, en_fr, fr_en, batch_size: int = 32) -> Dataset:
    """Add back-translated pairs to a dataset split."""
    originals = dataset["original"]
    simplifications = dataset["simplification"]
    labels = dataset["label"]

    print(f"  Back-translating {len(originals)} originals...")
    bt_originals = back_translate_batch(originals, en_fr, fr_en, batch_size)

    print(f"  Back-translating {len(simplifications)} simplifications...")
    bt_simplifications = back_translate_batch(simplifications, en_fr, fr_en, batch_size)

    augmented = Dataset.from_dict({
        "original": bt_originals + originals,
        "simplification": simplifications + bt_simplifications,
        "label": labels + labels,
    })

    return concatenate_datasets([dataset, augmented])


def main():
    parser = argparse.ArgumentParser(description="Pre-generate all CSMD dataset variants to disk.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Root directory for all dataset variants.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for back-translation.",
    )
    parser.add_argument(
        "--skip_back_translation",
        action="store_true",
        help="Skip back-translation (only prepare base + swap + holdouts).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Base dataset (no augmentation)
    print("=== Downloading base dataset ===")
    meaning = load_dataset("davebulaval/CSMD", "meaning")
    meaning_path = os.path.join(output_dir, "meaning")
    meaning.save_to_disk(meaning_path)
    print(f"  Saved to {meaning_path} (train={len(meaning['train'])})")

    # 2. Swap augmentation (commutative property)
    print("\n=== Downloading swap-augmented dataset ===")
    meaning_swap = load_dataset("davebulaval/CSMD", "meaning_with_data_augmentation")
    meaning_swap_path = os.path.join(output_dir, "meaning_with_swap")
    meaning_swap.save_to_disk(meaning_swap_path)
    print(f"  Saved to {meaning_swap_path} (train={len(meaning_swap['train'])})")

    # 3. Back-translation on top of swap
    if not args.skip_back_translation:
        print("\n=== Loading translation models ===")
        en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        en_fr = (en_fr_tokenizer, en_fr_model)
        fr_en = (fr_en_tokenizer, fr_en_model)

        print("\n=== Generating back-translated dataset ===")
        bt_train = create_back_translated_split(meaning_swap["train"], en_fr, fr_en, args.batch_size)
        meaning_bt = DatasetDict({
            "train": bt_train,
            "dev": meaning_swap["dev"],
            "test": meaning_swap["test"],
        })
        meaning_bt_path = os.path.join(output_dir, "meaning_with_back_translation")
        meaning_bt.save_to_disk(meaning_bt_path)
        print(f"  Saved to {meaning_bt_path} (train={len(bt_train)})")
    else:
        print("\n=== Skipping back-translation ===")

    # 4. Holdout datasets
    print("\n=== Downloading holdout datasets ===")
    holdout_identical = load_dataset("davebulaval/CSMD", "meaning_holdout_identical")
    holdout_identical_path = os.path.join(output_dir, "meaning_holdout_identical")
    holdout_identical.save_to_disk(holdout_identical_path)
    print(f"  Saved to {holdout_identical_path} (test={len(holdout_identical['test'])})")

    holdout_unrelated = load_dataset("davebulaval/CSMD", "meaning_holdout_unrelated")
    holdout_unrelated_path = os.path.join(output_dir, "meaning_holdout_unrelated")
    holdout_unrelated.save_to_disk(holdout_unrelated_path)
    print(f"  Saved to {holdout_unrelated_path} (test={len(holdout_unrelated['test'])})")

    print("\n=== Done ===")
    print(f"All datasets saved to {output_dir}/")
    for name in sorted(os.listdir(output_dir)):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full):
            print(f"  {name}/")


if __name__ == "__main__":
    main()
