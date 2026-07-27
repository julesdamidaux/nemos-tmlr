"""Score already-generated images with one automatic metric.

Reads a dataset's `metadata.json` (written by `create_dataset_T2I.py`) and writes a sibling
file holding the same entries with the chosen metric's per-image scores:

    --metric imagereward  ->  <dataset>/metadata_IR.json   key: image_reward_scores
    --metric hpsv2        ->  <dataset>/metadata_HPS.json  key: hps_scores

Usage (from inside scripts/create_dataset/):

    python score_images.py --dataset carrot-bowl --metric imagereward
    python score_images.py --dataset carrot-bowl --metric hpsv2

No score is ever invented: an image that cannot be loaded, or a prompt the scorer raises on,
yields `null` in the output.
"""
import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

# metric name -> (output-file suffix, key holding the scores in the output JSON)
METRICS = {
    "imagereward": ("IR", "image_reward_scores"),
    "hpsv2": ("HPS", "hps_scores"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="dataset name, e.g. flowers / ms-coco / carrot-bowl / flickr")
    p.add_argument("--metric", required=True, choices=sorted(METRICS),
                   help="which automatic metric to compute")
    p.add_argument("--root", default=".",
                   help="directory containing <dataset>/ (default: current directory)")
    p.add_argument("--input", default=None,
                   help="input metadata file (default: <root>/<dataset>/metadata.json)")
    p.add_argument("--output", default=None,
                   help="output file (default: <root>/<dataset>/metadata_<SUFFIX>.json)")
    p.add_argument("--hps-version", default="v2.1",
                   help="HPSv2 checkpoint variant passed to hpsv2.score (--metric hpsv2 only)")
    return p.parse_args()


def get_scorer(metric, hps_version):
    """Return `scorer(prompt, images) -> list[float]` for the requested metric."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if metric == "imagereward":
        from ImageReward import load

        model = load("ImageReward-v1.0")
        model.device = device
        print(f"ImageReward-v1.0 loaded on device: {model.device}")
        return lambda prompt, images: model.score(prompt, images)

    if metric == "hpsv2":
        import hpsv2

        print(f"Scoring with HPSv2 ({hps_version}) on device: {device}")
        return lambda prompt, images: hpsv2.score(images, prompt, hps_version=hps_version)

    raise ValueError(f"Unrecognized metric: {metric}")


def load_image(img_path):
    try:
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Warning: could not load {img_path}: {e}")
        return None


def main():
    args = parse_args()
    suffix, score_key = METRICS[args.metric]

    image_root = os.path.join(args.root, args.dataset)
    input_path = args.input or os.path.join(image_root, "metadata.json")
    output_path = args.output or os.path.join(image_root, f"metadata_{suffix}.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scorer = get_scorer(args.metric, args.hps_version)

    new_data = []
    for entry in tqdm(data, desc=f"Computing {args.metric} scores"):
        prompt = entry["prompt"]
        filenames = entry["filenames"]

        img_paths = [os.path.join(image_root, fname) for fname in filenames]
        images = [img for img in (load_image(p) for p in img_paths) if img is not None]

        if images:
            try:
                scores = scorer(prompt, images)
            except Exception as e:
                print(f"Warning: scorer failed on prompt: {prompt} -> {e}")
                scores = [None] * len(images)
        else:
            scores = [None] * len(filenames)

        if len(scores) != len(filenames):
            print(f"Warning: {len(scores)} scores for {len(filenames)} filenames on prompt: {prompt} "
                  f"(some images failed to load; scores are not aligned with filenames)")

        new_data.append({
            "prompt": prompt,
            "model": entry["model"],
            "filenames": filenames,
            score_key: scores,
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
