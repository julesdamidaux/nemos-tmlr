import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from nemos.rewards.pipelines import build_generator, is_black_image

# ================================
# PARAMETERS TO SET HERE
# ================================
model_names = [
    "Koala",
    "Sana",
    "LCM",
    "Unidiffuser",
    "SDXL-Turbo",
    "SSD-1B"
]
num_generations = 5
T = 1000
output_dir = "flowers"
prompts_path = "prompts/flowers.json"

# Initialize device and CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_score(prompt, image):
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
    img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    return 100 * (img_emb * txt_emb).sum(dim=-1).item()

def main():
    torch.cuda.empty_cache()
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")

    # Load or initialize metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    else:
        all_metadata = []

    existing_prompts_global = {entry["prompt"] for entry in all_metadata}

    for model_name in model_names:
        torch.cuda.empty_cache()
        print(f"\n▶ Starting generation for model: {model_name}")
        generator = build_generator(model_name, device)
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Prompts already done by this model and by others
        done_by_model = {e["prompt"] for e in all_metadata if e["model"] == model_name}
        done_by_others = {e["prompt"] for e in all_metadata if e["model"] != model_name}

        # 1) Complete on prompts produced by others
        missing_prompts = sorted(done_by_others - done_by_model)

        # 2) If insufficient, draw from list of new prompts
        if len(missing_prompts) < T:
            with open(prompts_path, "r", encoding="utf-8") as f:
                new_prompts = json.load(f)
            # take those not already in all_metadata
            candidates = [p for p in new_prompts if p not in existing_prompts_global]
            missing_prompts.extend(candidates)
            missing_prompts = missing_prompts[:T]

        print(f"→ {len(missing_prompts)} prompts to generate for {model_name}")

        # Calculate starting index for file numbering
        existing_files = [f for f in os.listdir(model_dir) if f.endswith(".png")]
        idx = max([int(f.split("_")[1].split(".")[0]) for f in existing_files], default=-1) + 1

        for prompt in tqdm(missing_prompts[:T], desc=model_name):
            filenames, scores = [], []
            for _ in range(num_generations):
                try:
                    img = generator(prompt)
                    if is_black_image(img):
                        continue
                    fname = f"{model_name}/img_{idx:05d}.png"
                    path = os.path.join(output_dir, fname)
                    img.save(path)
                    score = compute_clip_score(prompt, img)
                    filenames.append(fname)
                    scores.append(round(score, 2))
                    idx += 1
                except Exception as e:
                    print(f"Error ({model_name}) on \"{prompt}\": {e}")
                    continue
            if filenames:
                all_metadata.append({
                    "prompt": prompt,
                    "model": model_name,
                    "filenames": filenames,
                    "clip_scores": scores
                })
                existing_prompts_global.update(filenames)

    # Save updated metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Generation completed for all models. Total entries: {len(all_metadata)}")

if __name__ == "__main__":
    main()