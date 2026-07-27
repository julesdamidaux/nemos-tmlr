"""Text-to-image pipelines for the six candidate models.

Single source of truth for how each model is loaded and sampled: both the dataset builder
(`scripts/create_dataset/create_dataset_T2I.py`) and the online reward source import
:func:`build_generator` from here, so images are generated identically either way.

Identifiers are not pinned to a revision — see `docs/reproduce.md` ("Model versions").
"""
import torch

MODEL_IDS = {
    "Sana": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    "Unidiffuser": "thu-ml/unidiffuser-v1",
    "LCM": "SimianLuo/LCM_Dreamshaper_v7",
    "Koala": "etri-vilab/koala-lightning-700m",
    "SDXL-Turbo": "stabilityai/sdxl-turbo",
    "SSD-1B": "segmind/SSD-1B",
}


def build_generator(model_name, device):
    """Return `generator(prompt) -> PIL.Image` for `model_name`, loaded onto `device`."""
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline

    if model_name == "Sana":
        from diffusers import SanaPipeline
        pipe = SanaPipeline.from_pretrained(
            MODEL_IDS["Sana"],
            torch_dtype=torch.bfloat16
        ).to(device)
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        return lambda prompt: pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=10
        ).images[0]

    if model_name == "LCM":
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_IDS["LCM"],
            torch_dtype=torch.float16
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return lambda prompt: pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=8.0,
            lcm_origin_steps=50,
            output_type="pil"
        ).images[0]

    if model_name == "Unidiffuser":
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_IDS["Unidiffuser"],
            torch_dtype=torch.float16
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return lambda prompt: pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=10
        ).images[0]

    if model_name == "SDXL-Turbo":
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_IDS["SDXL-Turbo"],
            torch_dtype=torch.float16
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return lambda prompt: pipe(
            prompt=prompt,
            num_inference_steps=2,
            guidance_scale=0.0
        ).images[0]

    if model_name == "SSD-1B":
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_IDS["SSD-1B"],
            torch_dtype=torch.float16
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return lambda prompt: pipe(
            prompt=prompt,
            num_inference_steps=10,
            guidance_scale=7.5
        ).images[0]

    if model_name == "Koala":
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_IDS["Koala"],
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return lambda prompt: pipe(prompt, num_inference_steps=8).images[0]

    raise ValueError(f"Unrecognized model: {model_name}")


def is_black_image(image, threshold=5):
    """True if `image` is (nearly) all black — some pipelines fail this way."""
    grayscale = image.convert("L")
    hist = grayscale.histogram()
    avg_pixel = sum(i * hist[i] for i in range(256)) / sum(hist)
    return avg_pixel < threshold
