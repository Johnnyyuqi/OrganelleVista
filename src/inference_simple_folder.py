import argparse
import os
import json
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from pretrain_pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import build_transform 

def extract_state_dict(ckpt):
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "model" in ckpt and ckpt["model"].get("format") == "full_scratch_v1":
        return ckpt["model"]["full_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    raise ValueError("Unsupported checkpoint format.")

def detect_skip_conv_usage(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt)
    keys = [k[7:] if k.startswith("module.") else k for k in state_dict.keys()]
    return any("vae.decoder.skip_conv_" in k for k in keys)

def run_inference_with_prompts(ckpt_path, input_dir, json_path, output_dir, resolution=512):
    # Load the filename-to-prompt mapping.
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Prompt JSON file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        prompts_dict = json.load(f)
    print(f"Loaded {len(prompts_dict)} image-to-prompt mappings from JSON.")

    use_skip_conv = detect_skip_conv_usage(ckpt_path)
    print(f"[info] detected skip conv weights in checkpoint: {use_skip_conv}")

    model = Pix2Pix_Turbo(
        lora_rank_unet=8, 
        lora_rank_vae=4, 
        train_from_scratch=False, 
        resume_from_scratch_ckpt=True,
        enable_skip_conv=use_skip_conv,
    )

    if hasattr(model.vae.decoder, 'skip_conv_1'):
        layer = model.vae.decoder.skip_conv_1
        weight = layer.base_layer.weight if hasattr(layer, 'base_layer') else layer.weight
        print(f"Before Loading - Skip Conv 1 Mean: {weight.data.mean().item():.6f}")
        print(f"Before Loading - Skip Conv 1 Std:  {weight.data.std().item():.6f}")
    else:
        print("[info] skip conv layers are disabled for this checkpoint.")
    
    print(f"Loading checkpoint: {ckpt_path}...")
    model.load_full_checkpoint_for_scratch_training(ckpt_path)
    print("--- Checkpoint weights loaded ---")
    if hasattr(model.vae.decoder, 'skip_conv_1'):
        layer = model.vae.decoder.skip_conv_1
        weight = layer.base_layer.weight if hasattr(layer, 'base_layer') else layer.weight
        print(f"AFTER Loading  - Skip Conv 1 Mean: {weight.data.mean().item():.6f}")
        print(f"AFTER Loading  - Skip Conv 1 Std:  {weight.data.std().item():.6f}")
    model.cuda().eval()
    model.requires_grad_(False)

    T_res = build_transform(f"resize_{resolution}")
    os.makedirs(output_dir, exist_ok=True)


    print("Starting inference...")
    for img_name, caption in tqdm(prompts_dict.items()):
        img_path = os.path.join(input_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: image {img_name} was not found in the input directory; skipping.")
            continue

        input_pil = Image.open(img_path).convert("RGB")
        img_t = T_res(input_pil)
        img_t = transforms.ToTensor()(img_t).unsqueeze(0).cuda()
        
        tokenized_prompt = model.tokenizer(
            caption, 
            max_length=model.tokenizer.model_max_length,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.unsqueeze(0).cuda()

        with torch.no_grad():
            output_t = model(img_t, prompt_tokens=tokenized_prompt, deterministic=True)
        
        output_img = (output_t[0].cpu() * 0.5 + 0.5).clamp(0, 1)
        output_pil = transforms.ToPILImage()(output_img)
        
        save_path = os.path.join(output_dir, img_name)
        output_pil.save(save_path)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Run folder inference using a full Pix2Pix-Turbo checkpoint and a filename-to-prompt JSON mapping."
    )
    parser.add_argument("--ckpt_path", required=True, help="Full checkpoint (.pt) or model_state_dict snapshot (.pth).")
    parser.add_argument("--input_dir", required=True, help="Directory containing source images listed in the prompt JSON.")
    parser.add_argument("--json_path", required=True, help="JSON object mapping image filenames to text prompts.")
    parser.add_argument("--output_dir", required=True, help="Directory for generated images, saved with their original filenames.")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=512,
                        help="Square output resolution supported by the existing image transforms (default: 512).")
    return parser.parse_args(input_args)


if __name__ == "__main__":
    args = parse_args()
    run_inference_with_prompts(
        args.ckpt_path, args.input_dir, args.json_path, args.output_dir,
        resolution=args.resolution,
    )
