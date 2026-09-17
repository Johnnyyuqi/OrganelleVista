import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

#import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pretrain_pix2pix_turbo_dpo import Pix2Pix_Turbo
from my_utils.dpo_utils import parse_args_paired_training, parse_args_dpo, DPODataset, PairedDataset, structural_loss
from torch.utils.checkpoint import checkpoint

def get_batch_log_prob(
    model,
    batch,
    weight_dtype,
    latent_reward_weight=1.0,
    image_reward_weight=0.25,
):
    raw_model = model.module if hasattr(model, "module") else model
    model_device = next(raw_model.unet.parameters()).device

    with torch.autocast(device_type="cuda", dtype=weight_dtype):
        x_src = batch["pixel_values_src"].to(device=model_device, dtype=weight_dtype)
        prompt_tokens = batch["input_ids"].to(model_device)
        x_pred = model(x_src, prompt_tokens=prompt_tokens, deterministic=True)

    pred_image = x_pred.float()
    x_good = batch["pixel_values_good"].to(device=model_device, dtype=torch.float32)
    x_bad = batch["pixel_values_bad"].to(device=model_device, dtype=torch.float32)

    image_mse_good = (pred_image - x_good).pow(2).mean(dim=[1, 2, 3])
    image_mse_bad = (pred_image - x_bad).pow(2).mean(dim=[1, 2, 3])

    with torch.autocast(device_type="cuda", dtype=weight_dtype):
        pred_latents = raw_model.vae.encode(x_pred).latent_dist.mode() * raw_model.vae.config.scaling_factor
        with torch.no_grad():
            good_latents = raw_model.vae.encode(
                x_good.to(dtype=weight_dtype)
            ).latent_dist.mode() * raw_model.vae.config.scaling_factor
            bad_latents = raw_model.vae.encode(
                x_bad.to(dtype=weight_dtype)
            ).latent_dist.mode() * raw_model.vae.config.scaling_factor

    latent_mse_good = (pred_latents.float() - good_latents.float()).pow(2).mean(dim=[1, 2, 3])
    latent_mse_bad = (pred_latents.float() - bad_latents.float()).pow(2).mean(dim=[1, 2, 3])

    chosen_distance = (
        latent_reward_weight * latent_mse_good
        + image_reward_weight * image_mse_good
    )
    rejected_distance = (
        latent_reward_weight * latent_mse_bad
        + image_reward_weight * image_mse_bad
    )

    reward_good = -chosen_distance
    reward_bad = -rejected_distance
    aux_stats = {
        "image_mse_good": image_mse_good,
        "image_mse_bad": image_mse_bad,
        "latent_mse_good": latent_mse_good,
        "latent_mse_bad": latent_mse_bad,
        "chosen_distance": chosen_distance,
        "rejected_distance": rejected_distance,
    }
    return reward_good, reward_bad, aux_stats

def is_full_scratch_checkpoint(path):
    """
    Heuristically identify a full_scratch_v1 (full-model) checkpoint.
    Accept .pt/.pth files; return False if the object is not a dictionary or lacks recognized keys.
    """
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(ckpt, dict):
        return False
    # Use the format field written by save_full_checkpoint_for_scratch_training when available.
    # Check both the format marker and common key structures for compatibility.
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        fmt = ckpt["model"].get("format", None)
        if fmt == "full_scratch_v1":
            return True
        # Fallback: a state_dict entry also indicates a full checkpoint.
        if "state_dict" in ckpt["model"]:
            return True
    # Some implementations store state_dict at the top level.
    if "state_dict" in ckpt:
        return True
    return False


def build_pix2pix(args):
    """
    Construct Pix2Pix_Turbo according to the train_from_scratch initialization mode.
    Only construct the model here; the main training function handles checkpoint loading.
    """
    resume_from_scratch = False
    
    # 1. Check resume_from when resuming an interrupted run.
    if args.resume_from is not None and is_full_scratch_checkpoint(args.resume_from):
        resume_from_scratch = True
        print("Detected scratch checkpoint in resume_from.")

    # 2. Check ref_model_path when initializing DPO.
    # A scratch-format reference checkpoint also requires the matching model structure.
    if not resume_from_scratch and hasattr(args, "ref_model_path") and args.ref_model_path:
        if is_full_scratch_checkpoint(args.ref_model_path):
            resume_from_scratch = True
            print(f"Detected scratch checkpoint in ref_model_path: {args.ref_model_path}")

    if args.train_from_scratch and args.resume_from is None:
        print("Training from scratch...")
        net_pix2pix = Pix2Pix_Turbo(
            lora_rank_unet=args.lora_rank_unet,
            lora_rank_vae=args.lora_rank_vae,
            train_from_scratch=True
        )
        net_pix2pix.set_train()
        
    elif args.train_from_scratch and args.resume_from is not None:
        print("Warning: train_from_scratch=True with resume_from detected. "
              "Will construct from scratch backbone, then follow resume logic.")
        net_pix2pix = Pix2Pix_Turbo(
            lora_rank_unet=args.lora_rank_unet,
            lora_rank_vae=args.lora_rank_vae,
            train_from_scratch=True
        )
        net_pix2pix.set_train()
    
    # Use scratch-compatible initialization for either a resume or reference checkpoint.
    elif resume_from_scratch: 
        # Fine-tune or run DPO from a scratch-training checkpoint.
        print("Fine-tuning/DPO from scratch checkpoint (Initializing Encoder LoRA & Skip Convs)...")
        net_pix2pix = Pix2Pix_Turbo(
            lora_rank_unet=args.lora_rank_unet,
            lora_rank_vae=args.lora_rank_vae,
            train_from_scratch=False,
            resume_from_scratch_ckpt=True  # Initialize the checkpoint-compatible structure, as in inference.
        )
        net_pix2pix.set_train()
        
    else:
        # When not training from scratch, use stabilityai/sd-turbo as the default backbone.
        if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
            net_pix2pix = Pix2Pix_Turbo(
                lora_rank_unet=args.lora_rank_unet,
                lora_rank_vae=args.lora_rank_vae
            )
            net_pix2pix.set_train()
        else:
            net_pix2pix = Pix2Pix_Turbo(
                lora_rank_unet=args.lora_rank_unet,
                lora_rank_vae=args.lora_rank_vae,
                pretrained_name=getattr(args, "pretrained_name", None),
                pretrained_path=getattr(args, "pretrained_path", None),
            )
            net_pix2pix.set_train()
            
    return net_pix2pix


def collect_trainable_params(net_pix2pix, debug=False):
    """
    Collect trainable lightweight-layer parameters, deduplicated by object ID.
    - UNet/VAE LoRA parameters
    - unet.conv_in (all parameters)
    - vae.decoder.skip_conv_1..4 (all parameters)
    - vae.decoder.conv_in (all parameters)
    - vae.decoder.conv_out (all parameters)
    
    Args:
        net_pix2pix: Model instance.
        debug: Whether to print diagnostic information.
    """
    seen = set()
    uniq = []

    # LoRA of UNet & UNet conv_in
    for n, p in net_pix2pix.unet.named_parameters():
        if p.requires_grad:
            if id(p) not in seen:
                uniq.append(p)
                seen.add(id(p))
                
    for p in net_pix2pix.unet.conv_in.parameters():
        if p.requires_grad and id(p) not in seen:
            uniq.append(p)
            seen.add(id(p))

    # Collect VAE LoRA parameters.
    if net_pix2pix.train_encoder_lora:
        if debug:
            print("\n===== DEBUG: Collecting VAE params (train_encoder_lora=True) =====")
        
        encoder_count = 0
        decoder_count = 0
        other_count = 0
        
        # Train LoRA throughout the VAE (encoder and decoder).
        for n, p in net_pix2pix.vae.named_parameters():
            if p.requires_grad:
                # Determine which component owns the parameter.
                is_encoder = False
                is_decoder = False
                
                # Identify the component by parameter name.
                if n.startswith('encoder.'):
                    is_encoder = True
                elif n.startswith('decoder.'):
                    is_decoder = True
                
                if id(p) not in seen:
                    uniq.append(p)
                    seen.add(id(p))
                    
                    if debug:
                        if is_encoder:
                            encoder_count += 1
                            print(f"[ENCODER] {n}")
                        elif is_decoder:
                            decoder_count += 1
                        else:
                            other_count += 1
                            print(f"[OTHER] {n}")
        
        if debug:
            print(f"\n===== Summary =====")
            print(f"Encoder params added to optimizer: {encoder_count}")
            print(f"Decoder params added to optimizer: {decoder_count}")
            print(f"Other VAE params added: {other_count}")
            print(f"Total VAE params added: {encoder_count + decoder_count + other_count}")
    else:
        # Train decoder LoRA only.
        for n, p in net_pix2pix.vae.decoder.named_parameters():
            if ("lora" in n and "vae_skip" in n) and p.requires_grad:
                if id(p) not in seen:
                    uniq.append(p)
                    seen.add(id(p))
    
    # VAE decoder skip convs
    for m in [
        net_pix2pix.vae.decoder.skip_conv_1,
        net_pix2pix.vae.decoder.skip_conv_2,
        net_pix2pix.vae.decoder.skip_conv_3,
        net_pix2pix.vae.decoder.skip_conv_4,
    ]:
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                uniq.append(p)
                seen.add(id(p))

    # Collect all parameters of the VAE decoder's top-level conv_in and conv_out layers.
    if not net_pix2pix.train_encoder_lora:
        # VAE decoder conv_in
        if hasattr(net_pix2pix.vae.decoder, "conv_in"):
            for p in net_pix2pix.vae.decoder.conv_in.parameters():
                if p.requires_grad and id(p) not in seen:
                    uniq.append(p)
                    seen.add(id(p))
        
        # VAE decoder conv_out
        if hasattr(net_pix2pix.vae.decoder, "conv_out"):
            for p in net_pix2pix.vae.decoder.conv_out.parameters():
                if p.requires_grad and id(p) not in seen:
                    uniq.append(p)
                    seen.add(id(p))
    
    if debug:
        print(f"\n===== Total trainable params collected: {len(uniq)} =====\n")
    
    return uniq


def log_trainable_by_optimizer(model, optimizer, log_path="lora_train_status.log", debug=False):
    # 1. Build a lookup of parameter IDs registered with the optimizer.
    opt_param_ids = set([id(p) for group in optimizer.param_groups for p in group["params"]])
    
    if debug:
        print("\n===== DEBUG: Checking encoder in optimizer =====")
        encoder_param_count = 0
        encoder_in_opt_count = 0
        
        for n, p in model.vae.encoder.named_parameters():
            encoder_param_count += 1
            if id(p) in opt_param_ids:
                encoder_in_opt_count += 1
                print(f"[FOUND] Encoder param in optimizer: {n}")
        
        print(f"Total encoder params: {encoder_param_count}")
        print(f"Encoder params in optimizer: {encoder_in_opt_count}")
    
    # 2. Collect trainable leaf-module names, grouped by UNet/VAE.
    unet_submodules = {}
    vae_submodules = {}

    def _get_module_name(n):
        """Extract the leaf module name, such as 'conv_in' or 'to_k'."""
        if "lora" in n:
            parts = n.split('.')
            return parts[-4] if len(parts) >= 4 else 'LoRA_Unknown'
        else:
            module_name = n.split('.')[-2] 
            return module_name if module_name else n.split('.')[-3]

    # A. Iterate over all UNet parameters.
    for n, p in model.unet.named_parameters():
        if id(p) in opt_param_ids:
            module_name = _get_module_name(n)
            unet_submodules[module_name] = unet_submodules.get(module_name, 0) + 1
            
    # B. Iterate over all VAE parameters.
    for n, p in model.vae.named_parameters():
        if id(p) in opt_param_ids:
            module_name = _get_module_name(n)
            vae_submodules[module_name] = vae_submodules.get(module_name, 0) + 1

    # 3. Summarize training status at the component level.
    # Check directly whether encoder parameters belong to the optimizer.
    encoder_train = any(id(p) in opt_param_ids for n, p in model.vae.encoder.named_parameters())
    decoder_train = len(vae_submodules) > 0  # Track whether any VAE submodules are trainable.
    unet_train = len(unet_submodules) > 0

    # 4. Build the log contents.
    lines = []
    lines.append("====== Trainable Modules (Checked by Optimizer ID) ======\n")
    lines.append(f"Encoder Trainable = {encoder_train}\n")
    lines.append(f"Decoder Trainable = {decoder_train}\n")
    lines.append(f"UNet Trainable    = {unet_train}\n\n")

    # Detailed UNet module list.
    lines.append("--- UNet Trainable Sub-Modules (Total: {}) ---\n".format(sum(unet_submodules.values())))
    lines.append("Format: Sub-Module Name (Trained Parameter Count)\n")
    for module_name, count in sorted(unet_submodules.items()):
        lines.append(f"- UNet: {module_name} (Total Params: {count})\n")

    # Detailed VAE module list.
    lines.append("\n--- VAE Trainable Sub-Modules (Total: {}) ---\n".format(sum(vae_submodules.values())))
    lines.append("Format: Sub-Module Name (Trained Parameter Count)\n")
    for module_name, count in sorted(vae_submodules.items()):
        # Explicitly label skip convolutions as VAE decoder modules.
        tag = "VAE Decoder" if "skip_conv" in module_name or not encoder_train else "VAE Encoder/Decoder"
        lines.append(f"- {tag}: {module_name} (Total Params: {count})\n")

    # 5. Write the log file.
    with open(log_path, "w", encoding="utf-8")  as f:
        f.writelines(lines)

    print(f"Logged trainability and categorized sub-modules to {log_path}")
    print(f"  Encoder Trainable: {encoder_train}")
    print(f"  Decoder Trainable: {decoder_train}")
    print(f"  UNet Trainable: {unet_train}")


def main(args):
    # Disable WandB and set Hugging Face dataset/transformer offline flags.
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Use TensorBoard for tracking.
    args.report_to = "tensorboard" 
    use_dpo = False
    if hasattr(args, "beta_dpo") and args.beta_dpo > 0.0:
        use_dpo = True
    if hasattr(args, "train_method") and args.train_method == "dpo":
        use_dpo = True
    
    logging_dir = os.path.join(args.output_dir,"logs")

    accelerator_project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config
    )
        
    
    if accelerator.is_main_process:
        # Keep only scalar configuration values supported by TensorBoard.
        tracker_config = {}
        for k, v in vars(args).items():
            if isinstance(v, (int, float, str, bool)):
                tracker_config[k] = v
        
        # Initialize the TensorBoard tracker.
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        if use_dpo:
            print(f"🚀 MODE: DPO (Direct Preference Optimization) | beta={args.beta_dpo}")
        else:
            print(f"🚀 MODE: GAN / SFT Training")

    # 1. Construct the generator without loading a checkpoint here.
    net_pix2pix = build_pix2pix(args)

    # 2. Construct the frozen reference model for DPO.
    ref_pix2pix = None

    if use_dpo:
        ref_pix2pix = build_pix2pix(args)
        ref_pix2pix.set_eval()
        ref_pix2pix.requires_grad_(False)
        if accelerator.is_main_process:
            print("❄️ Reference Model Initialized and Frozen")
            
    # 3) xFormers / gradient ckpt / TF32
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
            if ref_pix2pix: ref_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 3. Construct the discriminator and perceptual networks.
    # if args.gan_disc_type == "vagan_clip":
    #     import vision_aided_loss
    #     net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    # else:
    #     raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")
    # net_disc = net_disc.cuda()
    # net_disc.requires_grad_(True)
    # net_disc.cv_ensemble.requires_grad_(False)
    # net_disc.train()
    net_disc = None
    if not use_dpo:
        if args.gan_disc_type == "vagan_clip":
            import vision_aided_loss
            net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        else:
            raise NotImplementedError
        net_disc = net_disc.cuda()
        net_disc.requires_grad_(True)
        net_disc.cv_ensemble.requires_grad_(False)
        net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()
    net_lpips.requires_grad_(False)

    # 4. Detect full_scratch_v1 checkpoints for backbone loading before prepare.
    pending_full_resume = None
    pending_light_resume = None

    # A. Prefer an explicit resume_from checkpoint for interrupted runs.
    if args.resume_from is not None:
        if is_full_scratch_checkpoint(args.resume_from):
            accelerator.print(f"Detected full_scratch_v1 checkpoint: {args.resume_from}")
            pending_full_resume = args.resume_from
        else:
            accelerator.print(f"Detected light checkpoint: {args.resume_from}")
            pending_light_resume = args.resume_from

    # B. Load the DPO reference model.
    ref_load_path = None
    if use_dpo and ref_pix2pix is not None:
        # 1. Prefer the explicitly supplied reference checkpoint.
        if args.ref_model_path:
            ref_load_path = args.ref_model_path
        # 2. Fall back to the full resume checkpoint if no reference path was supplied.
        elif pending_full_resume:
             ref_load_path = pending_full_resume
             accelerator.print(
                 "⚠️ WARNING: No --ref_model_path provided. "
                 "Falling back to --resume_from as reference model."
             )
        if ref_load_path is not None:
            accelerator.print(f"🔄 Loading Reference Model from: {ref_load_path}")
            ref_pix2pix.load_full_checkpoint(ref_load_path)
        else:
             accelerator.print("⚠️ WARNING: Reference Model is using random/base weights! This is likely WRONG for DPO.")
             
    # C. Initialize the DPO policy model.
    # For a new DPO run, initialize the policy from the SFT reference checkpoint.
    if use_dpo and args.ref_model_path and pending_full_resume is None and pending_light_resume is None:
        accelerator.print(f"🚀 DPO Start: Initializing Policy Model from {args.ref_model_path}")
        net_pix2pix.load_full_checkpoint(args.ref_model_path)

    # -------------------------------------------------------------------------

    # 5. Collect trainable parameters and construct optimizers and schedulers.
    # Enable diagnostics on the first call only.
    layers_to_opt = collect_trainable_params(net_pix2pix, debug=True)

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    num_total_gen_steps = args.max_train_steps if use_dpo else args.max_train_steps * 2
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_total_gen_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer_disc = None
    lr_scheduler_disc = None
    if not use_dpo:
        optimizer_disc = torch.optim.AdamW(
            net_disc.parameters(), lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
        )
        lr_scheduler_disc = get_scheduler(
            args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps, num_training_steps=num_total_gen_steps,
            num_cycles=args.lr_num_cycles, power=args.lr_power,
        )
    
    # 7. Load the training dataset for the selected method.
    if use_dpo:
        if DPODataset is None:
            raise ImportError("DPODataset not found! Check my_utils imports.")
        dataset_train = DPODataset(
            dataset_folder=args.dataset_folder,
            split="train", # DPO reads triplets directly from the dataset folders.
            image_prep=args.train_image_prep,
            tokenizer=net_pix2pix.tokenizer,
        )
        accelerator.print(f"Loaded DPODataset with {len(dataset_train)} triplets.")
    else:        
        dataset_train = PairedDataset(
            dataset_folder=args.dataset_folder,
            image_prep=args.train_image_prep,
            split="train",
            tokenizer=net_pix2pix.tokenizer,
        )
    dl_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    dataset_val = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.test_image_prep,
        split="test",
        tokenizer=net_pix2pix.tokenizer,
    )
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # 8. Load the full backbone before distributed preparation.
    if pending_full_resume is not None:
        accelerator.print("Loading full checkpoint into backbone before prepare (fine-tune mode).")
        _ = net_pix2pix.load_full_checkpoint(
            pending_full_resume,
            optimizer=None,
            lr_scheduler=None,
            train_loader=None,
        )

    prepare_args = [net_pix2pix, optimizer, dl_train, lr_scheduler]
    if not use_dpo:
        prepare_args.extend([net_disc, optimizer_disc, lr_scheduler_disc])
    
    prepared = accelerator.prepare(*prepare_args)
    
    if not use_dpo:
        net_pix2pix, optimizer, dl_train, lr_scheduler, net_disc, optimizer_disc, lr_scheduler_disc = prepared
    else:
        net_pix2pix, optimizer, dl_train, lr_scheduler = prepared
    net_pix2pix.vae.enable_tiling()

    # Prepare the reference model separately.
    if use_dpo:
        # 1. Enable VAE tiling.
        ref_pix2pix.vae.enable_tiling()
        # 2. Place the reference model on CPU initially to reduce GPU memory use.
        ref_pix2pix.to("cpu") 
        # The reference model is inference-only, so it does not use accelerator.prepare_model.
        # Manage its device placement explicitly.
    

    # 9. Resume lightweight checkpoints after unwrapping the model.
    global_step = 0
    if pending_full_resume is not None:
        accelerator.print(f"Resuming optimizer/scheduler state from full checkpoint: {pending_full_resume}")
        orig_model = accelerator.unwrap_model(net_pix2pix)
        start_step = orig_model.load_full_checkpoint(
            pending_full_resume,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=dl_train,
        )
        global_step = start_step
    if pending_light_resume is not None:
        accelerator.print(f"Resuming from light checkpoint: {pending_light_resume}")
        orig_model = accelerator.unwrap_model(net_pix2pix)
        start_step = orig_model.load_light_checkpoint(
            pending_light_resume,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=dl_train,
            map_location="cpu",
        )
        global_step = start_step

    if pending_full_resume is not None and args.train_from_scratch:
        accelerator.print("Full-scratch resume with train_from_scratch=True: will continue with provided optimizer/scheduler states if applicable.")

    # 10. Print optimizer and scheduler state.
    if optimizer is not None:
        for i, param_group in enumerate(optimizer.param_groups):
            accelerator.print(f"[resume] optimizer group {i} lr={param_group['lr']}")
        accelerator.print(f"[resume] optimizer state keys: {list(optimizer.state.keys())[:5]} ... (total={len(optimizer.state)})")
    if lr_scheduler is not None:
        accelerator.print(f"[resume] lr_scheduler last_lr={lr_scheduler.get_last_lr()}")

    # 11. Prepare the remaining networks.
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    t_clip_renorm = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 12. Set device and precision.
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    if use_dpo:
        net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    else:
        net_disc.to(accelerator.device, dtype=weight_dtype)
    
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    # 14. Set up the progress bar and discriminator attention.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    if not use_dpo:
        for name, module in net_disc.named_modules():
            if "attn" in name:
                module.fused_attn = False

    # 15. Compute reference FID features, if enabled.
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(
            os.path.join(args.dataset_folder, "test_B"),
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device("cuda"),
            mode="clean",
            custom_image_tranform=fn_transform,
            description="",
            verbose=True,
        )

    # 16. Run the training loop.
    if args.resume_from is None:
        global_step = 0
    dpo_history = [] 
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            # DPO training branch.
            if use_dpo:
                # -------------------------------------------------------------
                # Step 1: evaluate the reference model without gradients, then release its GPU memory.
                # -------------------------------------------------------------
                # Move the reference model to the GPU.
                ref_pix2pix.to(accelerator.device)
                
                with torch.no_grad():
                    # Compute reference reward scores.
                    # get_batch_log_prob already handles autocast internally.
                    ref_log_good, ref_log_bad, _ = get_batch_log_prob(
                        ref_pix2pix,
                        batch,
                        weight_dtype,
                        latent_reward_weight=args.dpo_latent_reward_weight,
                        image_reward_weight=args.dpo_image_reward_weight,
                    )
                    
                    # Detach reference scores from the computation graph.
                    ref_log_good = ref_log_good.detach()
                    ref_log_bad = ref_log_bad.detach()
                    
                    ref_rewards_chosen = ref_log_good.mean()
                    ref_rewards_rejected = ref_log_bad.mean()
                
                # Move the reference model back to CPU and clear the GPU cache after evaluation.
                # Offload reference parameters to reduce GPU memory use.
                ref_pix2pix.to("cpu") 
                torch.cuda.empty_cache()

                # -------------------------------------------------------------
                # Step 2: evaluate the policy model with gradient tracking.
                # -------------------------------------------------------------
                with accelerator.accumulate(net_pix2pix):
                    # Compute policy reward scores.
                    policy_log_good, policy_log_bad, dpo_aux = get_batch_log_prob(
                        net_pix2pix,
                        batch,
                        weight_dtype,
                        latent_reward_weight=args.dpo_latent_reward_weight,
                        image_reward_weight=args.dpo_image_reward_weight,
                    )
                    
                    policy_rewards_chosen = policy_log_good.mean()
                    policy_rewards_rejected = policy_log_bad.mean()
                    
                    # Compute the DPO loss.
                    policy_diff = policy_log_good - policy_log_bad
                    ref_diff = ref_log_good - ref_log_bad # Reference scores are detached from the computation graph.
                    
                    # Ensure the reference margin is on the policy's device.
                    ref_diff = ref_diff.to(policy_diff.device)

                    logits = args.beta_dpo * (policy_diff - ref_diff)
                    loss_dpo = -F.logsigmoid(logits).mean()
                    loss_anchor = dpo_aux["chosen_distance"].mean() * args.dpo_anchor_weight
                    loss = loss_dpo + loss_anchor
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                         params_to_clip = collect_trainable_params(accelerator.unwrap_model(net_pix2pix), debug=False)
                         accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    
                    # Logging
                    lossG = loss
                    dpo_margin = policy_rewards_chosen - policy_rewards_rejected
                    loss_l2 = dpo_aux["image_mse_good"].mean()
                    loss_lpips_val = dpo_aux["latent_mse_good"].mean()
                    
                    # Initialize the remaining logging variables.
                    lossD = torch.tensor(0.0) 
                    dpo_rewards_chosen = policy_rewards_chosen
                    dpo_rewards_rejected = policy_rewards_rejected
                    dpo_ref_rewards_chosen = ref_rewards_chosen
                    dpo_ref_rewards_rejected = ref_rewards_rejected
            else:
                l_acc = [net_pix2pix, net_disc]
                with accelerator.accumulate(*l_acc):
                    x_src = batch["conditioning_pixel_values"]
                    x_tgt = batch["output_pixel_values"]
                    B, C, H, W = x_src.shape

                    # forward pass
                    if args.train_from_scratch:
                        noise_map = torch.randn(B, 4, H//8, W//8, device=x_src.device, dtype=x_src.dtype)
                        x_tgt_pred = net_pix2pix(
                            x_src, 
                            prompt_tokens=batch["input_ids"], 
                            deterministic=False,
                            r=0.3,
                            noise_map=noise_map
                        )
                    else:
                        x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)

                    # Reconstruction losses
                    loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                    loss_F1 = F.l1_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l1
                    loss_lpips_val = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips

                    # Compute structural loss if enabled.
                    if args.lambda_structural > 0:
                        loss_structural = structural_loss(x_tgt_pred, x_tgt) * args.lambda_structural
                        loss = loss_l2 + loss_lpips_val + loss_structural
                    else:
                        loss = loss_l2 + loss_lpips_val

                    accelerator.backward(loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        # Disable verbose diagnostics during training.
                        accelerator.clip_grad_norm_(collect_trainable_params(accelerator.unwrap_model(net_pix2pix), debug=False), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    # Generator GAN loss
                    if args.train_from_scratch:
                        noise_map = torch.randn(B, 4, H//8, W//8, device=x_src.device, dtype=x_src.dtype)
                        x_tgt_pred = net_pix2pix(
                            x_src, 
                            prompt_tokens=batch["input_ids"], 
                            deterministic=False,
                            r=0.8,
                            noise_map=noise_map
                        )
                    else:
                        x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
            
                    lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                    accelerator.backward(lossG)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(collect_trainable_params(accelerator.unwrap_model(net_pix2pix), debug=False), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    # Discriminator loss
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                    accelerator.backward(lossD_real.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                    lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                    accelerator.backward(lossD_fake.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                    lossD = lossD_real + lossD_fake

            # Step/Log/Checkpoint/Eval
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips_val.detach().item()
                    
                    if use_dpo:
                        logs["loss_dpo"] = loss_dpo.detach().item()
                        logs["loss_anchor"] = loss_anchor.detach().item()
                        logs["loss_image_mse_chosen"] = loss_l2.detach().item()
                        logs["loss_latent_mse_chosen"] = loss_lpips_val.detach().item()
                        logs["dpo_rewards/policy_chosen"]   = dpo_rewards_chosen.detach().item()
                        logs["dpo_rewards/policy_rejected"] = dpo_rewards_rejected.detach().item()
                        logs["dpo_rewards/ref_chosen"] = dpo_ref_rewards_chosen.detach().item()
                        logs["dpo_rewards/ref_rejected"] = dpo_ref_rewards_rejected.detach().item()
                        logs["dpo_rewards/margin"] = dpo_margin.detach().item()
                        
                        # Record scalar metrics directly instead of using WandB tables or plots.
                        # TensorBoard handles scalar visualization without a separate table.
                        
                        progress_bar.set_postfix(
                            loss=lossG.detach().item(),
                            dpo=loss_dpo.detach().item(),
                            anchor=loss_anchor.detach().item(),
                            margin=dpo_margin.detach().item(),
                        )
                    else:
                        progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
                        path = accelerator.unwrap_model(net_pix2pix).save_full_checkpoint_for_scratch_training(
                            optimizer, lr_scheduler, global_step, ckpt_dir, train_loader=dl_train, keep_last=3
                        )
                        print(f"[checkpoint] saved -> {path}")

                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        for step_val, batch_val in enumerate(dl_val):
                            if step_val >= args.num_samples_eval:
                                break
                            x_src_v = batch_val["conditioning_pixel_values"].to(accelerator.device)
                            x_tgt_v = batch_val["output_pixel_values"].to(accelerator.device)
                            with torch.no_grad():
                                x_tgt_pred_v = accelerator.unwrap_model(net_pix2pix)(
                                    x_src_v, prompt_tokens=batch_val["input_ids"].to(accelerator.device), deterministic=True
                                )
                                loss_l2_v = F.mse_loss(x_tgt_pred_v.float(), x_tgt_v.float(), reduction="mean")
                                loss_lpips_v = net_lpips(x_tgt_pred_v.float(), x_tgt_v.float()).mean()
                                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred_v * 0.5 + 0.5)
                                x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                                caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred_v.device)
                                clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim = clipsim.mean()

                                l_l2.append(loss_l2_v.item())
                                l_lpips.append(loss_lpips_v.item())
                                l_clipsim.append(clipsim.item())

                            os.makedirs(os.path.join(args.output_dir, "eval", f"step_{global_step}"), exist_ok=True)
                            batch_size = x_tgt_pred_v.size(0)
                            for idx in range(batch_size):
                                output_pil = transforms.ToPILImage()(x_tgt_pred_v[idx].cpu() * 0.5 + 0.5)
                                original_name = batch_val["img_name"][idx]
                                outf = os.path.join(args.output_dir, "eval", f"step_{global_step}", original_name)
                                output_pil.save(outf)
                        
                        # Save images locally and record validation metrics in TensorBoard.
                        logs["val/l2"] = np.mean(l_l2) if len(l_l2) > 0 else 0.0
                        logs["val/lpips"] = np.mean(l_lpips) if len(l_lpips) > 0 else 0.0
                        logs["val/clipsim"] = np.mean(l_clipsim) if len(l_clipsim) > 0 else 0.0

                    # Record metrics in TensorBoard.
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    #args = parse_args_paired_training()
    args = parse_args_dpo()
    # Normalize the train_from_scratch argument name for backward compatibility.
    if hasattr(args, "train_from_sctatch") and not hasattr(args, "train_from_scratch"):
        # Map the legacy misspelled argument name automatically.
        args.train_from_scratch = args.train_from_sctatch
    main(args)
