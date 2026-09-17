import json
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
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

# import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pretrain_pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, PairedDataset, structural_loss


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
    if args.resume_from is not None and is_full_scratch_checkpoint(args.resume_from):
        resume_from_scratch = True
        print("load encoder's lora")
    
    if args.train_from_scratch and args.resume_from is None:
        print("Training from scratch...")
        net_pix2pix = Pix2Pix_Turbo(
            lora_rank_unet=args.lora_rank_unet,
            lora_rank_vae=args.lora_rank_vae,
            train_from_scratch=True,
            enable_skip_conv=False
        )
        net_pix2pix.set_train()
        
    elif args.train_from_scratch and args.resume_from is not None:
        # Construct from scratch and warn; checkpoint loading happens later.
        print("Warning: train_from_scratch=True with resume_from detected. "
              "Will construct from scratch backbone, then follow resume logic.")
        net_pix2pix = Pix2Pix_Turbo(
            lora_rank_unet=args.lora_rank_unet,
            lora_rank_vae=args.lora_rank_vae,
            train_from_scratch=True,
            enable_skip_conv=False
        )
        net_pix2pix.set_train()
    
    elif not args.train_from_scratch and resume_from_scratch:
        # Fine-tune from a scratch-training checkpoint.
        print("Fine-tuning from scratch checkpoint (Encoder will be frozen)...")
        net_pix2pix = Pix2Pix_Turbo(
            lora_rank_unet=args.lora_rank_unet,
            lora_rank_vae=args.lora_rank_vae,
            train_from_scratch=False,
            resume_from_scratch_ckpt=True,  # Enable initialization compatible with scratch-training checkpoints.
            enable_skip_conv=True
        )
        net_pix2pix.set_train()
        
    else:
        # When not training from scratch, use stabilityai/sd-turbo as the default backbone.
        if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
            net_pix2pix = Pix2Pix_Turbo(
                lora_rank_unet=args.lora_rank_unet,
                lora_rank_vae=args.lora_rank_vae,
                enable_skip_conv=True
            )
            net_pix2pix.set_train()
        else:
            # Extend this branch to support other backbone paths or names.
            net_pix2pix = Pix2Pix_Turbo(
                lora_rank_unet=args.lora_rank_unet,
                lora_rank_vae=args.lora_rank_vae,
                pretrained_name=getattr(args, "pretrained_name", None),
                pretrained_path=getattr(args, "pretrained_path", None),
                enable_skip_conv=True
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
    # for m in [
    #     net_pix2pix.vae.decoder.skip_conv_1,
    #     net_pix2pix.vae.decoder.skip_conv_2,
    #     net_pix2pix.vae.decoder.skip_conv_3,
    #     net_pix2pix.vae.decoder.skip_conv_4,
    # ]:
    #     for p in m.parameters():
    #         if p.requires_grad and id(p) not in seen:
    #             uniq.append(p)
    #             seen.add(id(p))
    should_train_skip = getattr(net_pix2pix, "enable_skip_conv", False)
    skip_convs = []
    if should_train_skip:  # Collect skip-connection parameters only when enabled.
        if hasattr(net_pix2pix.vae.decoder, "skip_conv_1"): skip_convs.append(net_pix2pix.vae.decoder.skip_conv_1)
        if hasattr(net_pix2pix.vae.decoder, "skip_conv_2"): skip_convs.append(net_pix2pix.vae.decoder.skip_conv_2)
        if hasattr(net_pix2pix.vae.decoder, "skip_conv_3"): skip_convs.append(net_pix2pix.vae.decoder.skip_conv_3)
        if hasattr(net_pix2pix.vae.decoder, "skip_conv_4"): skip_convs.append(net_pix2pix.vae.decoder.skip_conv_4)
        
        if debug:
            print(f"[INFO] enable_skip_conv is True, collecting skip connections.")
    else:
        if debug:
            print(f"[INFO] enable_skip_conv is False, IGNORING skip connections.")

    for m in skip_convs:
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
    """
    Group trainable leaf-module names by UNet/VAE using the optimizer's actual parameters.
    
    Args:
        model: Model instance.
        optimizer: Optimizer instance.
        log_path: Destination log file.
        debug: Whether to print detailed diagnostics.
    """
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


def build_train_val_subsets(args, tokenizer):
    """
    Build a deterministic 9:1 train/val split from the original training set.

    This keeps the official test split untouched, which is the safer protocol for
    model selection and closer to common top-conference practice.
    """
    dataset_train_full = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_image_prep,
        split="train",
        tokenizer=tokenizer,
    )
    dataset_val_full = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.test_image_prep,
        split="train",
        tokenizer=tokenizer,
    )

    num_samples = len(dataset_train_full)
    if num_samples < 2:
        raise ValueError("Need at least 2 training samples to create a 9:1 train/val split.")

    split_seed = args.seed if args.seed is not None else 42
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(num_samples, generator=generator).tolist()

    val_size = max(1, int(round(num_samples * 0.1)))
    val_size = min(val_size, num_samples - 1)
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    train_subset = torch.utils.data.Subset(dataset_train_full, train_indices)
    val_subset = torch.utils.data.Subset(dataset_val_full, val_indices)

    split_manifest = {
        "split_seed": split_seed,
        "train_ratio": 0.9,
        "val_ratio": 0.1,
        "num_total": num_samples,
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "train_files": [dataset_train_full.img_names[i] for i in train_indices],
        "val_files": [dataset_train_full.img_names[i] for i in val_indices],
    }

    return train_subset, val_subset, split_manifest


def main(args):
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.report_to = None # Disable experiment tracking.
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[ddp_kwargs]
    )

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

    # 1. Construct the generator without loading a checkpoint here.
    net_pix2pix = build_pix2pix(args)

    # 2) xFormers / gradient ckpt / TF32
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 3. Construct the discriminator and perceptual networks.
    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")
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
    if args.resume_from is not None:
        if is_full_scratch_checkpoint(args.resume_from):
            accelerator.print(f"Detected full_scratch_v1 checkpoint: {args.resume_from}")
            pending_full_resume = args.resume_from
        else:
            accelerator.print(f"Detected light checkpoint: {args.resume_from}")
            pending_light_resume = args.resume_from

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
    
    num_total_gen_steps = args.max_train_steps * 2
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_total_gen_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_total_gen_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 6. Prepare datasets.
    dataset_train, dataset_val, split_manifest = build_train_val_subsets(args, net_pix2pix.tokenizer)

    if accelerator.is_main_process:
        split_path = os.path.join(args.output_dir, "train_val_split.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2, ensure_ascii=False)
        accelerator.print(
            f"Deterministic train/val split created: "
            f"train={split_manifest['num_train']}, val={split_manifest['num_val']}, "
            f"seed={split_manifest['split_seed']}"
        )
        accelerator.print(f"Saved split manifest to {split_path}")

    dl_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # 7. Load the full backbone before prepare, without optimizer or scheduler state.
    if pending_full_resume is not None:
        accelerator.print("Loading full checkpoint into backbone before prepare (fine-tune mode).")
        _ = net_pix2pix.load_full_checkpoint_for_scratch_training(
            pending_full_resume,
            optimizer=None,
            lr_scheduler=None,
            train_loader=None,
        )
        if accelerator.is_main_process:
            print("\n===== Quick Encoder LoRA Check =====")
            count = 0
            
            for name, param in net_pix2pix.vae.encoder.named_parameters():
                if "lora" in name:
                    count += 1
                    if count <= 3:  # Print only the first three entries.
                        print(f"{name}:")
                        print(f"  shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
                        print(f"  requires_grad={param.requires_grad}")
            
            if count == 0:
                print("❌ No Encoder LoRA found! Structure not created!")
            else:
                print(f"✅ Total Encoder LoRA params: {count}")
            # Check that skip-connection layers exist before accessing them.
            if hasattr(net_pix2pix.vae.decoder, "skip_conv_1"):
                print("!!! Force Resetting Skip Connections to avoid Identity Leakage !!!")
                # Reset skip convolutions to near-zero values, matching the original experiment.
                torch.nn.init.constant_(net_pix2pix.vae.decoder.skip_conv_1.weight, 1e-5)
                torch.nn.init.constant_(net_pix2pix.vae.decoder.skip_conv_2.weight, 1e-5)
                torch.nn.init.constant_(net_pix2pix.vae.decoder.skip_conv_3.weight, 1e-5)
                torch.nn.init.constant_(net_pix2pix.vae.decoder.skip_conv_4.weight, 1e-5)
            else:
                print("!!! Skip Connections not found, skipping Force Reset (Normal for Stage 1 scratch training) !!!")

    # 8. Prepare distributed training.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    
    # Log training status after prepare with diagnostics enabled.
    if accelerator.is_main_process:
        real_model = accelerator.unwrap_model(net_pix2pix)
        log_trainable_by_optimizer(
            real_model,
            optimizer,
            log_path=os.path.join(args.output_dir, "trainable.log"),
            debug=True  # Enable detailed diagnostics.
        )

    # 9. Resume lightweight checkpoints after unwrapping the model.
    global_step = 0
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
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    # 13. Configure experiment trackers.
    # if accelerator.is_main_process:
    #     tracker_config = dict(vars(args))
    #     accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # 14. Set up the progress bar and discriminator attention.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # 15. Compute reference FID features, if enabled.
    # With an internal 9:1 train/val split, we keep the official test split untouched.
    # To avoid test leakage, folder-based FID against test_B is disabled here.
    if args.track_val_fid and accelerator.is_main_process:
        accelerator.print(
            "track_val_fid is disabled in this setup to avoid using test_B during training. "
            "Use the untouched test split only for final reporting."
        )
        args.track_val_fid = False

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

    completed_debug_steps = 0
    if global_step >= args.max_train_steps:
        accelerator.end_training()
        return
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
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
                        r=0.8,
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
                    loss = loss_l2 + loss_lpips_val
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
                    progress_bar.set_postfix(**logs)

                    # checkpoint
                    if not args.debug_steps and global_step % args.checkpointing_steps == 1:
                        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
                        path = accelerator.unwrap_model(net_pix2pix).save_full_checkpoint_for_scratch_training(
                            optimizer, lr_scheduler, global_step, ckpt_dir, train_loader=dl_train, keep_last=3
                        )

                        # Also export a plain model_state_dict .pth file for direct loading with torch.load.
                        try:
                            pth_path = os.path.join(ckpt_dir, f"full_model_{global_step}.pth")
                            model_state = accelerator.unwrap_model(net_pix2pix).state_dict()
                            torch.save({"model_state_dict": model_state}, pth_path)
                            if accelerator.is_main_process:
                                print(f"[checkpoint] saved pth -> {pth_path}")
                        except Exception as e:
                            if accelerator.is_main_process:
                                print(f"[checkpoint] failed to save .pth: {e}")
                        if accelerator.is_main_process:
                            print(f"[checkpoint] saved -> {path}")

                    # eval
                    if not args.debug_steps and global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        # if args.track_val_fid:
                        #     os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step_val, batch_val in enumerate(dl_val):
                            if step_val >= args.num_samples_eval:
                                break
                            x_src_v = batch_val["conditioning_pixel_values"].to(accelerator.device)
                            x_tgt_v = batch_val["output_pixel_values"].to(accelerator.device)
                            Bv, Cv, Hv, Wv = x_src_v.shape
                            assert Bv == 1, "Use batch size 1 for eval."
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

                            # Save visualization images.
                            os.makedirs(os.path.join(args.output_dir, "eval", f"step_{global_step}"), exist_ok=True)
                            batch_size = x_tgt_pred_v.size(0)
                            for idx in range(batch_size):
                                output_pil = transforms.ToPILImage()(x_tgt_pred_v[idx].cpu() * 0.5 + 0.5)
                                original_name = batch_val["img_name"][idx]
                                outf = os.path.join(args.output_dir, "eval", f"step_{global_step}", original_name)
                                output_pil.save(outf)

                        logs["val/l2"] = np.mean(l_l2) if len(l_l2) > 0 else 0.0
                        logs["val/lpips"] = np.mean(l_lpips) if len(l_lpips) > 0 else 0.0
                        logs["val/clipsim"] = np.mean(l_clipsim) if len(l_clipsim) > 0 else 0.0

                    accelerator.log(logs, step=global_step)

                completed_debug_steps += 1
                if global_step >= args.max_train_steps or (
                    args.debug_steps and completed_debug_steps >= args.debug_steps
                ):
                    accelerator.wait_for_everyone()
                    accelerator.print(f"Training stopped successfully at step {global_step}.")
                    accelerator.end_training()
                    return


if __name__ == "__main__":
    args = parse_args_paired_training()
    # Normalize the train_from_scratch argument name for backward compatibility.
    if hasattr(args, "train_from_sctatch") and not hasattr(args, "train_from_scratch"):
        # Map the legacy misspelled argument name automatically.
        args.train_from_scratch = args.train_from_sctatch
    main(args)
