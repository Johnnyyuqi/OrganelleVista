import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd
from torch.utils.checkpoint import checkpoint 



class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4, train_from_scratch=False, resume_from_scratch_ckpt=False,enable_skip_conv=True):
        print("!!! DEBUG: I AM THE NEW CODE !!!") # Print the implementation marker.
        print("Initializing Pix2Pix_Turbo...")
        super().__init__()
        
        self.train_encoder_lora = True
        self.enable_skip_conv = enable_skip_conv # Store the skip-connection setting.
        
        if train_from_scratch:
            print("Initializing model from scratch without any pretrained weights...")
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
            print("Loading text encoder...")
            self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
            print("Creating scheduler...")
            self.sched = make_1step_sched()
            print("Loading VAE...")
            
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
            vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
            vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

            # ========== Skip-convolution training configuration ==========
            self.enable_skip_conv = enable_skip_conv  # Preserve the skip-connection setting.

            if self.enable_skip_conv:
                print("✅ Enabling Skip Connections (Creating Layers)")
                vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
                vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
                vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
                vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
                vae.decoder.ignore_skip = False
                
                # Initialize weights.
                torch.nn.init.normal_(vae.decoder.skip_conv_1.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(vae.decoder.skip_conv_2.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(vae.decoder.skip_conv_3.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(vae.decoder.skip_conv_4.weight, mean=0.0, std=0.02)
            else:
                print("⛔ Disabling Skip Connections (Stage 1 Mode)")
                vae.decoder.ignore_skip = True # Tell the decoder forward pass to bypass skip layers.
            # ========== End skip-convolution training configuration ==========

            # Random initialization used by the scratch-training branch.
            # torch.nn.init.normal_(vae.decoder.skip_conv_1.weight, mean=0.0, std=0.02)
            # torch.nn.init.normal_(vae.decoder.skip_conv_2.weight, mean=0.0, std=0.02)
            # torch.nn.init.normal_(vae.decoder.skip_conv_3.weight, mean=0.0, std=0.02)
            # torch.nn.init.normal_(vae.decoder.skip_conv_4.weight, mean=0.0, std=0.02)


            self.train_encoder_lora = True
            base_names=("conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0")
            target_modules_vae = self._collect_vae_target_modules(vae, include_encoder=True, base_names=base_names)
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)

            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet
        
        else:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
            print("Loading text encoder...")
            self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
            print("Creating scheduler...")
            self.sched = make_1step_sched()
            print("Loading VAE...")

            vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
            vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
            vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
            # add the skip connection convs
            # vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            # vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            # vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            # vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            # vae.decoder.ignore_skip = False
            # ========== Skip-convolution training configuration ==========
            self.enable_skip_conv = enable_skip_conv 

            if self.enable_skip_conv:
                # add the skip connection convs
                vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
                vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
                vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
                vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
                vae.decoder.ignore_skip = False
            else:
                vae.decoder.ignore_skip = True
            # ========== End skip-convolution training configuration ==========
            unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

            if pretrained_path is not None:
                sd = torch.load(pretrained_path, map_location="cpu")
                encoded_base_names =("conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0")
                unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
                
                # Configure VAE LoRA when loading pretrained weights.
                self.train_encoder_lora = True
                target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4","to_k", "to_q", "to_v", "to_out.0",]
                #target_modules_vae = self._collect_vae_target_modules(vae, include_encoder=False, fallback=sd.get("vae_lora_target_modules"),basenames=encoded_base_names)
                #vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=target_modules_vae)
                vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=target_modules_vae)
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                _sd_vae = vae.state_dict()
                for k in sd["state_dict_vae"]:
                    _sd_vae[k] = sd["state_dict_vae"][k]
                vae.load_state_dict(_sd_vae, strict=False)
                unet.add_adapter(unet_lora_config)
                _sd_unet = unet.state_dict()
                for k in sd["state_dict_unet"]:
                    _sd_unet[k] = sd["state_dict_unet"][k]

            else:
                if resume_from_scratch_ckpt:
                    print("Fine-tune mode: initializing LoRA for UNet and full VAE (Encoder will be frozen)")
                    self.train_encoder_lora = True  # Create the LoRA structure required by the checkpoint.
                    
                    #torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
                    #torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
                    #torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
                    #torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
                    
                    # Add encoder and decoder LoRA to match the checkpoint structure.
                    base_names = ("conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                        "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                        "to_k", "to_q", "to_v", "to_out.0")
                    target_modules_vae = self._collect_vae_target_modules(vae, include_encoder=True, base_names=base_names)
                    vae_lora_config = LoraConfig(
                        r=lora_rank_vae, init_lora_weights="gaussian",
                        target_modules=target_modules_vae
                    )
                    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

                else:
                    # ========== Custom initialization branch ==========
                    # Mode: neither train_from_scratch nor scratch-checkpoint initialization is selected.
                    # Initialization steps:
                    # 1. Load pretrained SD-Turbo weights, as completed above.
                    # 2. Enable skip convolutions with random initialization instead of constant 1e-5.
                    # 3. Add LoRA to train the VAE encoder.
                    
                    print("Fine-tune mode (Custom): initializing LoRA for UNet and FULL VAE (Encoder+Decoder)")
                    
                    # 1. Enable encoder LoRA so collect_trainable_params includes encoder parameters.
                    self.train_encoder_lora = True  
                    
                    # 2. Randomly initialize skip convolutions, as in scratch training.
                    # Use normal_ rather than constant_(..., 1e-5).
                    print("--> Randomly initializing Skip Connections (Standard Gaussian)...")
                    torch.nn.init.normal_(vae.decoder.skip_conv_1.weight, mean=0.0, std=0.02)
                    torch.nn.init.normal_(vae.decoder.skip_conv_2.weight, mean=0.0, std=0.02)
                    torch.nn.init.normal_(vae.decoder.skip_conv_3.weight, mean=0.0, std=0.02)
                    torch.nn.init.normal_(vae.decoder.skip_conv_4.weight, mean=0.0, std=0.02)
                    
                    # 3. Add LoRA to both the encoder and decoder.
                    # Define the target layer names.
                    base_names = ("conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                        "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                        "to_k", "to_q", "to_v", "to_out.0")
                    
                    # Include encoder modules in LoRA targets.
                    target_modules_vae = self._collect_vae_target_modules(vae, include_encoder=True, base_names=base_names)
                    
                    print(f"--> Collected {len(target_modules_vae)} LoRA target modules for VAE (Encoder included).")
                    
                    vae_lora_config = LoraConfig(
                        r=lora_rank_vae, init_lora_weights="gaussian",
                        target_modules=target_modules_vae
                    )
                    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                               
                target_modules_unet = [
                    "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                    "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
                ]
                unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                    target_modules=target_modules_unet
                )
                unet.add_adapter(unet_lora_config)
                
                self.lora_rank_unet = lora_rank_unet
                self.lora_rank_vae = lora_rank_vae
                self.target_modules_vae = target_modules_vae
                self.target_modules_unet = target_modules_unet
                    
    # Check the setting before printing diagnostics.
    # Diagnostics at the end of Pix2Pix_Turbo.__init__.
            
        if self.enable_skip_conv:
            print("\n🔍 Skip Conv Initialization Check:")
            # Check that the attribute exists before reading it.
            if hasattr(vae.decoder, "skip_conv_1"):
                print(f"skip_conv_1 mean: {vae.decoder.skip_conv_1.weight.mean():.6f}")
                print(f"skip_conv_1 std:  {vae.decoder.skip_conv_1.weight.std():.6f}")
            else:
                print("⚠️ Warning: enable_skip_conv=True but skip_conv_1 layer is missing!")
        else:
            print("\n🔍 Skip Conv Initialization Check: SKIPPED (enable_skip_conv=False)")

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)
    
    def _collect_vae_target_modules(self, vae, include_encoder=False, fallback=None, base_names=("conv", "conv1", "conv2", "conv_shortcut", "conv_out", "to_k", "to_q", "to_v", "to_out.0")):
        """
        Collect VAE LoRA target module names.
        
        Args:
            vae: VAE model.
            include_encoder: Include encoder and decoder modules if True; decoder only otherwise.
            fallback: Fallback list of module names.
            base_names: Base module names to match.
        
        Returns:
            list: Target module names.
        """
        available = set()
        
        if include_encoder:
            # Collect encoder and decoder modules.
            for name, _m in vae.named_modules():
                last = name.split(".")[-1]
                for b in base_names:
                    if last == b:
                        available.add(b)
        else:
           
            # Collect decoder modules only.
            for name, _m in vae.decoder.named_modules():
                last = name.split(".")[-1]
                for b in base_names:
                    if last == b:
                        available.add(b)
            print(f"VAE Decoder available LoRA modules found: {available}")   
        
        if fallback:
            if include_encoder:
                # Use the full fallback list when including the encoder.
                result = sorted(list(set(fallback) & available)) or sorted(list(available))
            else:
                # Intersect with decoder modules to exclude encoder-only targets.
                result = sorted(list(set(fallback) & available)) or sorted(list(available))
        else:
            result = sorted(list(available))
        
        return result          
          
    def _collect_decoder_target_modules(self, vae, fallback=None, base_names=("conv", "conv1", "conv2", "conv_shortcut", "conv_out", "to_k", "to_q", "to_v", "to_out.0")):
        """
        Collect decoder-side VAE LoRA targets by matching leaf module names.
        Intersect fallback names, such as those in older checkpoints, with decoder modules to exclude encoder-only targets.
        """
        return self._collect_vae_target_modules(vae, include_encoder=False, fallback=fallback, base_names=base_names)
    
    

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()

        # Freeze all parameters first to avoid unintentionally trainable layers.
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        # Enable UNet LoRA training, preserving the original behavior.
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        # Keep unet.conv_in trainable, as in the original implementation.
        if hasattr(self.unet, "conv_in"):
            self.unet.conv_in.requires_grad_(True)

        # Use train_encoder_lora to control VAE encoder training.
        if self.train_encoder_lora:
            self.vae.requires_grad_(False)  # Disable gradients for all VAE parameters first.
            self.vae.train()  # Set training mode while keeping gradients disabled initially.
            #self.vae.requires_grad_(True)
            # Train LoRA throughout the VAE (encoder and decoder).
            for n, p in self.vae.named_parameters():
                if ("lora" in n) or ("skip_conv_" in n):
                    p.requires_grad = True
        else:
            # Train decoder LoRA only.
            for n, p in self.vae.decoder.named_parameters():
                if ("lora" in n) or ("skip_conv_" in n):
                    p.requires_grad = True
            if hasattr(self.vae.decoder, "conv_in"):
                self.vae.decoder.conv_in.requires_grad_(True)
            if hasattr(self.vae.decoder, "conv_out"): # Train the full output convolution when present.
                self.vae.decoder.conv_out.requires_grad_(True)
            # if "base_layer" in n and "lora" not in n:
            #     p.requires_grad = True
            # if ".0.lora" in n: # Here '0' denotes a submodule name.
            #     p.requires_grad = True
            

        # Explicitly ensure skip convolutions are trainable.
        # self.vae.decoder.skip_conv_1.requires_grad_(True)
        # self.vae.decoder.skip_conv_2.requires_grad_(True)
        # self.vae.decoder.skip_conv_3.requires_grad_(True)
        # self.vae.decoder.skip_conv_4.requires_grad_(True)
        # Explicitly ensure skip convolutions are trainable.
        # ========== Skip-convolution training configuration ==========
        if self.enable_skip_conv:
            if hasattr(self.vae.decoder, "skip_conv_1"): self.vae.decoder.skip_conv_1.requires_grad_(True)
            if hasattr(self.vae.decoder, "skip_conv_2"): self.vae.decoder.skip_conv_2.requires_grad_(True)
            if hasattr(self.vae.decoder, "skip_conv_3"): self.vae.decoder.skip_conv_3.requires_grad_(True)
            if hasattr(self.vae.decoder, "skip_conv_4"): self.vae.decoder.skip_conv_4.requires_grad_(True)
        # ========== End skip-convolution training configuration ==========
        
        if not self.train_encoder_lora:
            for n, p in self.vae.encoder.named_parameters():
                p.requires_grad = False



    # Forward pass with checkpointed decoding.
    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        # A. Encode text.
        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        # B. Encode the source image with the VAE.
        # Retrieve latents and skip activations.
        enc_output = self.vae.encode(c_t)
        if deterministic:
            # DPO/reference comparison must be noise-free; use the posterior mode.
            latents = enc_output.latent_dist.mode() * self.vae.config.scaling_factor
        else:
            # Blend encoded latents with noise for stochastic generation.
            latents_clean = enc_output.latent_dist.sample() * self.vae.config.scaling_factor
            if noise_map is None:
                noise_map = torch.randn_like(latents_clean)
            latents = latents_clean * r + noise_map * (1 - r)

        # Capture skip activations locally so subsequent model calls cannot overwrite them.
        current_skip_acts = self.vae.encoder.current_down_blocks

        # C. Run UNet inference.
        model_pred = self.unet(latents, self.timesteps, encoder_hidden_states=caption_enc).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, latents, return_dict=True).prev_sample
        x_denoised = x_denoised.to(model_pred.dtype)

        # D. Wrap decoding in activation checkpointing.
        # Include all operations requiring gradients in the helper function.
        def _decode_with_skips(latents_in, *skip_features):
            # Rebuild the skip-activation list from individual tensor arguments.
            self.vae.decoder.incoming_skip_acts = list(skip_features)
            return self.vae.decode(latents_in / self.vae.config.scaling_factor).sample

        # Use checkpointing only during training when denoised latents require gradients.
        if self.training and x_denoised.requires_grad:
            # Pass tensor inputs explicitly to the checkpoint function.
            # Expand the skip-activation list into separate arguments.
            output_image = checkpoint(
                _decode_with_skips, 
                x_denoised, 
                *current_skip_acts, 
                use_reentrant=False # Use non-reentrant checkpointing.
            )
        else:
            # Decode directly for validation, inference, or the reference model.
            self.vae.decoder.incoming_skip_acts = current_skip_acts
            output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample

        # Clear stored decoder skip activations.
        self.vae.decoder.incoming_skip_acts = None
        
        return output_image.clamp(-1, 1)

    def save_model(self, outf):
        sd = {}

        # Get target modules from unet's active lora modules
        unet_lora_layers = [name for name, module in self.unet.named_modules() if 'lora' in name]
        vae_lora_layers = [name for name, module in self.vae.named_modules() if 'lora' in name]

        # Get unique target modules
        unet_target_modules = list(set([name.split('.')[0] for name in unet_lora_layers if 'lora' in name]))
        vae_target_modules = list(set([name.split('.')[0] for name in vae_lora_layers if 'lora' in name]))

        # Get rank by finding a LoRA up module
        def get_lora_rank(model):
            for name, module in model.named_modules():
                if 'lora_up' in name:
                    return module.in_features
            return None  # Return None if no LoRA rank is found.

        unet_rank = get_lora_rank(self.unet) or 8  # Default to rank 8 if no UNet LoRA rank is found.
        vae_rank = get_lora_rank(self.vae) or 4    # Default to rank 4 if no VAE LoRA rank is found.

        sd["unet_lora_target_modules"] = unet_target_modules
        sd["vae_lora_target_modules"] = vae_target_modules
        sd["rank_unet"] = unet_rank
        sd["rank_vae"] = vae_rank
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}

        torch.save(sd, outf)
    
    def _export_lora_sd(self):
        """
        Return lightweight model state (LoRA, conv_in, VAE skip_conv_*) as a dictionary without writing to disk.
        """
        # 1. Collect target modules.
        unet_lora_layers = [name for name, _ in self.unet.named_modules() if 'lora' in name]
        vae_lora_layers  = [name for name, _ in self.vae.named_modules() if 'lora' in name]
        unet_target_modules = list(set([name.split('.')[0] for name in unet_lora_layers if 'lora' in name]))
        vae_target_modules  = list(set([name.split('.')[0] for name in vae_lora_layers if 'lora' in name]))

        # 2. Retrieve LoRA ranks.
        def get_lora_rank(model):
            for name, module in model.named_modules():
                if 'lora_up' in name:
                    return module.in_features
            return None
        unet_rank = get_lora_rank(self.unet) or 8
        vae_rank  = get_lora_rank(self.vae)  or 4

        # 3. Export only the trainable subset.
        sd_unet = {k: v for k, v in self.unet.state_dict().items() if ("lora" in k or "conv_in" in k)}
        sd_vae  = {k: v for k, v in self.vae.state_dict().items()  if ("lora" in k or "skip"    in k)}

        return {
            "unet_lora_target_modules": unet_target_modules,
            "vae_lora_target_modules":  vae_target_modules,
            "rank_unet": unet_rank,
            "rank_vae":  vae_rank,
            "state_dict_unet": sd_unet,
            "state_dict_vae":  sd_vae,
            "format": "light_v1",
        }
            
    def save_light_checkpoint(self, optimizer, lr_scheduler, global_step, out_dir, train_loader=None, keep_last=3):
        """
        Lightweight checkpoint: model LoRA and added layers, optimizer, scheduler, global_step, and optional sampler state.
        Write a single file to {out_dir}/checkpoint_{global_step}.pt.
        """
        import glob, os, torch
        os.makedirs(out_dir, exist_ok=True)
        ckpt = {
            "global_step": int(global_step),
            "model": self._export_lora_sd(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }

        # Save the DataLoader sampler state when available.
        if train_loader is not None and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "state_dict"):
            ckpt["sampler_state"] = train_loader.sampler.state_dict()

        path = os.path.join(out_dir, f"checkpoint_{global_step}.pt")
        torch.save(ckpt, path)

        # Remove old checkpoints, retaining the latest keep_last files.
        all_ckpts = sorted(glob.glob(os.path.join(out_dir, "checkpoint_*.pt")),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]))
        if len(all_ckpts) > keep_last:
            for old in all_ckpts[:-keep_last]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        return path
    @torch.no_grad()
    
    def load_light_checkpoint(self, path, optimizer=None, lr_scheduler=None, train_loader=None, map_location="cpu"):
        """
        Robust loader that accepts:
        - new light format: {"global_step", "model": {state_dict_unet, state_dict_vae, ...}, "optimizer", "lr_scheduler"}
        - legacy save_model() output: (a dict containing 'state_dict_unet'/'state_dict_vae'/... )
        - full checkpoint: {"model_state_dict": {...}, "optimizer_state_dict":..., ...}

        Returns: int(global_step)
        """
        import torch, os
        ckpt = torch.load(path, map_location=map_location)

        # --- 1) identify model payload (sd) ---
        if "model" in ckpt:
            sd = ckpt["model"]
        elif ("state_dict_unet" in ckpt) or ("state_dict_vae" in ckpt) or ("unet_lora_target_modules" in ckpt):
            sd = ckpt
        elif "model_state_dict" in ckpt or "state_dict" in ckpt:
            full_sd = ckpt.get("model_state_dict", ckpt.get("state_dict"))
            norm = {}
            for k, v in full_sd.items():
                nk = k[7:] if k.startswith("module.") else k
                norm[nk] = v
            sd_unet, sd_vae = {}, {}
            for k, v in norm.items():
                kl = k.lower()
                if kl.startswith("unet.") or ".unet." in k:
                    subk = k.split("unet.", 1)[1] if "unet." in k else k
                    if subk.startswith("unet."):
                        subk = subk[len("unet."):]
                    sd_unet[subk] = v
                elif kl.startswith("vae.") or ".vae." in k or "skip_conv" in kl or "decoder" in kl or "encoder" in kl:
                    subk = k.split("vae.", 1)[1] if "vae." in k else k
                    if subk.startswith("vae."):
                        subk = subk[len("vae."):]
                    sd_vae[subk] = v
                else:
                    if any(x in kl for x in ["lora", "conv_in", "proj_in", "to_k", "to_q", "to_v"]):
                        kk = k.split("unet.", 1)[1] if "unet." in k else k
                        sd_unet[kk] = v
            sd = {
                "state_dict_unet": sd_unet,
                "state_dict_vae": sd_vae,
                "unet_lora_target_modules": ckpt.get("unet_lora_target_modules", []),
                "vae_lora_target_modules": ckpt.get("vae_lora_target_modules", []),
                "rank_unet": ckpt.get("rank_unet", None),
                "rank_vae": ckpt.get("rank_vae", None),
            }
        else:
            raise ValueError(f"Unknown checkpoint format (missing 'model'/'state_dict_unet'/'model_state_dict'). Keys: {list(ckpt.keys())}")

        # --- 2) ensure adapters exist (try/except for idempotence) ---
        try:
            from peft import LoraConfig
            unet_target_modules = sd.get("unet_lora_target_modules", [])
            vae_target_modules = sd.get("vae_lora_target_modules", [])
            if unet_target_modules:
                unet_lora_config = LoraConfig(r=sd.get("rank_unet", 8), init_lora_weights="gaussian",
                                            target_modules=unet_target_modules)
                try:
                    self.unet.add_adapter(unet_lora_config)
                except Exception:
                    pass
            if vae_target_modules:
                # Ensure adapters exist without forcing replacement; decoder-only adapters remain supported.
                vae_lora_config = LoraConfig(r=sd.get("rank_vae", 4), init_lora_weights="gaussian",
                                            target_modules=vae_target_modules or self._collect_decoder_target_modules(self.vae))
                try:
                    self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                except Exception:
                    pass
        except Exception:
            pass

        # --- helper: merge into submodule state dict with key normalization ---
        def _merge_into(submodule, incoming_dict):
            cur = submodule.state_dict()
            updated = dict(cur)
            for k_in, v in incoming_dict.items():
                kk = k_in
                if kk.startswith("module."):
                    kk = kk[len("module."):]
                if kk.startswith("unet."):
                    kk = kk[len("unet."):]
                if kk.startswith("vae."):
                    kk = kk[len("vae."):]
                if kk in updated:
                    try:
                        updated[kk] = v.to(updated[kk].device)
                    except Exception:
                        updated[kk] = v
                else:
                    matches = [k2 for k2 in updated.keys() if k2.endswith(kk)]
                    if matches:
                        tgt = matches[0]
                        try:
                            updated[tgt] = v.to(updated[tgt].device)
                        except Exception:
                            updated[tgt] = v
                    else:
                        pass
            submodule.load_state_dict(updated, strict=False)

        # --- 3) merge unet & vae parts ---
        sd_unet = sd.get("state_dict_unet", {})
        sd_vae = sd.get("state_dict_vae", {})
        if sd_unet:
            _merge_into(self.unet, sd_unet)
        if sd_vae:
            _merge_into(self.vae, sd_vae)

        # --- 4) restore optimizer / lr_scheduler if available ---
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")
        if optimizer is not None and "optimizer" in ckpt:
            opt_sd = ckpt["optimizer"]
            for state in opt_sd.get("state", {}).values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            optimizer.load_state_dict(opt_sd)

        if lr_scheduler is not None and "lr_scheduler" in ckpt:
            try:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            except Exception:
                pass

        if train_loader is not None and "sampler_state" in ckpt:
            try:
                train_loader.sampler.load_state_dict(ckpt["sampler_state"])
                print("[load_light_checkpoint] Sampler state restored successfully.")
            except Exception as e:
                print(f"[load_light_checkpoint] Warning: could not restore sampler state ({e})")

        gs = ckpt.get("global_step", ckpt.get("step", 0))
        return int(gs)
    
    def load_full_checkpoint_for_scratch_training(self, path, optimizer=None, lr_scheduler=None, train_loader=None):
        """
        Load a full checkpoint for train_from_scratch.
        """
        import torch
        print(f"Loading full checkpoint from {path} ...")
        ckpt = torch.load(path, map_location="cpu")
        
        # Check the checkpoint format.
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt and ckpt["model"].get("format") == "full_scratch_v1":
            state_dict = ckpt["model"]["full_state_dict"]
        else:
            raise ValueError("Cannot load light checkpoint for train_from_scratch resume. Need full checkpoint.")
        
        # Remove the module. prefix.
        cleaned_state = {}
        for k, v in state_dict.items():
            clean_key = k[7:] if k.startswith("module.") else k
            cleaned_state[clean_key] = v
        
        # ========== Remap keys for LoRA-wrapped layers ==========
        # Checkpoints trained from scratch typically include base_layer in their keys.
        key_mapping = {
            "vae.encoder.conv_in.weight": "vae.encoder.conv_in.base_layer.weight",
            "vae.encoder.conv_in.bias": "vae.encoder.conv_in.base_layer.bias",
            "vae.decoder.conv_in.weight": "vae.decoder.conv_in.base_layer.weight",
            "vae.decoder.conv_in.bias": "vae.decoder.conv_in.base_layer.bias",
            # Map skip convolutions if present in the checkpoint.
            "vae.decoder.skip_conv_1.weight": "vae.decoder.skip_conv_1.base_layer.weight",
            "vae.decoder.skip_conv_2.weight": "vae.decoder.skip_conv_2.base_layer.weight",
            "vae.decoder.skip_conv_3.weight": "vae.decoder.skip_conv_3.base_layer.weight",
            "vae.decoder.skip_conv_4.weight": "vae.decoder.skip_conv_4.base_layer.weight",
        }
        
        remapped_state = {}
        for k, v in cleaned_state.items():
            if k in key_mapping:
                new_key = key_mapping[k]
                remapped_state[new_key] = v
                # print(f"  Remapped: {k} -> {new_key}") # Enable this print only when detailed key-remapping diagnostics are needed.
            else:
                remapped_state[k] = v
        
        # Load weights.
        result = self.load_state_dict(remapped_state, strict=False)
        
        # ========== Handle missing keys ==========
        if result.missing_keys:
            print(f"Handling {len(result.missing_keys)} missing keys...")
            current_state = self.state_dict()
            for key in result.missing_keys:
                # Case A: initialize missing lora_B weights to zero, disabling the LoRA update.
                if 'lora_B' in key:
                    current_state[key].zero_()
                    # print(f"  Zeroed lora_B: {key}")
                
                # Case B: a skip convolution added during stage 2.
                # The layer is absent from the checkpoint; retain its random initialization from __init__.
                elif 'skip_conv' in key:
                    pass 
                    # print(f"  Kept initialized skip_conv: {key}")

                # Case C: retain random initialization for LoRA A.
                elif 'lora_A' in key:
                    pass 
        
        print("Loaded full scratch training checkpoint successfully.")
        
        # Load optimizer and other state only when available.
        device = next(self.parameters()).device
        if optimizer is not None and "optimizer" in ckpt:
            try:
                opt_sd = ckpt["optimizer"]
                for state in opt_sd.get("state", {}).values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                optimizer.load_state_dict(opt_sd)
                print("Optimizer state loaded.")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
        
        if lr_scheduler is not None and "lr_scheduler" in ckpt:
            try:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                print("LR Scheduler state loaded.")
            except Exception as e:
                print(f"Warning: Failed to load LR Scheduler: {e}")
                
        # ========== Guard diagnostics against missing attributes ==========
        print("\n" + "="*60)
        print("FINAL VERIFICATION")
        print("="*60)

        # Check whether skip_conv_1 exists.
        if hasattr(self.vae.decoder, 'skip_conv_1'):
            print("\n1️⃣ Skip Conv Type:")
            print(f"  skip_conv_1 type: {type(self.vae.decoder.skip_conv_1).__name__}")

            print("\n2️⃣ Skip Conv Structure:")
            skip_1 = self.vae.decoder.skip_conv_1
            print(f"  Has base_layer: {hasattr(skip_1, 'base_layer')}")
            print(f"  Has lora_A: {hasattr(skip_1, 'lora_A')}")

            print("\n3️⃣ Skip Conv Weights:")
            # Inspect base_layer when LoRA-wrapped, or weight otherwise.
            if hasattr(skip_1, 'base_layer'):
                w = skip_1.base_layer.weight
            elif hasattr(skip_1, 'weight'):
                w = skip_1.weight
            else:
                w = None
            
            if w is not None:
                print(f"  Weight mean: {w.mean():.6f}")
                print(f"  Weight std: {w.std():.6f}")
            
            # Check whether LoRA B is zero, as expected at the start of fine-tuning.
            if hasattr(skip_1, 'lora_B'):
                # Select the first adapter.
                first_key = list(skip_1.lora_B.keys())[0]
                lora_b = skip_1.lora_B[first_key].weight
                is_zero = torch.allclose(lora_b, torch.zeros_like(lora_b), atol=1e-6)
                print(f"  lora_B[{first_key}] is initialized to zero: {is_zero}")
        else:
            print("\n⚠️ Skip Convs NOT detected in Decoder (Correct for Stage 1 / incorrect for Stage 2).")

        print("="*60 + "\n")
        
        return ckpt.get("global_step", 0)
    
    
    def load_full_checkpoint(self, path, optimizer=None, lr_scheduler=None, train_loader=None):
        """
        Load a full checkpoint containing LoRA, base weights, and optimizer state.
        Supported format: full_scratch_v1.
        Checkpoint keys are expected to include base_layer, lora_A, and lora_B paths.
        """
        import torch
        print(f"Loading full checkpoint from {path} ...")
        
        # 1. Load the checkpoint file.
        ckpt = torch.load(path, map_location="cpu")
        
        # 2. Extract the state dictionary.
        if "model" in ckpt and isinstance(ckpt["model"], dict) and "full_state_dict" in ckpt["model"]:
            state_dict = ckpt["model"]["full_state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            raise ValueError("Checkpoint format not recognized. Expected 'model.full_state_dict' or 'model_state_dict'.")

        # 3. Remove the module. prefix introduced by DDP.
        # Use checkpoint keys directly because they already contain base_layer/lora_A paths.
        cleaned_state = {}
        for k, v in state_dict.items():
            # Remove the module. prefix.
            clean_key = k[7:] if k.startswith("module.") else k
            cleaned_state[clean_key] = v

        # 4. Collect current model keys for diagnostic comparisons.
        current_model_keys = set(self.state_dict().keys())
        loaded_keys = set(cleaned_state.keys())
        
        # 5. Load the state dictionary.
        # Allow unmatched entries, such as buffers, with strict=False; inspect missing_keys below.
        result = self.load_state_dict(cleaned_state, strict=False)
        
        # 6. Report loading results in detail.
        print("\n" + "="*40)
        print("Checkpoint Loading Report")
        print("="*40)
        
        # Check keys expected by the model but not loaded from the checkpoint.
        if result.missing_keys:
            real_missing = [k for k in result.missing_keys if "text_encoder" not in k] # Ignore frozen text-encoder layers.
            if len(real_missing) > 0:
                print(f"⚠️ Warning: {len(real_missing)} keys missing in checkpoint (Model expects them):")
                for k in real_missing[:10]: # Print only the first ten entries.
                    print(f"  - {k}")
                if len(real_missing) > 10: print("  ... and more.")
            else:
                print("✅ No critical missing keys.")
        else:
            print("✅ All model keys matched successfully.")

        # Check checkpoint keys that the model does not expect.
        if result.unexpected_keys:
            print(f"ℹ️ Info: {len(result.unexpected_keys)} unexpected keys in checkpoint (Model doesn't need them):")
            # These are typically text-encoder parameters or buffers from older versions.
            # for k in result.unexpected_keys[:5]: print(f"  - {k}")
        else:
            print("✅ No unexpected keys.")

        # 7. Verify that LoRA and skip-convolution values were loaded.
        # Inspect a few representative layers.
        print("\n🔍 Value Verification:")
        
        def check_layer_loaded(layer_name, key_suffix, expected_pattern):
            # Find the corresponding key in cleaned_state.
            found = False
            for k, v in cleaned_state.items():
                if layer_name in k and key_suffix in k:
                    found = True
                    # Retrieve the corresponding model parameter.
                    try:
                        curr_param = self.state_dict()[k]
                        # Check for nonzero values; LoRA B may be zero, but base-layer weights should not be all zero.
                        is_zero = torch.allclose(curr_param, torch.zeros_like(curr_param), atol=1e-6)
                        stat_str = "Zero" if is_zero else "Non-Zero"
                        print(f"  ✅ Verified {k[-40:]}: Loaded ({stat_str})")
                    except KeyError:
                        print(f"  ❌ Failed to access {k} in current model.")
                    break
            if not found:
                print(f"  ⚠️ Could not find pattern '{layer_name}...{key_suffix}' in loaded checkpoint.")

        # Verify the VAE skip-convolution base layer.
        check_layer_loaded("skip_conv_1", "base_layer.weight", "Non-Zero")
        # Verify UNet LoRA A.
        check_layer_loaded("unet", "lora_A.default.weight", "Non-Zero")
        # Verify VAE LoRA A.
        check_layer_loaded("vae", "lora_A.vae_skip.weight", "Non-Zero")

        print("="*40 + "\n")

        # 8. Load optimizer state.
        device = next(self.parameters()).device
        if optimizer is not None and "optimizer" in ckpt:
            try:
                opt_sd = ckpt["optimizer"]
                # Ensure optimizer state is on the correct device.
                for state in opt_sd.get("state", {}).values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                optimizer.load_state_dict(opt_sd)
                print("✅ Optimizer state loaded.")
            except Exception as e:
                print(f"❌ Warning: Failed to load optimizer state: {e}")
        else:
            print("ℹ️ Optimizer not loaded (not provided or not in ckpt).")

        # 9. Load learning-rate scheduler state.
        if lr_scheduler is not None and "lr_scheduler" in ckpt:
            try:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                print("✅ LR Scheduler state loaded.")
            except Exception as e:
                print(f"❌ Warning: Failed to load LR Scheduler: {e}")
        
        # 10. Restore DataLoader sampler state if available.
        if train_loader is not None and "sampler_state" in ckpt:
            try:
                if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "load_state_dict"):
                    train_loader.sampler.load_state_dict(ckpt["sampler_state"])
                    print("✅ DataSampler state restored.")
            except Exception as e:
                print(f"ℹ️ Could not restore sampler state: {e}")

        # Return global_step.
        return ckpt.get("global_step", 0)
    
    
    def load_dpo_full_checkpoint_for_scratch_training(self, path, optimizer=None, lr_scheduler=None, train_loader=None):
        """
        Load a full checkpoint for train_from_scratch.
        Determine whether base_layer key remapping is required.
        """
        import torch
        print(f"Loading full checkpoint from {path} ...")
        ckpt = torch.load(path, map_location="cpu")
        
        # 1. Extract the state dictionary.
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt and ckpt["model"].get("format") == "full_scratch_v1":
            state_dict = ckpt["model"]["full_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            raise ValueError("Cannot load light checkpoint for train_from_scratch resume. Need full checkpoint.")
        
        # 2. Remove the module. prefix.
        cleaned_state = {}
        for k, v in state_dict.items():
            clean_key = k[7:] if k.startswith("module.") else k
            cleaned_state[clean_key] = v
        
        # 3. Map keys according to the current model structure.
        remapped_state = {}
        
        # Determine whether a model layer is LoRA-wrapped and has a base_layer.
        def has_base_layer(module_name):
            parts = module_name.split('.')
            m = self
            try:
                for p in parts:
                    m = getattr(m, p)
                return hasattr(m, 'base_layer')
            except AttributeError:
                return False

        # Layers requiring explicit inspection.
        special_layers = [
            "vae.encoder.conv_in", "vae.decoder.conv_in",
            "vae.decoder.skip_conv_1", "vae.decoder.skip_conv_2",
            "vae.decoder.skip_conv_3", "vae.decoder.skip_conv_4"
        ]

        for k, v in cleaned_state.items():
            target_key = k
            
            # Check whether this key belongs to a layer requiring special handling.
            # For example: k = "vae.decoder.skip_conv_1.weight".
            for sp_layer in special_layers:
                # Match weight or bias entries.
                if k == f"{sp_layer}.weight" or k == f"{sp_layer}.bias":
                    # Remap the key only if the current model layer actually has a base_layer.
                    if has_base_layer(sp_layer):
                        suffix = k.split('.')[-1] # weight or bias
                        target_key = f"{sp_layer}.base_layer.{suffix}"
                        # print(f"  [Mapping] {k} -> {target_key} (LoRA detected)")
                    else:
                        # Keep the original key for an unwrapped layer.
                        target_key = k
                        # print(f"  [Keep] {k} (No LoRA detected)")
                    break
            
            remapped_state[target_key] = v
        
        # 4. Load weights.
        result = self.load_state_dict(remapped_state, strict=False)
        
        # 5. Handle errors and report diagnostics.
        if result.missing_keys:
            print(f"Handling {len(result.missing_keys)} missing keys...")
            for key in result.missing_keys:
                if 'lora_B' in key:
                    pass # Zero initialization is expected for missing LoRA B weights.
                elif 'skip_conv' in key:
                    # A missing skip convolution here indicates a loading problem.
                    print(f"⚠️ CRITICAL WARNING: Missing skip_conv key: {key}. Weights will be random!")

        print("Loaded full scratch training checkpoint successfully.")
        
        # 6. Load optimizer state without changing the existing behavior.
        device = next(self.parameters()).device
        if optimizer is not None and "optimizer" in ckpt:
            try:
                opt_sd = ckpt["optimizer"]
                for state in opt_sd.get("state", {}).values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                optimizer.load_state_dict(opt_sd)
                print("Optimizer state loaded.")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
        
        if lr_scheduler is not None and "lr_scheduler" in ckpt:
            try:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                print("LR Scheduler state loaded.")
            except Exception as e:
                print(f"Warning: Failed to load LR Scheduler: {e}")
                
        # Print diagnostics.
        if hasattr(self.vae.decoder, 'skip_conv_1'):
             s1 = self.vae.decoder.skip_conv_1
             w = s1.base_layer.weight if hasattr(s1, 'base_layer') else s1.weight
             print(f"[Diagnose] skip_conv_1 mean: {w.mean():.6f} (Should NOT be 0.000010)")

        return ckpt.get("global_step", 0)
    
    def save_full_checkpoint_for_scratch_training(self, optimizer, lr_scheduler, global_step, out_dir, train_loader=None, keep_last=3):
        """
        Save a full checkpoint for train_from_scratch.
        """
        import os, glob, torch
        os.makedirs(out_dir, exist_ok=True)
        
        ckpt = {
            "global_step": int(global_step),
            "model": {
                "format": "full_scratch_v1",
                "full_state_dict": self.state_dict(),
            },
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        
        if train_loader is not None and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "state_dict"):
            ckpt["sampler_state"] = train_loader.sampler.state_dict()
        
        path = os.path.join(out_dir, f"full_checkpoint_{global_step}.pt")
        torch.save(ckpt, path)
        
        all_ckpts = sorted(glob.glob(os.path.join(out_dir, "full_checkpoint_*.pt")),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
        if len(all_ckpts) > keep_last:
            for old in all_ckpts[:-keep_last]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        
        return path
