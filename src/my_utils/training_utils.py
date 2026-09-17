import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn
from glob import glob


def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_steps", type=int, default=0, help="Stop after N new synchronized training steps; skip checkpoint writing and evaluation.")
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_l1", default=0.1, type=float) 
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)
    parser.add_argument("--lambda_structural", default=0.5, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resize_512", type=str)
    parser.add_argument("--test_image_prep", default="resize_512", type=str)
    
    parser.add_argument("--save_rng_state", type=bool, default=True,
                      help="Whether to save random number generator states")
    parser.add_argument("--verify_rng_state", type=bool, default=True,
                      help="Whether to verify RNG states after loading checkpoint")

    # validation eval args
    parser.add_argument("--eval_freq", default=1920, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=192, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=112, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--resume_from",type=str,default=None, help="A path to a checkpoint to resume training from.")
    parser.add_argument("--pretrained_name", type=str, default=None,
                        help="Built-in lightweight weights name, e.g. 'edge_to_image' or 'sketch_to_image_stochastic'.")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to local lightweight weights/LoRA checkpoint to load at initialization.")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=560,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=10e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch without loading any pretrained weights")
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def parse_args_unpaired_training():
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=None)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--resume_from",type=str,default=None, help="A path to a checkpoint to resume training from.")

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--validation_steps", type=int, default=500,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=560)

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=5e-6,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    args = parser.parse_args()
    return args

def safe_normalize_brightfield(pil_image):
    """Normalize a PIL image using intensity percentiles."""
    import numpy as np
    from PIL import Image
    
    # PIL → numpy
    img_array = np.array(pil_image).astype(np.float32)
    
    # Use numpy.percentile instead of torch.quantile.
    p1 = np.percentile(img_array, 1)
    p99 = np.percentile(img_array, 99)
    
    #print(f"Brightfield percentiles: p1={p1:.6f}, p99={p99:.6f}")
    
    if p99 > p1 + 1e-6:  # Ensure sufficient dynamic range.
        normalized = np.clip((img_array - p1) / (p99 - p1) * 255, 0, 255)
    else:
        print("WARNING: Very small dynamic range, using original")
        normalized = img_array
    
    # numpy → PIL
    return Image.fromarray(normalized.astype(np.uint8))

def normalize_brightfield_pil(pil_image):
    """Adjust brightness and contrast at the PIL image level."""
    import numpy as np
    
    # PIL -> numpy
    img_array = np.array(pil_image)
    
    # Standardize image intensities.
    mean_val = np.mean(img_array)
    std_val = np.std(img_array)
    
    if std_val > 1e-6:
        normalized = (img_array - mean_val) / std_val
        normalized = np.clip(normalized, -3, 3)
        normalized = ((normalized + 3) / 6 * 255).astype(np.uint8)
    else:
        normalized = img_array
    
    # numpy -> PIL
    return Image.fromarray(normalized)

def safe_normalize_fluorescence(img_tensor):
    """Normalize fluorescence intensities with safeguards for sparse signals."""
    # Ensure the input is in the expected range.
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # Compute percentiles while excluding near-zero background pixels.
    nonzero_mask = img_tensor > 0.01  # Treat values below 0.01 as background.
    
    if torch.sum(nonzero_mask) > 100:  # Require enough signal pixels for percentile estimation.
        signal_pixels = img_tensor[nonzero_mask]
        p5 = torch.quantile(signal_pixels, 0.05)
        p95 = torch.quantile(signal_pixels, 0.95)
        print(f"Fluorescence signal percentiles: p5={p5:.6f}, p95={p95:.6f}")
    else:
        # Use whole-image statistics if there are too few signal pixels.
        p5 = torch.quantile(img_tensor, 0.05)
        p95 = torch.quantile(img_tensor, 0.95)
        print(f"Fluorescence full percentiles: p5={p5:.6f}, p95={p95:.6f}")
    
    if p95 > p5 + 1e-8:
        normalized = torch.clamp((img_tensor - p5) / (p95 - p5), 0, 1)
    else:
        normalized = img_tensor
    
    # Check the final result.
    assert not torch.isnan(normalized).any(), "NaN detected in fluorescence normalization"
    assert not torch.isinf(normalized).any(), "Inf detected in fluorescence normalization"
    
    return normalized

def percentile_normalize(img_tensor, low_perc=1, high_perc=99):
    """Apply percentile normalization independently to each image."""
    low_val = torch.quantile(img_tensor, low_perc/100.0)
    high_val = torch.quantile(img_tensor, high_perc/100.0)
    img_normalized = torch.clamp((img_tensor - low_val) / (high_val - low_val), 0, 1)
    return img_normalized


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        Itialize the paired dataset object for loading and transforming paired data samples
        from specified dataset folders.

        This constructor sets up the paths to input and output folders based on the specified 'split',
        loads the captions (or prompts) for the input images, and prepares the transformations and
        tokenizer to be applied on the data.

        Parameters:
        - dataset_folder (str): The root folder containing the dataset, expected to include
                                sub-folders for different splits (e.g., 'train_A', 'train_B').
        - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
                       sub-folders and caption files within the dataset folder.
        - image_prep (str): The image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image, 
        its corresponding output image, the captions associated with the input image, 
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images, 
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally 
        and passed to this class through the `image_prep` parameter during initialization. The 
        tokenization process relies on the `tokenizer` also provided at initialization, which 
        should be compatible with the models intended to be used with this dataset.
        """
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name)).convert("RGB")
        #print("Original image mode:", input_img.mode)
        output_img = Image.open(os.path.join(self.output_folder, img_name)).convert("RGB")
        caption = self.captions[img_name]

        # input images scaled to 0,1
        # input_img = safe_normalize_brightfield(input_img)
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        #print("Tensor shape after T:", img_t.shape)
        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        # output_t = safe_normalize_fluorescence(output_t)  # Normalize the intensity distribution first.
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "img_name": img_name,
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        # find all images in the source and target folders with all IMG extensions
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }
    


def structural_loss(fake_image, real_image):

    def ssim_loss(x, y, kernel_size=11, sigma=1.5):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Match the input channel count dynamically.
        channels = x.shape[1]
        kernel = torch.ones((channels, 1, kernel_size, kernel_size)).to(x.device) / (kernel_size * kernel_size)

        mu_x = F_nn.conv2d(x, kernel, padding=kernel_size // 2, groups=channels)
        mu_y = F_nn.conv2d(y, kernel, padding=kernel_size // 2, groups=channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F_nn.conv2d(x * x, kernel, padding=kernel_size // 2, groups=channels) - mu_x_sq
        sigma_y_sq = F_nn.conv2d(y * y, kernel, padding=kernel_size // 2, groups=channels) - mu_y_sq
        sigma_xy = F_nn.conv2d(x * y, kernel, padding=kernel_size // 2, groups=channels) - mu_xy

        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        return 1 - ssim.mean()    
    
    # Compute multiscale feature-consistency loss.
    def feature_consistency_loss(x, y):
        losses = []
        for scale in [1, 0.5, 0.25]:
            if scale != 1:
                curr_x = F_nn.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
                curr_y = F_nn.interpolate(y, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                curr_x = x
                curr_y = y
            
            # Compute image gradients.
            grad_x_x = curr_x[:, :, :, 1:] - curr_x[:, :, :, :-1]
            grad_x_y = curr_x[:, :, 1:, :] - curr_x[:, :, :-1, :]
            grad_y_x = curr_y[:, :, :, 1:] - curr_y[:, :, :, :-1]
            grad_y_y = curr_y[:, :, 1:, :] - curr_y[:, :, :-1, :]
            
            # Compute gradient differences.
            loss_grad_x = F_nn.l1_loss(grad_x_x, grad_y_x) + F_nn.l1_loss(grad_x_y, grad_y_y)
            losses.append(loss_grad_x)
        
        return sum(losses)
    
    # Combine the losses.
    ssim = ssim_loss(fake_image, real_image)
    feat_cons = feature_consistency_loss(fake_image, real_image)
    
    return ssim + 0.1 * feat_cons
