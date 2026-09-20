# OrganelleVista

Virtual staining from brightfield microscopy to predicted fluorescence images.

**Cell types:** 3T3 · HepaRG · HUVEC · Jump Cell · HFF

[Demo](#system-demo) · [Setup](#setup) · [Datasets](#dataset-layout) · [Training](#training-workflow) · [Inference](#folder-inference)

## System demo

Select a cell type and target component, upload a brightfield image, and generate a predicted fluorescence image. The interface supports side-by-side inspection, brightness/contrast adjustment, and result download.

https://github.com/user-attachments/assets/5371ce5b-4097-495c-be08-50041810ac08

*Play the full 1 min 45 sec demo above.* · [Open video](https://github.com/user-attachments/assets/5371ce5b-4097-495c-be08-50041810ac08) · [Download original MP4](https://github.com/Johnnyyuqi/OrganelleVista/raw/refs/heads/main/docs/assets/organellevista-demo.mp4)

The video demonstrates the system interface. This repository provides training and folder-inference code; a hosted interactive demo is not linked here.

## Setup

Prerequisites: Conda/Miniconda, Linux x86_64, and an NVIDIA GPU/driver compatible with the CUDA 11.8 PyTorch build. Training and inference call CUDA directly; CPU-only execution is not supported. Steps 1–2 below use two GPUs; Step 3 and inference use one. The DPO example also requires BF16 support.

### First-time installation

Download or clone the source, then create a new environment. The environment is **not** included in the repository.

```bash
cd /path/to/OrganelleVista
conda create -n organellevista python=3.10 pip -y
conda activate organellevista

# Install a specific CUDA build before the remaining dependencies.
python -m pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
```

This setup was tested in a fresh Python 3.10 environment with Torch 2.0.1 / torchvision 0.15.2 (CUDA 11.8), xformers 0.0.22, and RTX 4090 GPUs. Dependency checks, short training runs, and folder inference passed; see the [validation record](docs/VALIDATION.md#clean-environment-and-gpu-validation) for scope and limitations. Run installation from the repository root. Keep the `setuptools` pin in [requirements.txt](requirements.txt): older dependencies require its `pkg_resources` module.

### Check the environment

```bash
python src/train_pretrained_pix2pix_turbo.py --help
python src/train_dpo_pix2pix_turbo.py --help
python src/inference_simple_folder.py --help
python -c "import torch; print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable: check the NVIDIA driver and GPU access'"
python -m xformers.info
```

Help commands check imports; CUDA detection and xformers information do not prove training works. Run the short training check described below after preparing data and weights. Resolve any `pip check` errors before training.

### Later sessions and data paths

```bash
conda activate organellevista
cd /path/to/OrganelleVista
export DATA_ROOT=/path/to/data
export MODEL_ROOT=/path/to/checkpoints
```

Replace both paths with your real dataset and checkpoint roots. If reusing the author's existing environment, activate `img2img-turbo` instead; that name is a local choice, not a project requirement. Do not recreate or overwrite an existing environment merely to change its name.

Datasets and experiment checkpoints must be supplied separately. SD-Turbo/CLIP/VGG weights must be downloaded or cached before offline operation. See [environment notes](docs/IMPLEMENTATION_NOTES.md#environment-and-pretrained-weights) for the historical environment and [validation results](docs/VALIDATION.md) for what has actually been tested.

## Dataset layout

Steps 1 and 2 use matching filenames in two folders:

```text
dataset/
  train_A/sample_001.png
  train_B/sample_001.png
  train_prompts.json
```

- **Step 1:** A and B contain the same brightfield image; noise is added internally.
- **Step 2:** A contains brightfield images; B contains paired fluorescence targets.
- Both stages create a 90:10 training/validation split. Supply at least two image pairs.

Prompt JSON files map filenames to text:

```json
{
  "sample_001.png": "fluorescence staining description",
  "sample_002.png": "fluorescence staining description"
}
```

For SSL without descriptive prompts, use empty strings. For reproductions, use the actual experiment prompts.

## Training workflow

Complete [Setup](#setup) and prepare the [datasets](#dataset-layout) first. Run the commands below from the repository root with your environment activated and `DATA_ROOT` / `MODEL_ROOT` set. Each stage has a complete command; filenames alone are not launch commands.

| Stage and full command | Data | Initialization |
| --- | --- | --- |
| [1. Self-supervised learning](#step-1-self-supervised-learning) | Brightfield → same brightfield image | Add `--train_from_scratch`; omit `--resume_from` |
| [2. Supervised fine-tuning](#step-2-supervised-fine-tuning) | Brightfield → fluorescence | Omit `--train_from_scratch`; load SSL weights with `--resume_from` |
| [3. DPO alignment](#step-3-dpo-alignment) | Source + preferred/rejected targets | Load supervised weights with `--ref_model_path` |
| [Inference](#folder-inference) | Source images + prompts | Load a full checkpoint with `--ckpt_path` |

**Step 1 and Step 2 use the same Python training script.** The commands below use the same two-GPU launcher and shared hyperparameters. To start SSL, remove `--resume_from` and add `--train_from_scratch`; to start supervised fine-tuning, reverse those changes and supply the SSL checkpoint. Also change the dataset and output paths for each stage.

| Behavior in the supplied code | Step 1 | Step 2 (from a full SSL checkpoint) |
| --- | --- | --- |
| Skip connections | Disabled automatically | Enabled automatically |
| VAE encoder LoRA | Trainable | Trainable in the current implementation |
| Training forward call | Noise input, `deterministic=False`, `r=0.8` | `deterministic=True` (VAE sampling can still vary) |
| Starting weights | Pretrained SD-Turbo backbone, no experiment checkpoint | SSL checkpoint |

Although some console messages say the Step 2 encoder is frozen, `train_encoder_lora=True` and `set_train()` keep encoder LoRA trainable in this branch.

The `--train_from_scratch` name does **not** mean random initialization of the entire model. Both branches retain reconstruction and GAN training. The Step 1 settings below match Step 2 for comparison; they are not a record of historical SSL hyperparameters.

### Step 1: Self-supervised learning

Replace `/path/to/brightfield_ssl` with your SSL dataset. Select two available GPUs by changing `--gpu_ids 2,3` in both commands.

```bash
accelerate launch \
  --config_file configs/accelerate_2gpu.yaml \
  --multi_gpu --num_processes=2 --gpu_ids 2,3 \
  src/train_pretrained_pix2pix_turbo.py \
  --train_from_scratch \
  --dataset_folder /path/to/brightfield_ssl \
  --output_dir outputs/heparg_ssl \
  --resolution 512 \
  --train_batch_size 1 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler constant \
  --max_train_steps 10000 \
  --num_training_epochs 50 \
  --enable_xformers_memory_efficient_attention \
  --viz_freq 25 \
  --report_to None \
  --lambda_clipsim 0 \
  --eval_freq 4170 \
  --num_samples_eval 4170 \
  --gradient_checkpointing
```

`--train_from_scratch` automatically disables skip connections. Do not pass `--resume_from` for a new SSL run. Despite the flag's name, this branch still loads the pretrained SD-Turbo backbone and retains reconstruction and GAN losses.

For a two-step check, append `--debug_steps 2 --gradient_accumulation_steps 1` and use `--output_dir outputs/debug_ssl`. Use the direct command above: the Step 2 shell launcher requires a checkpoint.

### Step 2: Supervised fine-tuning

Load a Step 1 checkpoint and omit `--train_from_scratch` to enable skip connections. The example uses an existing SSL checkpoint; replace it with your own saved checkpoint when running the stages in sequence.

```bash
accelerate launch \
  --config_file configs/accelerate_2gpu.yaml \
  --multi_gpu --num_processes=2 --gpu_ids 2,3 \
  src/train_pretrained_pix2pix_turbo.py \
  --dataset_folder "$DATA_ROOT/260612_wetlab_fix/20x/decrease_experiment/decrease_experiment_2" \
  --output_dir outputs/experiment_2 \
  --resume_from "$MODEL_ROOT/heparg_ssl/checkpoints/full_checkpoint_35841.pt" \
  --resolution 512 \
  --train_batch_size 1 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler constant \
  --max_train_steps 10000 \
  --num_training_epochs 50 \
  --enable_xformers_memory_efficient_attention \
  --viz_freq 25 \
  --report_to None \
  --lambda_clipsim 0 \
  --eval_freq 4170 \
  --num_samples_eval 4170 \
  --gradient_checkpointing
```

**Optional checked launcher:** replace the `accelerate launch ... src/train_pretrained_pix2pix_turbo.py` prefix with `bash scripts/train.sh --gpu_ids 2,3`, keeping all subsequent training arguments. This wrapper checks data, checkpoint existence, and GPU kernels before starting the same entry point. It requires `--resume_from`, so use the direct Step 1 command for new SSL training. CLI values override wrapper defaults; extra training arguments are forwarded to Python. `--dataset` aliases `--dataset_folder`; xformers and gradient checkpointing are always enabled by the wrapper.

**Two-step check for either direct command:** append `--debug_steps 2 --gradient_accumulation_steps 1` and choose a separate debug output directory. For the Step 2 wrapper, use `scripts/debug.sh` instead of `scripts/train.sh`. Success prints `Training stopped successfully at step 2.` Debug runs skip evaluation and checkpoint writing.

The commands explicitly retain the Steps 1–2 defaults: learning rate `1e-5`, gradient accumulation `8`, 10,000 synchronized steps, and at most 50 epochs. Checkpoints are written to `<output_dir>/checkpoints/`. Loading the SSL full checkpoint for Step 2 starts a new fine-tuning run, without restoring the old optimizer or step count.

### Step 3: DPO alignment

Prepare same-name triplets and paired evaluation data:

```text
heparg_dpo/
  input/<name>.png          # Brightfield source
  good/<name>.png           # Preferred target
  bad/<name>.png            # Rejected target
  input_prompts.json
  test_A/<eval_name>.png
  test_B/<eval_name>.png
  test_prompts.json
```

Both JSON files use the filename-to-caption format above. Evaluation data is required and used during training.

```bash
CUDA_VISIBLE_DEVICES=2 python src/train_dpo_pix2pix_turbo.py \
  --train_method dpo \
  --dataset_folder "$DATA_ROOT/260612_wetlab_fix/20x/heparg_dpo" \
  --output_dir outputs/heparg_whole_dpo \
  --pretrained_model_name_or_path stabilityai/sd-turbo \
  --ref_model_path "$MODEL_ROOT/heparg_whole_260622/checkpoints/full_checkpoint_54321.pt" \
  --beta_dpo 0.02 \
  --dpo_latent_reward_weight 1.0 \
  --dpo_image_reward_weight 0.25 \
  --dpo_anchor_weight 0.05 \
  --learning_rate 5e-7 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 200 \
  --max_grad_norm 0.5 \
  --mixed_precision bf16 \
  --dataloader_num_workers 0 \
  --seed 42 \
  --num_training_epochs 50
```

This command preserves every DPO option and value in the original experiment command. The packaged entry point is renamed to `src/train_dpo_pix2pix_turbo.py`; machine-specific paths use `DATA_ROOT` / `MODEL_ROOT`, and outputs are local to this repository. `CUDA_VISIBLE_DEVICES=2` makes GPU selection explicit. `--num_training_epochs 50` makes the existing default run length explicit; change it for your experiment.

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--train_method` | `dpo` | Select preference training |
| `--pretrained_model_name_or_path` | `stabilityai/sd-turbo` | SD-Turbo model identifier |
| `--ref_model_path` | Your supervised checkpoint | Initialize policy and frozen reference |
| `--beta_dpo` | `0.02` | Scale the policy/reference preference logits |
| `--dpo_latent_reward_weight` | `1.0` | Weight latent-space preference distances |
| `--dpo_image_reward_weight` | `0.25` | Weight image-space preference distances |
| `--dpo_anchor_weight` | `0.05` | Weight the preferred-target anchor loss |
| `--learning_rate` | `5e-7` | Optimizer learning rate |
| `--train_batch_size` / `--gradient_accumulation_steps` | `1` / `16` | Per-device batch size and accumulation window |
| `--lr_scheduler` / `--lr_warmup_steps` | `constant_with_warmup` / `200` | Warmup followed by constant learning rate |
| `--max_grad_norm` | `0.5` | Gradient clipping threshold |
| `--mixed_precision` | `bf16` | BF16 mixed precision |
| `--dataloader_num_workers` / `--seed` | `0` / `42` | Data loading workers and random seed |
| `--num_training_epochs` | `50` | Epoch limit; reduce for a short run |

This is a single-GPU run requiring BF16 support. Replace `--ref_model_path` with your Step 2 checkpoint; it initializes both policy and frozen reference models. To resume DPO, additionally pass `--resume_from /path/to/dpo_checkpoint.pt`, keeping the original Step 2 reference.

**DPO run length:** the original loop does not stop at `--max_train_steps`. Set `--num_training_epochs` (default 50) to bound training. DPO does not support the Step 1–2 debug option or shell launchers.

DPO writes TensorBoard logs:

```bash
tensorboard --logdir outputs/heparg_whole_dpo/logs
```

## Folder inference

Supply a full checkpoint, source images, and a filename-to-prompt JSON file:

```bash
CUDA_VISIBLE_DEVICES=2 python src/inference_simple_folder.py \
  --ckpt_path "$MODEL_ROOT/heparg_decrease_experiment/experiment_2/checkpoints/full_checkpoint_29681.pt" \
  --input_dir "$DATA_ROOT/260204_wetlab_fix/20x/20x-bright/new_test_folder/test_A" \
  --json_path "$DATA_ROOT/260204_wetlab_fix/20x/20x-bright/new_test_folder/test_prompts.json" \
  --output_dir outputs/experiment2_inference_result \
  --resolution 512
```

Change `CUDA_VISIBLE_DEVICES` to select the GPU. The script processes JSON-listed images, skips missing files, and saves results with the same filenames. Use flat filenames and a separate output directory; existing same-name outputs are overwritten. Supported resolutions are 256 and 512.

Inference accepts full `full_scratch_v1` checkpoints or `model_state_dict` snapshots, with UNet/VAE LoRA ranks 8/4. It uses the supervised model, whose VAE latent sampling can produce different outputs between runs; it does not automatically use DPO's latent-mode implementation.

## Reference

- [Implementation notes](docs/IMPLEMENTATION_NOTES.md): losses, model behavior, checkpoint formats, launcher options, and environment details.
- [Validation record](docs/VALIDATION.md): completed checks and remaining limitations.
- CLI help: `bash scripts/train.sh --help` or `python src/<entry_point>.py --help`. DPO-specific options are listed above because its base help omits them.
- [Source manifest](SOURCE_MANIFEST.json): original filenames and hashes, retained for provenance.

Retain [LICENSE](LICENSE) and [CLIP's license](vendor/openai_clip/LICENSE). Model weights, third-party dependencies, and datasets have separate licenses; data and checkpoints are not included in this source release.

Repository visibility and anonymous review options: [reviewer access](docs/REVIEW_ACCESS.md).
