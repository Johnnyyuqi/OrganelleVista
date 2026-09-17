# OrganelleVista

Train a virtual-staining model in three stages, then generate images from a folder.

| Stage | Data | Entry point |
| --- | --- | --- |
| 1. Self-supervised learning | Brightfield → same brightfield image | `src/train_pretrained_pix2pix_turbo.py --train_from_scratch` |
| 2. Supervised fine-tuning | Brightfield → fluorescence | `scripts/train.sh` |
| 3. DPO alignment | Source + preferred/rejected targets | `src/train_dpo_pix2pix_turbo.py` |
| Inference | Source images + prompts | `src/inference_simple_folder.py` |

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

This setup targets Python 3.10, Torch 2.0.1 / torchvision 0.15.2 with CUDA 11.8, and xformers 0.0.22. The CUDA selection follows [PyTorch's version-specific installation instructions](https://pytorch.org/get-started/previous-versions/#v201) and matches the inspected xformers build. It is a proposed clean-install configuration, **not yet a verified fresh-environment or GPU training result**. Dependencies for all stages, including local OpenAI CLIP and TensorBoard, are listed in [requirements.txt](requirements.txt); run installation from the repository root.

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

## Step 1: Self-supervised learning

Replace `/path/to/brightfield_ssl` with your SSL dataset. This example follows the Step 2 settings; it is not a record of historical SSL hyperparameters.

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

## Step 2: Supervised fine-tuning

Load a Step 1 checkpoint and omit `--train_from_scratch` to enable skip connections. The example uses an existing SSL checkpoint; replace it with your own saved checkpoint when running the stages in sequence.

```bash
bash scripts/train.sh \
  --dataset_folder "$DATA_ROOT/260612_wetlab_fix/20x/decrease_experiment/decrease_experiment_2" \
  --output_dir outputs/experiment_2 \
  --gpu_ids 2,3 \
  --resume_from "$MODEL_ROOT/heparg_ssl/checkpoints/full_checkpoint_35841.pt" \
  --resolution 512 \
  --train_batch_size 1 \
  --enable_xformers_memory_efficient_attention \
  --viz_freq 25 \
  --report_to None \
  --lambda_clipsim 0 \
  --eval_freq 4170 \
  --num_samples_eval 4170 \
  --gradient_checkpointing
```

The launcher checks prerequisites and starts two GPU processes. CLI values override its defaults; extra training arguments are forwarded to Python. For example, append `--learning_rate 1e-5 --max_train_steps 20000`. `--dataset` is an alias for `--dataset_folder`. Both xformers and gradient checkpointing are enabled by the launcher even if omitted from the command.

For a two-step check, replace `scripts/train.sh` with `scripts/debug.sh` and use `--output_dir outputs/debug_experiment_2`. Success prints `Training stopped successfully at step 2.` The debug run skips evaluation and checkpoint writing.

Steps 1–2 default to learning rate `1e-5`, gradient accumulation `8`, 10,000 synchronized steps, and at most 50 epochs. Checkpoints are written to `<output_dir>/checkpoints/`. Loading the SSL full checkpoint for Step 2 starts a new fine-tuning run, without restoring the old optimizer or step count.

## Step 3: DPO alignment

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
  --seed 42
```

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
- [Validation record](docs/VALIDATION.md): completed checks and unverified GPU paths.
- CLI help: `bash scripts/train.sh --help` or `python src/<entry_point>.py --help`. DPO-specific options are listed above because its base help omits them.
- [Source manifest](SOURCE_MANIFEST.json): original filenames and hashes, retained for provenance.

Retain [LICENSE](LICENSE) and [CLIP's license](vendor/openai_clip/LICENSE). Model weights, third-party dependencies, and datasets have separate licenses; data and checkpoints are not included in this source release.

Private-repository setup and anonymous review options: [reviewer access](docs/REVIEW_ACCESS.md).
