# Implementation notes

Detailed behavior of the packaged code. See the [README](../README.md) for commands and [VALIDATION.md](VALIDATION.md) for verification results.

## Environment and pretrained weights

The observed environment uses Python 3.10, Torch 2.0.1+cu117, torchvision 0.15.2+cu117, diffusers 0.25.1, transformers 4.35.2, and Accelerate 1.1.1. `requirements.txt` records relevant dependency versions. `docs/environment-observed.txt` contains the full environment version snapshot without local installation URLs.

To install the recorded dependencies into an appropriate Python environment, run this command from the release root:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 -r requirements.txt
```

This installation recipe has not been verified in a fresh environment. The observed xformers 0.0.22 build targets Torch 2.0.1/CUDA 11.8, while the observed Torch installation uses CUDA 11.7. Successful imports alone do not establish GPU-kernel compatibility. The launcher checks xformers forward and backward execution on both selected GPUs before training. On a new machine, use an xformers build compatible with its Torch/CUDA installation. The environment snapshot is not a cross-platform lockfile.

The training code loads `stabilityai/sd-turbo`, CLIP ViT-B/32, and LPIPS/VGG weights. First use requires downloads or existing caches. `--lambda_clipsim 0` does not remove CLIP from validation or from the GAN discriminator.

For offline operation, prepare all required caches first. Setting `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` disables Hugging Face network access, but does not control CLIP or torchvision downloads.

## Dataset format and experiment behavior

```text
dataset/
  train_A/<name>.png
  train_B/<name>.png
  train_prompts.json
```

The prompt file maps paired image filenames to text captions:

```json
{
  "sample_001.png": "fluorescence staining description",
  "sample_002.png": "fluorescence staining description"
}
```

Paired files in `train_A` and `train_B` must have matching names. The entry point deterministically splits the training dataset into approximately 90% training and 10% validation, without reading the test split. It writes the split manifest to the output directory. The default image transform is `resize_512`; changing `--resolution` alone does not change that transform.

For the full checkpoint used in the example, loading starts a new fine-tuning run: it loads the model, reinitializes skip convolutions, starts the step count at zero, and does not restore the old optimizer. This preserves the original experiment behavior.

The original code computes L1 and structural losses but does not add them to the reconstruction loss used for backpropagation. `viz_freq` does not control a separate visualization loop; validation writes visualization images. These behaviors have not been changed in this release.

## Changes from the original project

- Added `--debug_steps N` to stop after N new synchronized training steps while skipping validation and checkpoint writing.
- Added a stopping condition for `max_train_steps`, which previously affected the progress bar without stopping training.
- Bundled the configuration and launch scripts, with normal training output directed to this release directory by default.
- Added command-line parsing for dataset, checkpoint, GPU, output-directory, and port options.
- Translated comments and docstrings into English without changing executable Python logic in the translation pass.
- Normalized source line endings to LF. `SOURCE_MANIFEST.json` records the original source hashes, not hashes of the modified release files.

The source project outside this release directory has not been modified.

## Step 1 details

`--train_from_scratch` still loads pretrained SD-Turbo VAE, UNet, and text-encoder weights. It initializes trainable LoRA modules and disables encoder-to-decoder skip connections. Training uses `deterministic=False` and the latent mixture `0.8 * z + 0.2 * epsilon`, rather than a configurable `z + sigma_s * epsilon` schedule. The shared loop retains MSE/LPIPS reconstruction and GAN losses. The methods description should reflect this implementation.

## DPO details

DPO uses `pretrain_pix2pix_turbo_dpo.py`, with latent-mode encoding and checkpointed decoding. Its loss combines weighted latent/image preference scores relative to a frozen reference and a chosen-target anchor. It still loads LPIPS and CLIP for evaluation.

`max_train_steps` configures the schedule and progress bar but does not stop the original DPO loop. `num_training_epochs` (default 50) bounds the run. Checkpointing defaults to every 560 synchronized steps, evaluation every 1920 steps, and evaluation samples to 192. Both checkpointing and evaluation also run at step 1 (`step % frequency == 1`). DPO does not support `--debug_steps`. Its base-parser help omits the separately parsed DPO options shown in the README.

If `input_prompts.json` is absent, the triplet loader enumerates input images and uses empty captions. Paired evaluation data and `test_prompts.json` are always required. The supplied test split is used during training; keep separate held-out data for final reporting if evaluation results guide model selection. Multi-GPU DPO has not been validated.

## Inference details

The loader accepts `model_state_dict` snapshots or checkpoints with `model.format == "full_scratch_v1"` and `model.full_state_dict`. Lightweight LoRA-only checkpoints are unsupported. Although skip detection recognizes a top-level `state_dict`, the model loader does not accept that format alone.

Skip connections are detected from checkpoint keys. LoRA ranks are fixed at UNet 8 and VAE 4; different ranks require matching model configuration. The script uses the supervised model's latent sampling, even with `deterministic=True`, so outputs may vary across runs. It does not automatically select DPO's latent-mode implementation.

## Launcher options

`scripts/train.sh` checks dependencies, paired-image paths, checkpoint existence, and CUDA/xformers on both selected GPUs before launch. `scripts/debug.sh` uses the same launcher with two debug steps and gradient accumulation 1. Neither script starts SSL or DPO.

| CLI option | Environment variable | Default |
| --- | --- | --- |
| `--dataset_folder` / `--dataset` | `DATASET_DIR` | Required |
| `--resume_from` | `RESUME_FROM` | Required |
| `--output_dir` | `OUTPUT_DIR` | `outputs/experiment_2` |
| `--gpu_ids` | `GPU_IDS` | `2,3` |
| `--main_process_port` | `MASTER_PORT` | `29501` |

CLI options override environment variables. `PYTHON_BIN` selects the Python executable (default `python`). The existing `scripts/local.example.sh` supplies original-machine paths as an alternative to explicit CLI arguments. Relative launcher paths are resolved from the release root.

## Source provenance

The training entry points no longer have `_no_network` suffixes. The DPO model was renamed to `pretrain_pix2pix_turbo_dpo.py`. `SOURCE_MANIFEST.json` retains original source filenames and hashes for provenance, not current release-file hashes. The original project outside this release was not modified.
