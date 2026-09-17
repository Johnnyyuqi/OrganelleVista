# Validation record

## Packaging checks performed on 2026-09-16

The following checks used the original machine's `img2img-turbo` Conda environment:

- Python compilation and syntax checks for all three shell scripts passed.
- The training entry point's `--help` command passed and included `--debug_steps`.
- Imports passed for Torch, torchvision, Accelerate, transformers, diffusers, PEFT, LPIPS, CLIP, vision_aided_loss, clean-fid, xformers, and the local training modules.
- All A/B image paths for the 27,984 prompts in `decrease_experiment_2` existed. The first image pair decoded successfully.
- The actual SD-Turbo tokenizer loaded from the local cache. Reading a sample through the training dataset class produced A/B tensors of shape `(3, 512, 512)` and token IDs of shape `(1, 77)`. The deterministic split contained 25,186 training samples and 2,798 validation samples.
- The specified checkpoint existed and was 5,269,382,169 bytes. The full checkpoint was not loaded during these checks, so parameter compatibility was not verified.
- Running `bash scripts/debug.sh` stopped at GPU preflight with `CUDA unavailable`. `nvidia-smi` also could not communicate with the NVIDIA driver in that session. No long training run was started and no output was written to the original experiment directory.
- xformers metadata was readable, but its GPU forward and backward kernels could not run. Its build CUDA version differed from the Torch CUDA version; compatibility still needed GPU execution checks.

Code packaging, dependency imports, and real dataset reading passed. Two-GPU training, checkpoint loading, loss backpropagation, validation, and checkpoint writing were not verified end to end in that session. These are historical results, not a fresh assessment of current GPU availability.

After GPU access is available, run `bash scripts/debug.sh` with the dataset and checkpoint configured as described in the README. A short training run succeeds when it prints `Training stopped successfully at step 2.` and exits with code 0. Validation and checkpoint writing require separate verification during normal training.

## English documentation update on 2026-09-17

Translated the README, this validation record, and all Chinese source comments and docstrings into English. Added complete training and debugging commands, parameter-routing details, and examples of scalar overrides.

Validation passed:

- Python abstract syntax trees before and after translation matched after excluding docstrings, confirming unchanged executable logic.
- All Python source files parsed successfully, and all shell scripts passed `bash -n`.
- All eight shell examples in the README passed shell syntax checks.
- A scan of readable text files throughout the release found no remaining Chinese characters. Binary model weights and tokenizer vocabulary assets were left unchanged.

No training run was started for this documentation-only change.

## DPO packaging checks on 2026-09-17

Added `train_dpo_pix2pix_turbo_no_network.py`, `pretrain_pix2pix_turbo_no_network.py`, and `my_utils/dpo_utils.py` from the original project. Their executable Python ASTs match the originals after excluding docstrings; only comments/docstrings and line endings were changed. The shared `model.py` dependency was already included. TensorBoard 2.19.0 is now listed in `requirements.txt`.

Checks using the existing `img2img-turbo` environment passed:

- DPO entry point, model, utilities, and TensorBoard import successfully. All local modules resolve within the release directory.
- The complete DPO command in the README parses with the supplied preference weights, learning rate, gradient accumulation, and BF16 setting.
- All image paths exist for 3,000 training triplets in `input_prompts.json` and 4,160 evaluation pairs in `test_prompts.json`.
- The real tokenizer loads from the local cache. The first training triplet and evaluation pair load through the actual dataset classes, producing finite image tensors of shape `(3, 512, 512)` and token IDs of shape `(1, 77)`.
- The reference checkpoint exists and is 5,275,433,609 bytes. Its full weights were not loaded in this check.
- Source syntax, English-text checks for the added files, and README shell-example syntax checks passed.

`torch.cuda.is_available()` returned False in this session. No DPO training was launched; checkpoint compatibility, BF16 GPU execution, backpropagation, evaluation, and checkpoint writing remain unverified end to end. Existing datasets and checkpoints were read for checks but were not copied into the release.

The original DPO loop has no stopping condition for `max_train_steps`. This behavior was preserved and is documented in the README; the supervised entry point's short-debug controls do not apply to DPO.

## DPO model module rename

Renamed the release model module from `src/pretrain_pix2pix_turbo_no_network.py` to `src/pretrain_pix2pix_turbo_dpo.py` and updated the DPO training entry point's import. The model file contents are byte-for-byte unchanged. `SOURCE_MANIFEST.json` retains the original source filename and hash for provenance.

Verified with the existing `img2img-turbo` environment: the DPO entry point imports the renamed release module and exposes the same `Pix2Pix_Turbo` class object; all Python source files parse; no obsolete model import remains; and the DPO entry point's `--help` command exits successfully. The training entry point filename and documented training command are unchanged. No GPU training was launched for this rename.

## Training entry point renames

Renamed `src/train_pretrained_pix2pix_turbo_no_network.py` to `src/train_pretrained_pix2pix_turbo.py` and `src/train_dpo_pix2pix_turbo_no_network.py` to `src/train_dpo_pix2pix_turbo.py`. Both files retained identical contents during renaming. Updated `scripts/train.sh`, `scripts/preflight.py`, and README commands. Historical names in this record and `SOURCE_MANIFEST.json` identify the original source files.

Verification passed: both entry points import their intended release model classes; both new Python commands accept `--help`; all Python and shell syntax checks pass; stub execution confirms that both `train.sh` and `debug.sh` route to the renamed supervised entry; and README shell examples parse. The actual CPU preflight also passes, including the renamed module import, 27,984 paired image paths, first-pair decoding, and checkpoint existence. No GPU training was started for this rename.

## Folder inference packaging

Added `src/inference_simple_folder.py` from the original project. Its existing local dependencies (`pretrain_pix2pix_turbo.py`, `model.py`, and `my_utils/training_utils.py`) were already present. Replaced the hardcoded main-block paths with required CLI options, restricted the CLI resolution to the existing 256/512 transforms, and translated comments and console messages into English. The original inference function and model selection were otherwise retained. The original source hash is recorded in `SOURCE_MANIFEST.json`.

Checks passed in the existing `img2img-turbo` environment:

- The new entry point imports the supervised model from the release directory and its `--help` command succeeds.
- The complete original-machine README inference command parses correctly.
- The documented checkpoint exists. All 4,160 JSON-listed input image paths exist, and captions are strings.
- The first actual input image decodes and passes both the 256 and 512 resize transforms.
- Small synthetic checkpoints verify skip-convolution detection for a DDP-prefixed skip key and a full checkpoint without skip layers.
- Python syntax, English-text checks, and README shell-example syntax pass.

CUDA was unavailable during these checks. The real checkpoint was not loaded into a model, and no GPU inference or generated image output was verified. These checks establish packaging, argument handling, dependency resolution, and input preprocessing, not end-to-end model correctness.

## Setup documentation audit

Checked all third-party imports in `src/` against `requirements.txt` and bundled CLIP. The dependency list covers those imports, and DPO's TensorBoard requirement is explicitly pinned. All three entry points' `--help` commands pass in the existing environment; README and implementation-note shell examples pass `bash -n`.

Rewrote Setup to create a new `organellevista` Conda environment with Python 3.10 before activation and dependency installation. New-install instructions explicitly select the Torch 2.0.1 / torchvision 0.15.2 CUDA 11.8 wheels to match the inspected xformers 0.0.22 extension build. Existing `img2img-turbo` users can keep their environment; no installed packages were changed.

The historical environment fails `pip check` because its opencv-python-headless 4.13.0.92 requires NumPy >=2, while NumPy 1.26.4 is installed. This conflict is documented instead of claiming a fully validated environment. A fresh Conda environment was not installed in this audit, and the proposed CUDA 11.8 recipe has not been tested end to end. GPU kernel and training validation remain outstanding.
