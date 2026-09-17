"""Check imports, paired images, checkpoint and CUDA without starting training."""
import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
p = argparse.ArgumentParser()
p.add_argument('--dataset_folder', required=True, type=Path)
p.add_argument('--resume_from', required=True, type=Path)
p.add_argument('--gpu_ids', default='2,3')
p.add_argument('--cpu_only', action='store_true', help='Validate CPU prerequisites only; does not certify training.')
a = p.parse_args()
errors = []
for module in ['torch', 'torchvision', 'accelerate', 'transformers', 'diffusers', 'peft', 'lpips', 'clip', 'vision_aided_loss', 'cleanfid', 'xformers.ops', 'train_pretrained_pix2pix_turbo']:
    try:
        importlib.import_module(module)
        print('OK import', module)
    except Exception as exc:
        errors.append(f'{module}: {exc}')
try:
    from PIL import Image
    prompts = json.loads((a.dataset_folder / 'train_prompts.json').read_text())
    if not isinstance(prompts, dict) or len(prompts) < 2:
        raise ValueError('train_prompts.json must map at least two filenames to captions')
    for name, caption in prompts.items():
        if not isinstance(caption, str):
            raise ValueError(f'Caption must be text: {name}')
        for split in ['train_A', 'train_B']:
            path = a.dataset_folder / split / name
            if not path.is_file():
                raise FileNotFoundError(path)
    for split in ['train_A', 'train_B']:
        with Image.open(a.dataset_folder / split / next(iter(prompts))) as im:
            im.convert('RGB').resize((512, 512)).load()
    print(f'OK {len(prompts)} image pairs; first pair decoded')
except Exception as exc:
    errors.append(f'dataset: {exc}')
if not a.resume_from.is_file():
    errors.append(f'checkpoint does not exist: {a.resume_from}')
else:
    print(f'OK checkpoint exists ({a.resume_from.stat().st_size} bytes); weights not loaded')
if not a.cpu_only:
    try:
        import torch
        from xformers.ops import memory_efficient_attention
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; check NVIDIA driver and GPU access')
        ids = [int(i) for i in a.gpu_ids.split(',')]
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError('Provide exactly two distinct GPU IDs')
        for device in ids:
            with torch.cuda.device(device):
                q = torch.randn(1, 64, 8, 64, device=f'cuda:{device}', requires_grad=True)
                memory_efficient_attention(q, q, q).sum().backward()
                torch.cuda.synchronize()
                print('OK CUDA/xformers forward+backward', device, torch.cuda.get_device_name(device))
    except Exception as exc:
        errors.append(f'GPU: {exc}')
for error in errors:
    print('FAIL', error, file=sys.stderr)
if errors:
    sys.exit(1)
print('CPU prerequisites passed (GPU training unverified).' if a.cpu_only else 'Preflight passed; launch training for end-to-end validation.')
