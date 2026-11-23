"""Evaluate saved CIFAR-10 weights without training or downloading pretrained weights."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import timm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from train_utils import evaluate_classifier, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--model", default="fastvit_t8")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("build/checkpoint-test.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model = timm.create_model(args.model, pretrained=False, num_classes=10).to(device)
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)
    dataset = CIFAR10(root=args.data_root, train=False, download=args.download, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    record = {
        "evaluated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": "Full official CIFAR-10 test split; deterministic timm evaluation transform; no training",
        "historical_provenance": "Checkpoint training and historical test identity are not established by this evaluation",
        "model": args.model, "data_config": config, "transform": str(transform),
        "device": str(device), "platform": platform.platform(), "python": platform.python_version(),
        "versions": {x: importlib.metadata.version(x) for x in ["torch", "torchvision", "timm", "numpy", "Pillow"]},
        "command": sys.argv, "batch_size": args.batch_size, "num_workers": args.num_workers,
        "test_files_md5": dict(CIFAR10.test_list), "results": [],
    }
    for checkpoint in args.checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        start = time.monotonic()
        result = evaluate_classifier(model, loader, device)
        result.update(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      elapsed_seconds=time.monotonic() - start)
        record["results"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{checkpoint.name}: {result['correct']}/{result['sample_count']} = {result['accuracy']:.4%}", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
