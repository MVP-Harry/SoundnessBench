import os
import csv
import torch
import argparse
from datasets import load_dataset
from utils import get_model

from spec_generator.gen_utils import generate_onnx

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='SoundnessBench', type=str)
args = parser.parse_args()
output_dir = args.output_dir

# # Load from HF Hub
dataset = load_dataset("SoundnessBench/SoundnessBench")

os.makedirs(output_dir, exist_ok=True)

for split, split_dataset in dataset.items():
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path, exist_ok=True)

    model_name = split_dataset[0]["model_name"]
    model = get_model(model_name)
    total_params = sum([v.numel() for k, v in model.state_dict().items()])

    # Recover weights from flattened vector
    model_weights_flat = torch.tensor(split_dataset[0]["model_weights"])
    assert model_weights_flat.numel() == total_params, "Mismatch in parameter count"

    offset = 0
    for k, v in sorted(model.state_dict().items()):
        param_size = v.numel()
        recovered_param = model_weights_flat[offset:offset + param_size].reshape(v.shape)
        v.copy_(recovered_param)
        offset += param_size

    # Save onnx
    generate_onnx(model, os.path.join(split_path, "model.onnx"), split_dataset[0]["input_shape"])

    vnnlib_path = os.path.join(split_path, "vnnlib")
    os.makedirs(vnnlib_path, exist_ok=True)

    # Save vnnlib and instances.csv
    rows = []
    for i, item in enumerate(split_dataset):
        vnnlib = item["vnnlib"]
        timeout = item["timeout"]
        vnnlib_name = f"{i}.vnnlib"
        with open(os.path.join(vnnlib_path, vnnlib_name), "w") as f:
            f.write(vnnlib)
        rows.append(["model.onnx", f"vnnlib/{vnnlib_name}", timeout])

    with open(os.path.join(split_path, "instances.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
