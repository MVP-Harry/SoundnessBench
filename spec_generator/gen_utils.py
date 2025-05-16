import os
import torch
import onnx

config = {
    'synthetic1d': {
        'timeout': 100,
        'mean': torch.tensor([0]),
        'std': torch.tensor([1]),
        'lower_limit': -0.1,
        'upper_limit': 0.1,
        'num_classes': 2
    },
    'synthetic2d': {
        'timeout': 100,
        'mean': torch.tensor([0]),
        'std': torch.tensor([1]),
        'lower_limit': -0.1,
        'upper_limit': 0.1,
        'num_classes': 2
    },
}

def generate_onnx(model, onnx_model_path, input_shape):
    device = next(model.parameters()).device
    model = model.cpu()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=13, #17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    model = model.to(device)
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print(f"Model successfully converted to {onnx_model_path}")


def save_model_and_data(model, data, output_path):
    """
    Generate index.txt to specify the data in benchmark,
    each line represents the index of the data in counterset.
    true_counterexamples: list[tuple(X, y, X_ori, y_ori, idx in counterset)]
    """
    input_shape = data[0][0].shape
    onnx_path = os.path.join(output_path, "model.onnx")
    if os.path.exists(f"{output_path}.optimized"):
        os.remove(f"{output_path}.optimized")
    generate_onnx(model, onnx_path, input_shape)
    torch.save(data, os.path.join(output_path, "data.pt"))


def create_input_bounds(img: torch.Tensor, eps: float,
                        mean: torch.Tensor, std: torch.Tensor,
                        lower_limit=0, upper_limit=1) -> torch.Tensor:
    device = img.device
    mean = mean.to(device).view(-1, 1, 1)
    std = std.to(device).view(-1, 1, 1)
    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32, device=device)
    bounds[..., 0] = (torch.clamp((img - eps), lower_limit, upper_limit) - mean) / std
    bounds[..., 1] = (torch.clamp((img + eps), lower_limit, upper_limit) - mean) / std
    return bounds.view(-1, 2)


def save_vnnlib(input_bounds: torch.Tensor, label: int,
                spec_path: str, total_output_class: int):
    if isinstance(label, torch.Tensor):
        label = label.item()
    
    with open(spec_path, "w") as f:

        f.write(f"; Property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")

        # disjunction version:
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))\n")
