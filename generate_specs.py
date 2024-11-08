import os
import torch
import onnx
import shutil
import csv
from utils import get_model


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


def convert_pt_to_onnx(model_arch, pt_model_path, onnx_model_path, input_shape):
    model = get_model(model_arch, None)
    model.load_state_dict(torch.load(pt_model_path)["state_dict"])
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    print(f"Model successfully converted to {onnx_model_path}")


def save_model_and_data(data, fname, result_path, ckpt_type, model_arch, input_shape):
    """
    Generate index.txt to specify the data in benchmark,
    each line represents the index of the data in counterset.
    true_counterexamples: list[tuple(X, y, X_ori, y_ori, idx in counterset)]
    """

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Created directory: {result_path}")
    else:
        print(f"Directory already exists: {result_path}")

    if ckpt_type == 'pt':
        print("Converting pt to onnx and saving.")
        convert_pt_to_onnx(model_arch, f'{fname}.{ckpt_type}', f'{result_path}/model.onnx', input_shape)
    ckpt_path = f'{result_path}/model.{ckpt_type}'
    print(f"Copied ckpt {fname}.{ckpt_type} to {os.path.abspath(ckpt_path)}")
    shutil.copy(f'{fname}.{ckpt_type}', ckpt_path)

    # Each item in data is a tuple like (original_x, original_y, counter_x).
    # counter_x is a hidden counterexample, and it can be None for normal examples
    # without a hidden counterexample.
    data_path = f'{result_path}/data.pt'
    torch.save(data, data_path)


def create_input_bounds(img: torch.Tensor, eps: float,
                        mean: torch.Tensor, std: torch.Tensor,
                        lower_limit=0, upper_limit=1) -> torch.Tensor:
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = (torch.clamp((img - eps), lower_limit, upper_limit) - mean) / std
    bounds[..., 1] = (torch.clamp((img + eps), lower_limit, upper_limit) - mean) / std
    return bounds.view(-1, 2)


def save_vnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int):
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


def gen_properties(data, ckpt_type, epsilon, dataset, model_path):
    if not os.path.exists(f'{model_path}/vnnlib'):
        os.makedirs(f'{model_path}/vnnlib')

    instances = []
    dataset_config = config[dataset]
    for i in range(len(data)):
        vnnlib_path = f'vnnlib/{i}.vnnlib'
        x, y = data[i][:2]
        input_bounds = create_input_bounds(
            x, epsilon, dataset_config['mean'], dataset_config['std'],
            dataset_config['lower_limit'], dataset_config['upper_limit'])
        save_vnnlib(input_bounds, y, os.path.join(model_path, vnnlib_path),
                    total_output_class=dataset_config['num_classes'])
        if ckpt_type == 'onnx':
            instances.append((
                'model.onnx',
                vnnlib_path,
                dataset_config['timeout']
            ))
        else:
            instances.append((vnnlib_path,))

    instance_path = f'{model_path}/instances.csv'
    with open(instance_path, 'w') as f:
        csv.writer(f).writerows(instances)
    print('Saving instance.csv to', os.path.abspath(instance_path))