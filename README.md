# *SoundnessBench*: A Comprehensive Benchmark to Evaluate the Soundness of Neural Network Verifiers
![Verification flow](/assets/flow.png)

## Overview

This repository contains the source code of SoundnessBench, a benchmark designed to thoroughly evaluate the soundness of neural network (NN) verifiers. It addresses the critical lack of ground-truth labels in previous benchmarks for NN verifiers by providing numerous unverifiable instances with hidden counterexamples, enabling it to effectively reveal internal bugs in NN verifiers that falsely claim verifiability on those instances. SoundnessBench aims to support developers in evaluating and improving NN verifiers. See our paper for more details.

## Download SoundnessBench
SoundnessBench is hosted on [HuggingFace](https://huggingface.co/datasets/SoundnessBench/SoundnessBench). To directly download the benchmark, use
```bash
git clone https://huggingface.co/datasets/SoundnessBench/SoundnessBench
```

The downloaded benchmark should contain a total of 26 models across 9 distinct NN architectures with different input sizes and perturbation radii. The table below shows the 9 architectures.

| Name       | Model Architecture                                    | Activation Function |
| ---------- | ---------------------------------------------------- | -------------------- |
| CNN 1 Conv  | Conv 10 × 3 × 3, FC 1000, FC 100, FC 20, FC 2     | ReLU                 |
| CNN 2 Conv  | Conv 5 × 3 × 3, Conv 10 × 3 × 3, FC 1000, FC 100, FC 20, FC 2 | ReLU |
| CNN 3 Conv  | Conv 5 × 3 × 3, Conv 10 × 3 × 3, Conv 20 × 3 × 3, FC 1000, FC 100, FC 20, FC 2 | ReLU |
| CNN AvgPool | Conv 10 × 3 × 3, AvgPool 3 × 3, FC 1000, FC 100, FC 20, FC 2 | ReLU |
| MLP 4 Hidden| FC 100, FC 1000, FC 1000, FC 1000, FC 20, FC 2 | ReLU |
| MLP 5 Hidden| FC 100, FC 1000, FC 1000, FC 1000, FC 1000, FC 20, FC 2 | ReLU |
| CNN Tanh    | Conv 10 × 3 × 3, FC 1000, FC 100, FC 20, FC 2     | Tanh                 |
| CNN Sigmoid    | Conv 10 × 3 × 3, FC 1000, FC 100, FC 20, FC 2     | Sigmoid                 |
| VIT        | Modified VIT [17] with patch size 1 × 1, 2 attention heads and embedding size 16 | ReLU |

To download the benchmark and convert it to the format following [VNN-COMP](https://sites.google.com/view/vnn2024) and [their benchmarks](https://github.com/ChristopherBrix/vnncomp2024_benchmarks), you can use the following command:
```bash
python convert_from_hf.py --output_dir OUTPUT_DIR
```
`OUTPUT_DIR` will be the folder to store the converted benchmark. It is `SoundnessBench` by default.

Each subfolder should contain:
* `model.onnx`: Model in ONNX format with both model architecture and parameters
* `vnnlib/`: A folder of instances in [VNN-LIB](https://www.vnnlib.org/) format
* `instances.csv`: A list of [VNN-LIB](https://www.vnnlib.org/) files

## Basic Usage -- Evaluate Your Verifier

You may run your verifier on our benchmark to produce a `results.csv` file containing the results.
The format of this CSV file should follow [the format in VNN-COMP](https://github.com/ChristopherBrix/vnncomp2024_benchmarks/blob/main/run_single_instance.sh#L104), where each line contains the result on one instance.
This CSV file can be automatically produced if you use the [official script](https://github.com/ChristopherBrix/vnncomp2024_benchmarks/blob/main/run_all_categories.sh) from VNN-COMP to run your verifier, with [three scripts provided by the verifier](https://github.com/stanleybak/vnncomp2021?tab=readme-ov-file#scripts).
See an [example](https://github.com/ChristopherBrix/vnncomp2024_results/blob/main/alpha_beta_crown/2024_lsnc/results.csv) of the result file.

We provide an evaluation script to quickly count the results in a results CSV file.
Our script reports the following metrics:
* `clean_instance_verified_ratio`
* `clean_instance_falsified_ratio`
* `unverifiable_instance_verified_ratio`
* `unverifiable_instance_falsified_ratio`

A non-zero value for `unverifiable_instance_verified_ratio` indicates unsound results by the verifier.

Run the evaluation script by:
```bash
python verifiers/eval.py --model_dir <model_dir> --csv_file <csv_file>
```
* `<model_dir>` is the path to the folder containing `model.onnx`, `vnnlib/`, and `instances.csv` as described above, like `SoundnessBench/synthetic_cnn_1_conv_ch3_eps0.2/`.

* `<csv_file>` is optionally the path to the CSV file containing the results of your verifier. If not provided, the script will look for any `.csv` file of the existing verifiers in the `<model_dir>/results` folder.

## Run Existing Verifiers on *SoundnessBench*

In this section, we provide steps for running several verifiers included in our paper.

### Step 1: Install Verifiers

You can install the verification environments for **alpha-beta-CROWN**, **NeuralSAT**, **PyRAT** and **Marabou** using the following command:
```bash
bash install.sh
```
This will install all supported verifiers by default.

If you want to install a subset of verifiers, specify them using the `--verifier` option:
```bash
bash install.sh --verifier <verifier_name_1>,<verifier_name_2> ...
```
Valid `<verifier_name>` can be one of the following: `abcrown`, `neuralsat`, `pyrat` or `marabou_vnncomp_2023` and `marabou_vnncomp_2024`. We use *Conda* (e.g., [miniconda](https://docs.anaconda.com/miniconda/)) to manage environments for PyRAT, alpha-beta-CROWN, and NeuralSAT, and *Docker* to install and run Marabou.

- To enable [Gurobi](https://www.gurobi.com/) for alpha-beta-CROWN and NeuralSAT, you may be able to obtain a [free academic license](https://portal.gurobi.com/iam/licenses/request/?type=academic) and install Gurobi following the [instructions](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer#section:Installation). After installing Gurobi, you may need to set up the environment variables in `~/.bashrc` as follows and then source it by running `source ~/.bashrc`:

    ```bash
    export GUROBI_HOME="/path/to/gurobi"
    export PATH="${GUROBI_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
    ```

- To enable Gurobi for Marabou, copy your Web License file (gurobi.lic) into `verifiers/marabou_setup/` before running the installation script. The installation script will automatically copy the license file to the appropriate location in the Docker container.

<!-- - Please note that the installation of Marabou may fail on AMD CPUs since AMD CPUs do not support AVX-512 which is required to build OpenBLAS in Marabou. -->


### Step 2: Run Verification

You can run the verifiers using the `run_all_verification.py` script as follows:
```bash
python verifiers/run_all_verification.py --model_dir <model_dir> --verifier <verifier_name_1> <verifier_name_2> ...
```
The `--model_dir` argument should point to the directory where the models are stored.

Valid `<verifier_name>` can be one of the following:
- `abcrown_act`: alpha-beta-CROWN with activation split
- `abcrown_input`: alpha-beta-CROWN with input split
- `neuralsat_act`: NeuralSAT with activation split
- `neuralsat_input`: NeuralSAT with input split
- `pyrat`: PyRAT
- `marabou_vnncomp_2023`: Marabou with vnncomp 2023 version
- `marabou_vnncomp_2024`: Marabou with vnncomp 2024 version

## Train New Models

We provide an easy pipeline for users to train new models that contain hidden adversarial examples.

### Step 1: Install Dependencies
```bash
conda create -n "SoundnessBench" python=3.11
conda activate SoundnessBench
pip install -r requirements.txt
```

### Step 2: Train New Models
To train a new model, you can run the following command, which will first generate the synthetic dataset used for training, and then apply a two-objective training framework (see our paper for details) to the model.
```bash
python adv_training.py --output_dir OUTPUT_DIR --dataset DATASET --model MODEL_NAME --epsilon EPSILON
```

* For `OUTPUT_DIR`, we will use a folder to store the log, model checkpoint,
and generated specifications for verification. It is `SoundnessBench_raw` by default. And models will be saved in its subfolders according to `MODEL_NAME` and `EPSILON`.

* For `DATASET`, it can be `synthetic1d` or `synthetic2d` for 1D or 2D synthetic datasets, respectively.
The 1D dataset is used to train MLP models, while the 2D dataset is used to train CNN and ViT models.

* For `MODEL_NAME`, choose a class or function from [`models`](./models).
For example, `synthetic_mlp_4_hidden_ch1`.
Or it may contain arguments, e.g., `ViT(in_ch=1, patch_size=7, img_size=28)`.

* For `EPSILON`, it is the perturbation radius for the training dataset.

* Some other hyperparameters that may be tuned:
`--auto_alpha` (0.1 by default; note that the actual value for alpha is relative to epsilon, i.e., actual alpha = auto_alpha * epsilon),
`--lr` (0.001 by default).

* Please refer to [`adv_training.py`](./adv_training.py) for more details on the arguments.

Example:
```bash
python adv_training.py --output_dir SoundnessBench_raw --model synthetic_mlp_4_hidden_ch1 --epsilon 0.2 --auto_alpha 0.1 --lr 0.001
```

### Step 3: Evaluate the Trained Model and Generate Specifications

After the training has completed, simpliy **add `--gen`** at the end of the training command to run stronger attacks (i.e., with more restarts) and generate specifications for verification.
