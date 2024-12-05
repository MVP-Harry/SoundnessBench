# *SoundnessBench*: A Comprehensive Benchmark to Evaluate the Soundness of Neural Network Verifiers
![Verification flow](/assets/flow.png)

## Overview

This repository contains the source code of SoundnessBench, a benchmark designed to thoroughly evaluate the soundness of neural network (NN) verifiers. It addresses the critical lack of ground-truth labels in previous benchmarks for NN verifiers by providing numerous unverifiable instances with hidden counterexamples, so that it can effectively reveal internal bugs of NN verifiers if they falsely claim verifiability on those unverifiable instances. SoundnessBench aims to support developers in evaluating and improving NN verifiers. See our paper for more details.

LINK TO PAPER

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

Each folder should contain:
* `model.onnx`: Model in ONNX format with both model architecture and parameters
* `vnnlib/`: A folder of instances in [VNN-LIB](https://www.vnnlib.org/) format
* `instances.csv`: A list of [VNN-LIB](https://www.vnnlib.org/) files
* `model.pt`: Model checkpoint in PyTorch format with parameters only (not needed for verification)
* `data.pt`: Raw data with instances (not needed for verification)

The format of our benchmarks follows [VNN-COMP](https://sites.google.com/view/vnn2024) and [their benchmarks](https://github.com/ChristopherBrix/vnncomp2024_benchmarks). 

## Basic Usage

You may run your verifier on our benchmark to produce a `results.csv` file containing the results.
The format of this CSV file should follow [the format in VNN-COMP](https://github.com/ChristopherBrix/vnncomp2024_benchmarks/blob/main/run_single_instance.sh#L104), where each line contains the result on one instance. 
This CSV file can be automatically produced if you use the [official script](https://github.com/ChristopherBrix/vnncomp2024_benchmarks/blob/main/run_all_categories.sh) from VNN-COMP to run your verifier, with [three scripts provided by the verifier](https://github.com/stanleybak/vnncomp2021?tab=readme-ov-file#scripts).
See an [example](https://github.com/ChristopherBrix/vnncomp2024_results/blob/main/alpha_beta_crown/2024_lsnc/results.csv) of the result file.

We provide an evaluation script to quickly count the results in a results CSV file.
Our script would report metrics including:
* `clean_instance_verified_ratio`
* `clean_instance_falsified_ratio`
* `unverifiable_instance_verified_ratio`
* `unverifiable_instance_falsified_ratio` 

A non-zero value for `unverifiable_instance_verified_ratio` indicates unsound results by the verifier. 

Run the evaluation script by:
```bash
python eval.py <path to results.csv> <path to metrics.csv>
```

## Run Existing Verifiers on *SoundnessBench*

In this section, we provide steps for running several verifiers included in our paper. 
Our experiments require `conda` (e.g., [miniconda](https://docs.anaconda.com/miniconda/)).

### Step 1: Install Verifiers

For verification with NeuralSAT and Marabou on certain networks, [Gurobi](https://www.gurobi.com/) is needed, and please obtain a license. You may be able to obtain a [free academic license](https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2Fiam%2Flicenses%2Frequest%2F%3Ftype%3Dacademic).

You can install **alpha-beta-CROWN**, **NeuralSAT**, and **PyRAT** using the following command:
```bash
bash install.sh
```

Running **Marabou** is a bit more complex. Our experiments use **Docker** to install Marabou. 
Please set up a Docker environment and mount [marabou_vnncomp_2023](https://github.com/wu-haoze/Marabou/tree/vnn-comp-23) or [marabou_vnncomp_2024](https://github.com/wu-haoze/Marabou/tree/vnn-comp-24) in your container. After that, run the following to install Marabou:

```bash
conda create -y --name marabou python=3.8
conda activate marabou
cd ~/marabou/vnncomp
bash install_tool.sh
cp -r ~/marabou/opt ~/
```

### Step 2: Run Verification

#### 2.1: Run alpha-beta-CROWN

We provide the config file used in our experiments for alpha-beta-crown [here](./config.yaml).

To run **alpha-beta-CROWN** with activation split:

```bash
python run_all_verification.py --verifier abcrown --model_folder <path_to_soundness_bench> --config_dir <path_to_config_file>
```

To run **alpha-beta-CROWN** with input split:

```bash
python run_all_verification.py --verifier abcrown --model_folder <path_to_soundness_bench> --split_type input
```

#### 2.2: Run NeuralSAT

To run **NeuralSAT** with activation split:

```bash
python run_all_verification.py --verifier neuralsat --model_folder <path_to_soundness_bench>
```

To run **NeuralSAT** with input split:

```bash
python run_all_verification.py --verifier neuralsat --model_folder <path_to_soundness_bench> --split_type input
```

#### 2.3: Run PyRAT

To run **PyRAT**:

```bash
python run_all_verification.py --verifier pyrat --model_folder <path_to_soundness_bench>
```

#### 2.4: Run Marabou

We use **Docker** to run **Marabou**. Please specify the container name in the script. If you want to run a specific version of **Marabou**, just specify the name of the container corresponding to its version.

```bash
# Replace <marabou_container> with your Marabou container.
python run_all_verification.py --verifier marabou --model_folder <path_to_soundness_bench> --container_name <marabou_container>
```

### Step 3: Count results in results.csv

The `run_all_verification.py` script generates `<toolname>_results.csv` files in the `./results` directory. You can use the `eval.py` script to count the results in these files.

```bash
python eval.py <path to results.csv> <path to metrics.csv>
```

## Train New Models

We provide an easy pipeline for users to train new models that contain hidden adversarial examples.

### File Usage Overview

* `synthetic_data_generation.py` generates synthetic data with pre-defined hidden counterexamples
* `adv_training.py` trains models on the synthetically generated dataset using a two-objective training framework
* `cross_attack_evaluation.py` uses very strong adversarial attacks to make sure that no trivial counterexamples can be found and saves those instances into VNNLIB files
* `models` contain the definitions of all NN architectures we use in SoundnessBench

### Install Dependencies
```
conda create -n "SoundnessBench" python=3.11
conda activate SoundnessBench
pip install -r requirements.txt
```

### Detailed Walkthrough
To train a new model, you can run the following command, which will first call `synthetic_data_generation.py` to generate the synthetic dataset used for training, and then apply a two-objective training framework (see our paper for details) to the model.
```bash
python adv_training.py
                       # GENERAL OPTIONS
                       --fname {model, FNAME}
                       --model {synthetic_mlp_default, MODEL} # model architecture
                       
                       # DATASET OPTIONS
                       --dataset {synthetic1d, synthetic2d}
                       --synthetic_size {10, SYNTHETIC_SIZE} # number of hidden instances generated
                       --input_size {5, INPUT_SIZE} # input dimension
                       --input_channel {1, INPUT_CHANNEL} # number of input channels
                       --data_range {0.1, DATA_RANGE} # the range of the dataset [-c, c]
                       --epsilon {0.1, EPSILON} # perturbation radius

                       # TRAINING OPTIONS
                       --batch_size {512, BATCH_SIZE}
                       --epochs {5000, EPOCHS}
                       --lr-max {1e-3, LR-MAX}
                       --seed {0, SEED}

                       # ADDITIONAL TECHNIQUES OPTIONS
                       # See our paper for details
                       --margin_obj
                       --counter_margin {0.01, COUNTER_MARGIN} 
                       --window_size {1, WINDOW_SIZE}
```

Below are some example commands:

```bash
# Example training command for a MLP model
python adv_training.py --fname model --model synthetic_mlp_default --dataset synthetic1d --synthetic_size 10 --input_size 5 --epsilon 0.02 --epochs 5000 --lr-max 1e-3 --alpha 0.02 --margin_obj --window_size 300

# Example training command for a CNN model
python adv_training.py --fname model --model synthetic_cnn_default --dataset synthetic2d --synthetic_size 10 --input_size 5 --epsilon 0.05 --epochs 5000 --lr-max 1e-3 --alpha 0.0005 --margin_obj --window_size 300
```

After the training has completed, run the following command to evaluate the trained model to see if the counterexamples are truly hidden and generate onnx model and VNNLIB files. The parameters should be consistent with the parameters during training to avoid errors.

```bash
python cross_attack_evalution.py --fname FNAME # filename of your trained model
                                 --model MODEL # model architecture
                                 --dataset {synthetic1d, synthetic2d}
                                 --data_range DATA_RANGE # data range previously used
                                 --epsilon EPSILON # perturbation radius previously used
                                 --batch_size {512, BATCH_SIZE}
                                 --result_path {result, PATH} # root path of models' data directory
                                 --output_path {verification, PATH} # path to store VNNLIB
```

Below are some example commands:

```bash
# For the previously trained MLP model:
python cross_attack_evaluation.py --fname model --model synthetic_mlp_default --dataset synthetic1d --epsilon 0.02 --pgd --result_path verification/model

# For the previously trained CNN model:
python cross_attack_evaluation.py --fname model --model synthetic_cnn_default --dataset synthetic2d --epsilon 0.02 --pgd --result_path verification/model
```

## Citation
CITATION

## Contact

Please reach out to us if you have any questions or suggestions.
You can submit an issue or pull request, or send an email to:

- Harry Zhou ([hzhou27@g.ucla.edu](mailto:hzhou27@g.ucla.edu)), UCLA
- Hongji Xu ([hx84@duke.edu](mailto:hx84@duke.edu)), Duke University
- Andy Xu ([Xuandy05@gmail.com](mailto:Xuandy05@gmail.com)), UCLA
- Zhouxing Shi ([zshi@cs.ucla.edu](mailto:zshi@cs.ucla.edu)), UCLA

Thank you for your interest in SoundnessBench. We hope that our work will serve as a valuable resource for developers building and improving NN verifiers in the future.
