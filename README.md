# *SoundnessBench*: A Comprehensive Benchmark to Evaluate the Soundness of Neural Network Verifiers
![Verification flow](/assets/flow.png)

## Overview

This repository contains the source code of SoundnessBench, a benchmark designed to thoroughly evaluate the soundness of neural network (NN) verifiers. It addresses the critical lack of ground-truth labels in previous benchmarks for NN verifiers by providing numerous unverifiable instances with hidden counterexamples, so that it can effectively reveal internal bugs of NN verifiers if they falsely claim verifiability on those unverifiable instances. SoundnessBench aims to support developers in evaluating and improving NN verifiers. See our paper for more details.

LINK TO PAPER

## Download SoundnessBench
SoundnessBench is hosted on [Hugging Face](https://huggingface.co/datasets/SoundnessBench/SoundnessBench). To directly download the benchmark, use 
```bash
git clone https://huggingface.co/datasets/SoundnessBench/SoundnessBench
```

The downloaded benchmark should contain a total of 26 models across 9 distinct NN architectures with different input sizes and perturbation radii.
![Model architectures](/assets/model_architectures.png)

Each folder should contain 
* `model.onnx` and `model.pt`, the final model checkpoints
* `data.pt`, checkpoint to instances that verification will be performed on
* `instances.csv` and `vnnlib`, VNNLIB files that make parsing data easier

## Basic Usage
We provide a script to quickly count the results in the results.csv files, including metrics such as `clean_instance_verified_ratio`, `clean_instance_falsified_ratio`, `unverifiable_instance_verified_ratio`, and `unverifiable_instance_falsified_ratio`. It supports any csv file that complies with the VNNCOMP format ([example](https://github.com/ChristopherBrix/vnncomp2024_results/blob/main/never2/results.csv)).

```bash
python eval.py <path to results.csv> <path to output.csv>
```

## Run Existing Verifiers on *SoundnessBench*

Our experiments are all based on managing environments with conda, so make sure you have conda installed before starting.

---

### Step 1: Install Verification Tools

For verification with NeuralSAT on certain networks, you will need to use [Gurobi](https://www.gurobi.com/). 

Please ensure that the Gurobi license is placed in the root directory of this project. You can obtain a free academic Gurobi license [here](https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2Fiam%2Flicenses%2Frequest%2F%3Ftype%3Dacademic).

You can install **alpha-beta-CROWN**, **NeuralSAT**, and **PyRAT** using the following command:
```bash
bash install.sh
```

Running **Marabou** is a bit more complex. Our experiments use **Docker** to install Marabou. Please set up the Docker environment and mount [marabou_vnncomp_2023](https://github.com/wu-haoze/Marabou/tree/vnn-comp-23) or [marabou_vnncomp_2024](https://github.com/wu-haoze/Marabou/tree/vnn-comp-24) in your container. Once you have everything set up, run the following script to install Marabou:

```bash
conda create -y --name marabou python=3.8
conda activate marabou
cd ~/marabou/vnncomp
bash install_tool.sh
cp -r ~/marabou/opt ~/
```

---

### Step 2: Run Verification

#### 2.1: Run alpha-beta-CROWN

To run **alpha-beta-CROWN** with activation split:

```bash
python run_all_verification.py --verifier abcrown --model_folder <path_to_soundness_bench>
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
python eval.py <path to results.csv> <path to output.csv>
```

## Train New Models
We provide an easy pipeline for users to train new models that contain hidden adversarial examples.

### File Usage Overview
* `synthetic_data_generation.py` generates synthetic data with pre-defined hidden counterexamples
* `adv_training.py` trains models on the synthetically generated dataset using a two-objective training framework
* `cross_attack_evaluation.py` uses very strong adversarial attacks to make sure that no trivial counterexamples can be found and saves those instances into VNNLIB files
* `models` contain the definitions of all NN architectures we use in SoundnessBench
<!-- * `run_all_verification.py` provides an easy way to run verification on SoundnessBench for four verifiers: alpha-beta-CROWN, Marabou, NerualSAT, and PyRat, results will be saved in `results` folder. -->

### Install Dependencies
TODO

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
                       --lr-type {cyclic, decay, flat}
                       --seed {0, SEED}

                       # ATTACK OPTIONS
                       --attack {pgd, fgsm, none} # adversarial attack used in training
                       --attack_eval {pgd, fgsm, aa, none} # adversarial attack used in evaluation
                       --alpha {0.01, ALPHA} # attack alpha used in PGD attack
                       --attack-iters {75, ATTACK-ITERS} # number of attack iterations used in PGD attack
                       --pgd_loss_type {ce, margin} 
                       --restarts {75, RESTARTS} # number of random restarts used in PGD attack during evaluation
                       --restarts_train {75, RESTARTS_TRAIN} # number of random restarts used in PGD attack during training
                       
                       # ADDITIONAL TECHNIQUES OPTIONS
                       # See our paper for details
                       --margin_obj
                       --counter_margin {0.01, COUNTER_MARGIN} 
                       --window_size {1, WINDOW_SIZE}
                       
                       # OTHER OPTIONS
                       --eval_interval {200, EVAL_INTERVAL}
                       --save_interval {200, SAVE_INTERVAL}
                       --log_interval {1, LOG_INTERVAL}
```

Below are some example commands:
```bash
# Example training command for a MLP model
python adv_training.py --fname model --model synthetic_mlp_default --dataset synthetic1d --synthetic_size 10 --input_size 5 --epsilon 0.02 --epochs 5000 --lr-max 1e-3 --restarts_train 75 --restarts 75 --attack-iters 75 --alpha 0.02 --margin_obj --window_size 300

# Example training command for a CNN model
python adv_training.py --fname model --model synthetic_cnn_default --dataset synthetic2d --synthetic_size 10 --input_size 5 --epsilon 0.05 --epochs 5000 --lr-max 1e-3 --restarts_train 100 --restarts 100 --attack-iters 100 --alpha 0.0005 --margin_obj --window_size 300

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
