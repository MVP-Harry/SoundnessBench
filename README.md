# SoundnessBench: A Comprehensive Benchmark to Evaluate the Soundness of Neural Network Verifiers

## Overview

This repository contains the source code of SoundnessBench, a benchmark designed to thoroughly evaluate the soundness of neural network (NN) verifiers. It addresses the critical lack of ground-truth labels in previous benchmarks for NN verifiers by providing numerous unverifiable instances with hidden counterexamples, so that it can effectively reveal internal bugs of NN verifiers if they falsely claim verifiability on those unverifiable instances. SoundnessBench aims to support developers in evaluating and improving NN verifiers. See our paper for more details.

LINK TO PAPER

## Download SoundnessBench
SoundnessBench is hosted on [Hugging Face](https://huggingface.co/datasets/SoundnessBench/SoundnessBench). To directly download the benchmark, use 
```bash
git clone https://huggingface.co/datasets/SoundnessBench/SoundnessBench
```

*HZ: currently this is private?*

The downloaded benchmark should contain a total of 26 models across 9 distinct NN architectures with different input sizes and perturbation radii.
![Model architectures](/assets/model_architectures.png)

Each folder should contain 
* `model.onnx` and `model.pt`, the final model checkpoints
* `data.pt`, checkpoint to instances that verification will be performed on
* `instances.csv` and `vnnlib`, VNNLIB files that make parsing data easier

## Tutorial
![Verification flow](/assets/flow.png)
After downloading SoundnessBench, you can test any desired NN verifier by providing it with the necessary files. (*HZ: is support for VNNLIB format mandatory?*) Each NN verifier may require a slightly different setup for optimal performance. We provide a script `run_all_verification.py` that supports running our benchmark on four well-established verifiers: [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN/), [Marabou](https://github.com/NeuralNetworkVerification/Marabou), [NerualSAT](https://github.com/dynaroars/neuralsat), and [PyRat](https://github.com/pyratlib/pyrat).

*HZ: perhaps Hongji can elaborate on this a bit more? Shall we also provide a setup instruction for other verifiers?*

## Data Generation, Training, and Evaluation
*HZ: do we need to provide details on how we generate data, train models, and evaluate? I checked the README of sorry-bench and DecodingTrust, and they seem to omit this part? Currently I'm providing a high-level overview of each file that we have.*

### File Usage
* `synthetic_data_generation.py` generates synthetic data with pre-defined hidden counterexamples
* `adv_training.py` trains models on the synthetically generated dataset using a two-objective training framework
* `cross_attack_evaluation.py` uses very strong adversarial attacks to make sure that no trivial counterexamples can be found and saves those instances into VNNLIB files
* `models` contain the definitions of all NN architectures we use in SoundnessBench
* `run_all_verification.py` provides an easy way to run verification on SoundnessBench for four verifiers: alpha-beta-CROWN, Marabou, NerualSAT, and PyRat, results will be saved in `results` folder.

### Run Benchmark

Our experiments are all based on managing environments with conda, so make sure you have conda installed before starting.

---

### Step 1: Install Verification Tools

For verification with NeuralSAT on certain networks, you will need to use [Gurobi](https://www.gurobi.com/). 

Please ensure that the Gurobi license is placed in the root directory of the project. You can obtain a free academic Gurobi license [here](https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2Fiam%2Flicenses%2Frequest%2F%3Ftype%3Dacademic).

You can install **abcrown**, **NeuralSAT**, and **PyRAT** using the following command:
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

#### 2.1: Run abcrown

To run **abcrown** with activation split:

```bash
python run_all_verification.py --verifier abcrown --model_folder <path_to_soundness_bench>
```

To run **abcrown** with input split:

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
# Change <marabou_container> to your Marabou container.
python run_all_verification.py --verifier marabou --model_folder <path_to_soundness_bench> --container_name <marabou_container>
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
