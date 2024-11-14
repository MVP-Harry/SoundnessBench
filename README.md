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

The downloaded benchmark should a total of 26 models across 9 distinct NN architectures with different input sizes and perturbation radii.
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
* `run_all_verification.py` provides an easy way to run verification on SoundnessBench for four verifiers: alpha-beta-CROWN, Marabou, NerualSAT, and PyRat

## Citation
CITATION

## Contact
Please reach out to us if you have any questions or suggestions. You can submit an issue or pull request, or send an email to [hzhou27@g.ucla.edu](mailto:hzhou27@g.ucla.edu). 

Thank you for your interest in SoundnessBench. We hope that our work will serve as a valuable resource for developers building and improving NN verifiers in the future.
