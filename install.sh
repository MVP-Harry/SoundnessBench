#!/bin/bash

eval "$(conda shell.bash hook)"

VERIFIER_DIR=verifiers
MARABOU_SETUP_DIR=$VERIFIER_DIR/marabou_setup

# Parse --verifier argument
verifiers=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --verifier)
        IFS=',' read -ra parts <<< "$2"
        for v in "${parts[@]}"; do
            verifiers+=("$v")
        done
        shift 2
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Default list if no --verifier is given
if [[ ${#verifiers[@]} -eq 0 ]]; then
    verifiers=(pyrat abcrown neuralsat marabou_vnncomp_2023 marabou_vnncomp_2024)
fi

check_installed() {
local name=$1
local env=$2
local dir=$3
local docker=$4

local found=0

if [[ -n "$env" ]]; then
    if conda info --envs | grep -qE "^$env[[:space:]]"; then
    echo "[Warning] Conda environment '$env' for '$name' already exists."
    echo "          If you want to re-install, please remove it manually (conda env remove -n $env)."
    found=1
    fi
fi

if [[ -n "$dir" && -d "$dir" ]]; then
    echo "[Warning] Directory '$dir' for '$name' already exists."
    echo "          If you want to re-install, please remove it manually (rm -rf $dir)."
    found=1
fi

if [[ -n "$docker" ]]; then
    if docker ps -a --format '{{.Names}}' | grep -q "^${docker}$"; then
    echo "[Warning] Docker container '$docker' for '$name' already exists."
    echo "          If you want to re-install, please remove it manually (docker rm -f $docker)."
    found=1
    fi
fi

return $found
}

install_pyrat() {
    # check_installed "pyrat" "pyrat" "" "" || return
    echo "Installing PyRAT..."
    git clone https://git.frama-c.com/pub/pyrat.git $VERIFIER_DIR/pyrat
    conda env create -f $VERIFIER_DIR/pyrat/pyrat_env.yml --name pyrat
}

install_abcrown() {
    # check_installed "abcrown" "alpha-beta-crown" "" "" || return
    echo "Installing alpha-beta-CROWN..."
    conda env create -f $VERIFIER_DIR/alpha-beta-CROWN_vnncomp2024/complete_verifier/environment_pyt231.yaml --name alpha-beta-crown
}

install_neuralsat() {
    # check_installed "neuralsat" "neuralsat" "" "" || return
    echo "Installing NeuralSAT..."
    conda env create -f $VERIFIER_DIR/neuralsat/env.yaml --name neuralsat
}

install_marabou_common() {
    local folder=$1
    local branch=$2

    # check_installed "$folder" "" "$VERIFIER_DIR/$folder" "$folder" || return
    echo "Installing Marabou ($folder)..."
    git clone https://github.com/wu-haoze/Marabou.git $VERIFIER_DIR/$folder
    (cd $VERIFIER_DIR/$folder; git switch $branch)

    # We encountered an error of "ImportError: cannot import name 'MarabouCore' from 'maraboupy' (/marabou/maraboupy/__init__.py)"
    # if we do not explicitly expose the MarabouCore module.
    echo "from .MarabouCore import *" > $VERIFIER_DIR/$folder/maraboupy/__init__.py

    # We replace the original prepare_instance.sh file with our prepare_instance.sh files
    # which comment the checking of supported benchmarks.
    cp $MARABOU_SETUP_DIR/prepare_instance_${folder}.sh $VERIFIER_DIR/$folder/vnncomp/prepare_instance.sh
    # We will run test instances via the run_single_instance.sh from vnncomp
    cp $MARABOU_SETUP_DIR/run_single_instance.sh $VERIFIER_DIR/$folder/vnncomp/run_single_instance.sh
}

install_marabou_docker() {
    local folder=$1
    local name=$2

    echo "Setting up Docker for $name..."
    docker build -f $MARABOU_SETUP_DIR/Dockerfile -t $name .

    # launch marabou docker with mounting the current directory and the marabou directory
    docker run -dit --name $name \
        -v $(pwd)/$VERIFIER_DIR/$folder:/root/marabou \
        -v $(pwd):/host_dir \
        $name

    # create the conda environment, set up gurobi license, and install marabou
    docker exec -it $name bash -c "source ~/miniconda3/etc/profile.d/conda.sh && \
        conda create -n marabou python=3.8 -c conda-forge && \
        conda install -c conda-forge libstdcxx-ng>=12 --override-channels --yes && \
        conda activate marabou && \
        mkdir -p /opt/gurobi/ && \
        cp /host_dir/$MARABOU_SETUP_DIR/gurobi.lic /opt/gurobi/ && \
        cp /host_dir/$MARABOU_SETUP_DIR/gurobi.lic /root/ && \
        cd /root/marabou/vnncomp && \
        bash install_tool.sh"
}

check_gurobi_license() {
    if [ ! -f $MARABOU_SETUP_DIR/gurobi.lic ]; then
        echo "[Warning] gurobi.lic not found, please download your Web License and place it in $MARABOU_SETUP_DIR"
    fi
}

# Dispatcher
for verifier in "${verifiers[@]}"; do
    case "$verifier" in
        pyrat)
            if check_installed "pyrat" "pyrat" "" ""; then
                install_pyrat
            fi
            ;;
        abcrown)
            if check_installed "abcrown" "alpha-beta-crown" "" ""; then
                install_abcrown
            fi
            ;;
        neuralsat)
            if check_installed "neuralsat" "neuralsat" "" ""; then
                install_neuralsat
            fi
            ;;
        marabou_vnncomp_2023)
            if check_installed "marabou_vnncomp_2023" "" "" "marabou_vnncomp_2023"; then
                check_gurobi_license
                install_marabou_common "marabou_vnncomp_2023" "vnn-comp-23"
                # Pybind11 2.3.0 used in marabou_vnncomp_2023 is outdated and does not include <cstdint> internally where needed
                # We modify it a bit to include <cstdint> internally when installed
                cp $MARABOU_SETUP_DIR/download_pybind11.sh $VERIFIER_DIR/marabou_vnncomp_2023/tools/download_pybind11.sh
                install_marabou_docker "marabou_vnncomp_2023" "marabou_vnncomp_2023"
                docker exec -it marabou_vnncomp_2023 bash -c "cp -r /root/marabou/opt /root/"
            fi
            ;;
        marabou_vnncomp_2024)
            if check_installed "marabou_vnncomp_2024" "" "" "marabou_vnncomp_2024"; then
                check_gurobi_license
                install_marabou_common "marabou_vnncomp_2024" "vnn-comp-24"
                install_marabou_docker "marabou_vnncomp_2024" "marabou_vnncomp_2024"
            fi
            ;;
        *)
            echo "Unknown verifier: $verifier"
            ;;
    esac
done
