#!/bin/bash

INIT_DIR=$PWD

echo $INIT_DIR

BENCHMARK=$2
ONNX_FILE=$(realpath $3)
VNNLIB_FILE=$(realpath $4)
NET_NAME=$(basename ${ONNX_FILE%.onnx})
PROP_NAME=$(basename ${VNNLIB_FILE%.vnnlib})

echo "Solving benchmark set '$BENCHMARK' with onnx file '$ONNX_FILE' and vnnlib file '$VNNLIB_FILE'"
echo "Network name: $NET_NAME"
echo "Property name: $PROP_NAME"

####### Creating working directories #######

# list="acasxu cgan collins_rul_cnn dist_shift ml4acopf nn4sys tllverifybench traffic_signs_recognition vggnet16 vit"

# if [[ $list =~ (^|[[:space:]])$BENCHMARK($|[[:space:]]) ]]; then
#     echo "Supported benchmark"
# else
#     echo "Unsupported benchmark" $BENCHMARK
#     exit 1
# fi


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKING_DIR_BENCHMARK=$SCRIPT_DIR/data/$BENCHMARK/
WORKING_DIR_INSTANCE=$WORKING_DIR_BENCHMARK/"$NET_NAME"_"$PROP_NAME"/

if [[ ! -d $WORKING_DIR_BENCHMARK ]]
then
    mkdir -p $WORKING_DIR_BENCHMARK
fi
if [[ -d $WORKING_DIR_INSTANCE ]]
then
    rm -rf $WORKING_DIR_INSTANCE
fi
mkdir $WORKING_DIR_INSTANCE

echo "Working directory: $WORKING_DIR_INSTANCE"

# kill any remaining python processes.
pkill -9 python
pkill -9 python3
pkill -9 run_instance.sh
sleep 1
pkill -9 python
pkill -9 python3
pkill -9 run_instance.sh
sleep 1

############################### NETWORK PROCESSING #################################

####### Simplify #######
ONNX_FILE_SIMP=$WORKING_DIR_BENCHMARK/"$NET_NAME"_simp.onnx
if [[ ! -f $ONNX_FILE_SIMP ]]
then
    python3 -m onnxsim $ONNX_FILE $ONNX_FILE_SIMP
    if [ $? != 0 ]
    then
        echo "Simplification failed with default argument"
        python3 -m onnxsim $ONNX_FILE $ONNX_FILE_SIMP --mutable-initializer
    fi
else
    echo "Simplified network already exists"
fi
if [[ -f $ONNX_FILE_SIMP ]]
then
   echo "Simplified ONNX file: $ONNX_FILE_SIMP"
else
    echo "Simplified ONNX file NOT FOUND!"
    exit 1
fi

###### Remove Softmax ######
ONNX_FILE_PRESOFTMAX=$WORKING_DIR_BENCHMARK/"$NET_NAME"_simp_presoftmax.onnx
if [[ ! -f $ONNX_FILE_PRESOFTMAX ]]
then
    python3 $SCRIPT_DIR/process_network/get_presoftmax_network.py $ONNX_FILE_SIMP $ONNX_FILE_PRESOFTMAX
else
    echo "Pre-softmax network already exists"
fi
if [[ -f $ONNX_FILE_PRESOFTMAX ]]
then
    echo "Pre-softmax ONNX file: $ONNX_FILE_PRESOFTMAX"
else
    echo "Pre-softmax ONNX file NOT FOUND!"
    exit 1
fi

###### Convert large networks ######
echo "Convert Max to ReLU"
ONNX_FILE_POSTDNNV=$WORKING_DIR_BENCHMARK/"$NET_NAME"_simp_presoftmax_postdnnv.onnx
if [[ ! -f $ONNX_FILE_POSTDNNV ]]
then
    python3 $SCRIPT_DIR/process_network/simplify_with_dnnv.py $ONNX_FILE_PRESOFTMAX $ONNX_FILE_POSTDNNV
    if [ $? != 0 ]
    then
        echo "DNNV preprocessing failed"
        cp $ONNX_FILE_PRESOFTMAX $ONNX_FILE_POSTDNNV
    fi
fi

if [[ -f $ONNX_FILE_POSTDNNV ]]
then
    echo "Post-DNNV ONNX file: $ONNX_FILE_POSTDNNV"
else
    echo "Post-DNNV ONNX file NOT FOUND!"
    exit 1
fi


############################### PROPERTY PROCESSING #################################
VNNLIB_FILE_PICKLED=$WORKING_DIR_INSTANCE/vnnlib.pkl
IPQ_FILE=$WORKING_DIR_INSTANCE/query.ipq
python3 $SCRIPT_DIR/process_property/parse_vnnlib.py $ONNX_FILE_POSTDNNV $VNNLIB_FILE $VNNLIB_FILE_PICKLED $IPQ_FILE
if [[ -f $VNNLIB_FILE_PICKLED ]]
then
    echo "VNNLIB parsed: $VNNLIB_FILE_PICKLED"
else
    echo "VNNLIB not parsed!"
    exit 1
fi

exit 0

if [[ -f $IPQ_FILE ]]
then
    echo "Ipq created: $IPQ_FILE"
else
    echo "Ipq not created"
fi

# Warmup, using a 1 second timeout.
echo
echo "TODO: running warmup..."
echo


script_name=$(realpath $0)
script_path=$(dirname "$script_name")
project_path=$(dirname "$script_path")


# kill any remaining python processes.
pkill -9 Marabou
pkill -9 python
pkill -9 python3
sleep 2
pkill -9 Marabou
pkill -9 python
pkill -9 python3
sleep 2
echo "Preparation finished."

# script returns a 0 exit code if successful. If you want to skip a benchmark category you can return non-zero.
exit 0
