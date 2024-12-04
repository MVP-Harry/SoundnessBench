current_dir=$(pwd)
eval "$(conda shell.bash hook)"
# install pyrat 2024 for vnncomp
git clone https://git.frama-c.com/pub/pyrat.git && cd pyrat
conda env create -f pyrat_env.yml --name pyrat
cd ${current_dir}
# install abcrown 2024 vnncomp
cd alpha-beta-CROWN_vnncomp2024
conda env create -f environment_pyt231.yaml --name alpha-beta-crown
cd ${current_dir}
# install neuralsat 2024 vnncomp
cd neuralsat
conda env create -f env.yaml --name neuralsat
cd ${current_dir}