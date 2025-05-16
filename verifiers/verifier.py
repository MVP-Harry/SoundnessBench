from abc import ABC, abstractmethod
import re
import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional


def build_command(base_cmd, **kwargs):
    cmd = base_cmd.copy()
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([key, str(value)])
    return cmd


class Verifier(ABC):
    def __init__(self, name, bench_dir, results_dir, backup_dir=None):
        self.name = name
        self.bench_dir = Path(bench_dir).resolve()
        self.results_dir = Path(results_dir) / self.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if backup_dir is not None:
            self.backup_dir = Path(backup_dir).resolve() / self.name
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.backup_dir = None
        self.original_dir = Path.cwd()
        self.env = os.environ.copy()
        self.match_result_pattern = self.match_time_pattern = None
        # some verifier does not handle timeout properly,
        # so we handle it by system-level timeout as it is done in vnncomp
        # the same additional time is added for initialization.
        self.init_time = 10

    @abstractmethod
    def run(self, idx, model_path, property_path, timeout) -> Tuple[Optional[str], Optional[float]]:
        pass

    def match_result(self, line):
        if self.match_result_pattern:
            match = re.search(self.match_result_pattern, line)
            return match.group(1).lower() if match is not None else None
        else:
            raise NotImplementedError

    def match_time(self, line):
        if self.match_time_pattern:
            match = re.search(self.match_time_pattern, line)
            return match.group(1).lower() if match is not None else None
        else:
            raise NotImplementedError

    def run_command(self, command, idx=None):
        command = ' '.join(command)
        print(f"Executing: {command}\n")
        if idx is None:
            os.system(f"{command}")
        else:
            log_file = self.results_dir / f"{idx}.log"
            os.system(f"{command} 2>&1 | tee {log_file}")
            if self.backup_dir is not None:
                os.system(f"cp {log_file} {self.backup_dir / f'{idx}.log'}")
            return log_file

    def finalize(self, log_file):
        ret = "Error"
        time = None
        with open(log_file) as file:
            for line in file:
                if ret == "Error":
                    result_match = self.match_result(line)
                    if result_match is not None:
                        ret = result_match
                if time is None:
                    t = self.match_time(line)
                    if t is not None:
                        time = float(t)
                if ret != "Error" and time is not None:
                    break

        return ret, time


class VerifierABCROWN(Verifier):
    def __init__(self, name, bench_dir, results_dir, split_type, backup_dir=None):
        super().__init__(name, bench_dir, results_dir, backup_dir)
        self.target_dir = Path(
            "verifiers/alpha-beta-CROWN_vnncomp2024/complete_verifier"
        ).resolve()
        self.split_type = split_type
        self.match_result_pattern =r"Result: (\S+)"
        self.match_time_pattern = r"Time: ([\d.]+)"

    def run(self, idx, model_path, property_path, timeout):
        base_cmd_str = (
            f'timeout {timeout + self.init_time} conda run --no-capture-output -n alpha-beta-crown '
            f'--cwd {self.target_dir} '
            f'python abcrown.py '
            f'--config {Path("verifiers/abcrown_config.yaml").resolve()} '
            f'--onnx_path {model_path} '
            f'--vnnlib_path {property_path} '
            f'--override_timeout {timeout} '
        )

        if self.split_type == "input":
            base_cmd_str += (
                "--enable_input_split "
                "--branching_method sb "
                "--bound_prop_method crown "
            )
        if "vit" in str(self.bench_dir) or "ViT" in str(self.bench_dir):
            base_cmd_str += "--conv_mode matrix "

        base_cmd_str += (
            f'; status=$?; '
            f'if [ $status -eq 124 ]; then echo "\nResult: timeout Time: {timeout + self.init_time}"; fi; '
            f'exit $status;'
        )

        command = build_command([
            "bash", "-c", f"'{base_cmd_str}'"
        ])

        log_file = self.run_command(command, idx)
        return self.finalize(log_file)


class VerifierNeuralSAT(Verifier):
    def __init__(self, name, bench_dir, results_dir, split_type, backup_dir=None):
        super().__init__(name, bench_dir, results_dir, backup_dir)
        self.split_type = split_type
        self.match_result_pattern = r"Result: (\w+)"
        self.match_time_pattern = r"Runtime: ([\d.]+)"

    def run(self, idx, model_path, property_path, timeout):
        # command = build_command([
        #     "conda", "run", "--no-capture-output", "-n", "neuralsat",
        #     "python", "verifiers/neuralsat/neuralsat-pt201/main.py",
        #     "--device", "cuda",
        #     "--net", str(model_path),
        #     "--spec", str(property_path),
        #     "--timeout", str(timeout),
        #     "--force_split", str(self.split_type),
        # ])
        command = build_command([
            "bash", "-c",
            f'\'timeout {timeout + self.init_time} conda run --no-capture-output -n neuralsat '
            f'python verifiers/neuralsat/neuralsat-pt201/main.py '
            f'--device cuda '
            f'--net {model_path} '
            f'--spec {property_path} '
            f'--timeout {timeout} '
            f'--force_split {self.split_type}; '
            f'status=$?; '
            f'if [ $status -eq 124 ]; then echo "\nResult: Timeout, Runtime: {timeout + self.init_time}"; fi; '
            f'exit $status;\''
        ])
        log_file = self.run_command(command, idx)
        return self.finalize(log_file)


class VerifierPyRAT(Verifier):
    def __init__(self, name, bench_dir, results_dir, backup_dir=None):
        super().__init__(name, bench_dir, results_dir, backup_dir)
        self.match_result_pattern = r"Result = (\w+), Time = ([\d.]+) s, Safe space = ([\d.]+) %, number of analysis = (\d+)"
        self.match_time_pattern = r'Time = ([\d.]+) s'

    def run(self, idx, model_path, property_path, timeout):

        command = build_command([
            "bash", "-c",
            f'\'timeout {timeout + self.init_time} conda run --no-capture-output -n pyrat '
            f'python verifiers/pyrat/pyrat.pyc '
            f'--model_path {model_path} '
            f'--property_path {property_path} '
            f'--timeout {timeout} '
            f'--split True '
            f'--domains [zonotopes,poly,symbox] '
            f'--check before '
            f'--attack pgd --attack_it 100 --pgd_steps 100; '
            f'status=$?; '
            f'if [ $status -eq 124 ]; then echo "\nResult = Timeout, Time = {timeout + self.init_time} s, Safe space = 0.00 %, number of analysis = 0"; fi; '
            f'exit $status;\''
        ])
        log_file = self.run_command(command, idx)
        return self.finalize(log_file)


class VerifierMarabou(Verifier):
    def __init__(self, name, bench_dir, results_dir, container_name, backup_dir=None):
        super().__init__(name, bench_dir, results_dir, backup_dir)
        self.container_name = container_name
        self.match_result_pattern = r"Appending result '(\w+)' to csv file"
        self.match_time_pattern = r"Runtime: ([\d.]+)"

    def run(self, idx, model_path, property_path, timeout):
        def is_container_running(name):
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={name}"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            print(result.stdout.strip())
            return result.stdout.strip() != ""

        if not is_container_running(self.container_name):
            raise RuntimeError("Container {container_name} is not running")
        container_model_path = f"/host_dir/{model_path.relative_to(Path.cwd())}"
        container_property_path = f"/host_dir/{property_path.relative_to(Path.cwd())}"

        # marabou does not handle timeout, instead, it is done in the script run_single_instance.sh from vnncomp
        # we set a 5s TIMEOUT_TOLERANCE in the script for initialization as well. 
        run_command = [
            "docker", "exec", self.container_name, "bash", "-c",
            f'"source /root/miniconda3/bin/activate marabou && '
            f'cd ~/marabou/vnncomp && '
            f'bash run_single_instance.sh v1 . {f"{model_path}_{idx}/"} {container_model_path} {container_property_path} '
            f'{timeout} out.csv counterexample"'
        ]
        # self.run_command(prepare_command)
        log_file = self.run_command(run_command, idx)

        # force cleanup any running marabou process by restarting the container
        cleanup_command = [
            "docker", "restart", self.container_name
        ]
        self.run_command(cleanup_command)

        return self.finalize(log_file)
