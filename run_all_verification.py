import argparse
import re
from collections import defaultdict
import os
import subprocess
import datetime
from pathlib import Path
import csv
def get_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier", required=True, nargs='+', type=str)
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        type=str,
    )
    parser.add_argument("--model_folder", default=None, type=str)
    parser.add_argument("--timeout", default=100, type=int)
    parser.add_argument("--drop_rate", default=None, type=float)
    parser.add_argument("--perturb_alpha", default=None, type=float)
    parser.add_argument("--eps", default=None, type=float)
    parser.add_argument("--split_type", default="hidden", type=str)
    parser.add_argument("--attack_it", default=None, type=int)
    parser.add_argument("--pgd_steps", default=None, type=int)
    parser.add_argument("--container_name", default=None, type=str)
    args = parser.parse_args()

def parse_abcrown_log(log_file_path):
    result = {"unsat": [], "sat": [], "unknown": []}

    with open(log_file_path, "r") as log_file:
        log_data = log_file.read()

        safe_incomplete_match = re.search(
            r"safe-incomplete \(total \d+\), index: \[([0-9, ]+)\]", log_data
        )
        if safe_incomplete_match:
            result["unsat"].extend(map(int, safe_incomplete_match.group(1).split(",")))

        safe_match = re.search(r"safe \(total \d+\), index: \[([0-9, ]+)\]", log_data)
        if safe_match:
            result["unsat"].extend(map(int, safe_match.group(1).split(",")))

        unknown_match = re.search(
            r"unknown \(total \d+\), index: \[([0-9, ]+)\]", log_data
        )
        if unknown_match:
            result["unknown"].extend(map(int, unknown_match.group(1).split(",")))

        unsafe_match = re.search(
            r"unsafe-pgd \(total \d+\), index: \[([0-9, ]+)\]", log_data
        )
        if unsafe_match:
            result["sat"].extend(map(int, unsafe_match.group(1).split(",")))
            
        unsafe_match_bab = re.search(
            r"unsafe-bab \(total \d+\), index: \[([0-9, ]+)\]", log_data
        )
        if unsafe_match_bab:
            result["sat"].extend(map(int, unsafe_match_bab.group(1).split(",")))

        result["unsat"] = sorted(result["unsat"])
        result["sat"] = sorted(result["sat"])
        result["unknown"] = sorted(result["unknown"])
    return result


def parse_neuralsat_log(log_file_path):
    result = defaultdict(list)
    index = 0

    with open(log_file_path, "r") as log_file:
        lines = log_file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            result_match = re.search(r"Result: (\w+)", line)
            if result_match:
                current_result = result_match.group(1).lower()
                result[current_result].append(index)
                index += 1
                i += 1

            i += 1

    return result


def parse_pyrat_log(log_file_path):
    result = defaultdict(list)
    index = 0

    with open(log_file_path, "r") as log_file:
        for line in log_file:
            match = re.search(
                r"Result = (\w+), Time = ([\d.]+) s, Safe space = ([\d.]+) %, number of analysis = (\d+)",
                line,
            )
            if match:
                current_result = match.group(1).lower()
                if current_result == "true":
                    current_result = "unsat"  # 将 "true" 转换为 "unsat"
                result[current_result].append(index)
                index += 1

    return result


def parse_marabou_log(log_file_path):
    result = defaultdict(list)
    index = 0

    with open(log_file_path, "r") as log_file:
        lines = log_file.readlines()

    for i, line in enumerate(lines):
        result_match = re.search(r"Appending result '(\w+)' to csv file", line)
        if result_match:
            current_result = result_match.group(1).lower()
            result[current_result].append(index)
            index += 1

    return result

def save_parsed_result(parsed_result, output_file_path, args=None):
    with open(output_file_path, "w") as output_file:
        if args is not None:
            output_file.write("Run Arguments:\n")
            for arg, value in vars(args).items():
                output_file.write(f"{arg}: {value}\n")
            output_file.write("\nParsed Results:\n\n")

        for key in parsed_result:
            output_file.write(f"{key}: {parsed_result[key]}\n")


def build_command(base_cmd, **kwargs):
    cmd = base_cmd.copy()
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([key, str(value)])
    return cmd


def run_verifier(
    verifier: str,
    config_path: str,
    timeout: int = None,
    drop_rate: float = None,
    perturb_alpha: float = None,
    eps: float = None,
    attack: int = None,
    attack_it: int = None,
    pgd_steps: int = None,
    container_name: str = None,
    split_type: str = "hidden",
):
    config_path = Path(config_path).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_basename = config_path.name
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_basename = Path(f"{verifier}-{config_basename}-{timestamp}.log")
    log_file = log_dir / log_file_basename

    with log_file.open("a") as lf:
        lf.write(f"run {verifier} in {config_path}\n\n")

    try:
        if verifier == "abcrown":
            original_dir = Path.cwd()
            env = os.environ.copy()
            env["CONFIG_PATH"] = str(config_path)
            env_name = "alpha-beta-crown"
            target_dir = Path(
                "alpha-beta-CROWN_vnncomp2024/complete_verifier"
            ).resolve()
            os.chdir(target_dir)
            base_cmd = [
                "conda",
                "run",
                "-n",
                env_name,
                "--cwd",
                str(target_dir),
                "python",
                "abcrown.py",
                "--config",
                str(config_path / "config.yaml"),
            ]

            if split_type == "input":
                input_cmd = [
                    "--enable_input_split",
                    "--branching_method",
                    "sb",
                    "--bound_prop_method",
                    "crown",
                    "--min_batch_size_ratio",
                    "0.0",
                    "--start_save_best",
                    "-1",
                    "--ibp_enhancement",
                    "--compare_input_split_with_old_bounds",
                    "--sb_sum",
                    "--touch_zero_score",
                    "0.1",
                ]
                base_cmd += input_cmd

            optional_args = {
                "--timeout": timeout,
                "--keep_domain_ratio": drop_rate,
                "--perturbed_alpha": perturb_alpha,
                "--epsilon": eps,
            }
            command = build_command(base_cmd, **optional_args)

            os.chdir(original_dir)
            with log_file.open("a") as lf:
                print(f"Executing: {' '.join(command)}\n")
                lf.write(f"Executing: {' '.join(command)}\n\n")
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                lf.write(result.stdout)

            parsed_result = parse_abcrown_log(log_file)

        elif verifier == "neuralsat":
            env_name = "neuralsat"
            model_path = config_path / "model.onnx"
            instances_file = config_path / "instances.csv"
            env = os.environ.copy()
            if not instances_file.exists():
                print(f"Instances file not found: {instances_file}")
                return

            with instances_file.open("r") as f:
                reader = csv.reader(f)
                instances = [row for row in reader]

            for _, property_path, _ in instances:
                property_full_path = config_path / property_path
                base_cmd = [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",
                    "neuralsat/neuralsat-pt201/main.py",
                    "--device",
                    "cuda",
                    "--net",
                    str(model_path),
                    "--spec",
                    str(property_full_path),
                ]
                optional_args = {
                    "--timeout": timeout,
                    "--drop_rate": drop_rate,
                    "--perturb_alpha": perturb_alpha,
                    "--force_split": split_type,
                }
                command = build_command(base_cmd, **optional_args)

                with log_file.open("a") as lf:
                    print(f"Executing: {' '.join(command)}\n")
                    lf.write(f"Executing: {' '.join(command)}\n\n")
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                    lf.write(result.stdout)
            parsed_result = parse_neuralsat_log(log_file)

        elif verifier == "pyrat":
            env_name = "pyrat"
            model_path = config_path / "model.onnx"
            instances_file = config_path / "instances.csv"

            if not instances_file.exists():
                print(f"Instances file not found: {instances_file}")
                return

            with instances_file.open("r") as f:
                reader = csv.reader(f)
                instances = [row for row in reader]

            for _, property_path, _ in instances:
                property_full_path = config_path / property_path
                base_cmd = [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",
                    "pyrat/pyrat.pyc",
                    "--model_path",
                    str(model_path),
                    "--property_path",
                    str(property_full_path),
                    "--split",
                    "True",
                    "--domains",
                    "[zonotopes,poly,symbox]",
                    "--check",
                    "before",
                    "--attack",
                    "pgd",
                    "--attack_it",
                    "100",
                    "--pgd_steps",
                    "100",
                ]
                optional_args = {
                    "--timeout": timeout,
                    "--epsilon": eps,
                }
                command = build_command(base_cmd, **optional_args)

                with log_file.open("a") as lf:
                    print(f"Executing: {' '.join(command)}\n")
                    lf.write(f"Executing: {' '.join(command)}\n\n")
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                    lf.write(result.stdout)

            parsed_result = parse_pyrat_log(log_file)

        elif verifier == "marabou":
            model_path = config_path / "model.onnx"
            instances_file = config_path / "instances.csv"

            if not instances_file.exists():
                print(f"Instances file not found: {instances_file}")
                return

            with instances_file.open("r") as f:
                reader = csv.reader(f)
                instances = [row for row in reader]

            def is_container_running(name):
                result = subprocess.run(
                    ["docker", "ps", "-q", "-f", f"name={name}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                print(result.stdout.strip())
                return result.stdout.strip() != ""

            if not is_container_running(container_name):
                print("no docker is running")
                exit()
            print("docker is running")
            for _, property_path, _ in instances:
                property_full_path = Path(config_path) / property_path
                output_path = "out.csv"

                container_model_path = f"/host_dir/{model_path.relative_to(Path.cwd())}"
                container_property_path = f"/host_dir/{property_full_path.relative_to(Path.cwd())}"

                print(property_full_path, container_model_path, container_property_path)

                prepare_command = [
                    "docker", "exec", container_name, "bash",
                    "-c",
                    f'source /root/miniconda3/bin/activate marabou && cd ~/marabou/vnncomp && bash prepare_instance.sh v1 {log_file_basename.name} "{container_model_path}" "{container_property_path}"'
                ]

                run_command = [
                    "docker", "exec", container_name, "bash",
                    "-c",
                    f'source /root/miniconda3/bin/activate marabou && cd ~/marabou/vnncomp && bash run_single_instance.sh v1 . {log_file_basename.name} "{container_model_path}" "{container_property_path}" {timeout} {output_path} counterexample'
                ]

                with log_file.open("a") as lf:
                    print(f"Executing: {' '.join(prepare_command)}\n")
                    lf.write(f"Executing: {' '.join(prepare_command)}\n\n")

                    prepare_result = subprocess.run(
                        prepare_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    lf.write(prepare_result.stdout)
                    
                    print(f"Executing: {' '.join(run_command)}\n")
                    lf.write(f"Executing: {' '.join(run_command)}\n\n")
                    run_result = subprocess.run(
                        run_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    lf.write(run_result.stdout)
            
            parsed_result = parse_marabou_log(log_file)
        else:
            print("Use one of ['abcrown', 'neuralsat', 'pyrat', 'marabou']")
            exit(1)
        result_dir = Path("results")
        result_dir.mkdir(parents=True, exist_ok=True)
        save_parsed_result(
            parsed_result, result_dir / log_file_basename.with_suffix(".output"), args
        )

    except Exception as e:
        with log_file.open("a") as lf:
            lf.write(f"An error occurred: {e}\n")
        print(f"An error occurred: {e}")


def main():
    get_args()
    for verifier in args.verifier:
        if args.model_dir is not None:
            run_verifier(
                verifier=verifier,
                config_path=args.model_dir,
                timeout=args.timeout,
                drop_rate=args.drop_rate,
                perturb_alpha=args.perturb_alpha,
                split_type=args.split_type,
                eps=args.eps,
                attack_it=args.attack_it,
                container_name=args.container_name,
                pgd_steps=args.pgd_steps,
            )

        elif args.model_folder is not None:
            for subdir in os.listdir(args.model_folder):
                subdir_path = os.path.join(args.model_folder, subdir)
                if os.path.isdir(subdir_path):
                    run_verifier(
                        verifier=verifier,
                        config_path=subdir_path,
                        timeout=args.timeout,
                        drop_rate=args.drop_rate,
                        perturb_alpha=args.perturb_alpha,
                        split_type=args.split_type,
                        eps=args.eps,
                        attack_it=args.attack_it,
                        container_name=args.container_name,
                        pgd_steps=args.pgd_steps,
                    )
        else:
            pass


if __name__ == "__main__":
    main()
