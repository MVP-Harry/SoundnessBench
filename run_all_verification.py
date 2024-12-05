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
    parser.add_argument("--config_dir", default="./config.yaml", type=str)
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

def parse_abcrown_log(log_data):
    safe_incomplete_match = re.search(
        r"safe-incomplete \(total \d+\), index: \[([0-9, ]+)\]", log_data
    )
    if safe_incomplete_match:
        return "unsat"

    safe_match = re.search(r"safe \(total \d+\), index: \[([0-9, ]+)\]", log_data)
    if safe_match:
        return "unsat"

    unknown_match = re.search(
        r"unknown \(total \d+\), index: \[([0-9, ]+)\]", log_data
    )
    if unknown_match:
        return "unknown"

    unsafe_match = re.search(
        r"unsafe-pgd \(total \d+\), index: \[([0-9, ]+)\]", log_data
    )
    if unsafe_match:
        return "sat"
            
    unsafe_match_bab = re.search(
        r"unsafe-bab \(total \d+\), index: \[([0-9, ]+)\]", log_data
    )
    if unsafe_match_bab:
        return "sat"

    return None


def build_command(base_cmd, **kwargs):
    cmd = base_cmd.copy()
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([key, str(value)])
    return cmd


def run_verifier(
    verifier: str,
    bench_dir: str,
    config_dir=None,
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
    bench_dir = Path(bench_dir).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_basename = bench_dir.name
    result_csv_dir = Path("results")
    result_csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_basename = Path(f"{verifier}-{config_basename}-{timestamp}.log")
    log_file = log_dir / log_file_basename
 
    with log_file.open("a") as lf:
        lf.write(f"run {verifier} in {bench_dir}\n\n")

    try:
        if verifier == "abcrown":
            if split_type == "hidden":
                trans_type = "activation"
            else:
                trans_type = split_type
            results_csv = Path(result_csv_dir / f"{verifier}_{trans_type}_results.csv")
            if not results_csv.exists():
                with results_csv.open("w", newline="") as f:
                    writer = csv.writer(f)

            original_dir = Path.cwd()
            env = os.environ.copy()
            print(str(bench_dir))
            env["bench_dir"] = str(bench_dir)
            env_name = "alpha-beta-crown"
            target_dir = Path(
                "alpha-beta-CROWN_vnncomp2024/complete_verifier"
            ).resolve()
            os.chdir(target_dir)
            
            instances_file = bench_dir / "instances.csv"
            with instances_file.open("r") as f:
                reader = csv.reader(f)
                instances = [row for row in reader]
            print(str(Path(config_dir)))
            for i, (model_path, property_path, _) in enumerate(instances):
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
                    str(Path(config_dir).resolve()),
                    "--root_path",
                    str(bench_dir),
                    "--onnx_path",
                    str(bench_dir / "model.onnx"),
                    "--start",
                    str(i),
                    "--end",
                    str(i + 1),
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
                    
                for line in result.stdout.splitlines():
                    result_match = parse_abcrown_log(line)
                    if result_match is not None:
                        break
                for line in result.stdout.splitlines():
                    time = re.search(r"mean time for ALL instances \(total \d+\):([\d.]+)", line)
                    if time is not None:
                        break
                
                if result_match is None:
                    current_result = "Unknown"
                else:
                    current_result = result_match
                
                if time is not None:
                    time = time.group(1).lower()
                    
                with results_csv.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["SoundnessBench", os.path.abspath(model_path), os.path.abspath(property_path), time, current_result, time])

        elif verifier == "neuralsat":
            env_name = "neuralsat"
            model_path = bench_dir / "model.onnx"
            instances_file = bench_dir / "instances.csv"
            if split_type == "hidden":
                trans_type = "activation"
            else:
                trans_type = split_type
            results_csv = Path(result_csv_dir / f"{verifier}_{trans_type}_results.csv")
            if not results_csv.exists():
                with results_csv.open("w", newline="") as f:
                    writer = csv.writer(f)
                
            env = os.environ.copy()
            if not instances_file.exists():
                print(f"Instances file not found: {instances_file}")
                return

            with instances_file.open("r") as f:
                reader = csv.reader(f)
                instances = [row for row in reader]

            for _, property_path, _ in instances:
                property_full_path = bench_dir / property_path
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
                    
                for line in result.stdout.splitlines():
                    result_match = re.search(r"Result: (\w+)", line)
                    if result_match is not None:
                        break
                for line in result.stdout.splitlines():
                    time = re.search(r"Runtime: ([\d.]+)", line)
                    if time is not None:
                        break
                    
                if result_match is None:
                    current_result = "Unknown"
                else:
                    current_result = result_match.group(1).lower()
                    
                if time is not None:
                    time = time.group(1).lower()
                    
                with results_csv.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["SoundnessBench", os.path.abspath(model_path), os.path.abspath(property_path), time, current_result, time])

        elif verifier == "pyrat":
            env_name = "pyrat"
            model_path = bench_dir / "model.onnx"
            instances_file = bench_dir / "instances.csv"

            results_csv = Path(result_csv_dir / f"{verifier}_results.csv")
            if not results_csv.exists():
                with results_csv.open("w", newline="") as f:
                    writer = csv.writer(f)
            
            if not instances_file.exists():
                print(f"Instances file not found: {instances_file}")
                return

            with instances_file.open("r") as f:
                reader = csv.reader(f)
                instances = [row for row in reader]

            for _, property_path, _ in instances:
                property_full_path = bench_dir / property_path
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
                
                for line in result.stdout.splitlines():
                    result_match = re.search(
                        r"Result = (\w+), Time = ([\d.]+) s, Safe space = ([\d.]+) %, number of analysis = (\d+)",
                        line,
                    )
                    if result_match is not None:
                        break
                    
                for line in result.stdout.splitlines():
                    time = re.search(r"Time = ([\d.]+ s)", line)
                    if time is not None:
                        break
                    
                if result_match is None:
                    current_result = "Unknown"
                else:
                    current_result = result_match.group(1).lower()
                
                if time is not None:
                    time = time.group(1).lower()
                    
                with results_csv.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["SoundnessBench", os.path.abspath(model_path), os.path.abspath(property_path), time, current_result, time])

        elif verifier == "marabou":
            model_path = bench_dir / "model.onnx"
            instances_file = bench_dir / "instances.csv"

            results_csv = Path(result_csv_dir / f"{verifier}_{container_name}_results.csv")
            if not results_csv.exists():
                with results_csv.open("w", newline="") as f:
                    writer = csv.writer(f)
            
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
                property_full_path = Path(bench_dir) / property_path
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
                    
                for line in result.stdout.splitlines():
                    result_match = re.search(r"Appending result '(\w+)' to csv file", line)
                    if result_match is not None:
                        break
                
                for line in result.stdout.splitlines():
                    time = re.search(r"Runtime = ([\d.]+)", line)
                    if time is not None:
                        break
                    
                if result_match is None:
                    current_result = "Unknown"
                else:
                    current_result = result_match.group(1).lower()
                    
                with results_csv.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["SoundnessBench", os.path.abspath(model_path), os.path.abspath(property_path), time, current_result, time])
                    
        else:
            print("Use one of ['abcrown', 'neuralsat', 'pyrat', 'marabou']")
            exit(1)

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
                config_dir=Path(args.config_dir).resolve(),
                bench_dir=args.model_dir,
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
                        config_dir=Path(args.config_dir).resolve(),
                        bench_dir=subdir_path,
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
