import argparse
import os
from pathlib import Path
import csv
from verifier import VerifierABCROWN, VerifierNeuralSAT, VerifierPyRAT, VerifierMarabou
from typing import Optional

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, type=str)
parser.add_argument("--verifier", required=True, nargs='+', type=str)
parser.add_argument("--select_instance", nargs='+', type=int, default=None)
parser.add_argument("--override_timeout", type=float)
parser.add_argument("--backup_dir", type=str, default=None, help="If it is provided, copy the reults to this directory.")
parser.add_argument('--output_suffix', type=str, default='')
args = parser.parse_args()


def run_verifier(verifier: str, bench_dir: str, select_instance: Optional[list] = None,
                 override_timeout: Optional[int] = None, backup_dir: Optional[str] = None):
    bench_dir = Path(bench_dir).resolve()

    output_suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    results_dir = bench_dir / f"results{output_suffix}"
    results_dir.mkdir(exist_ok="True")

    if backup_dir is not None:
        backup_dir = Path(f"{backup_dir}{output_suffix}").resolve()
        backup_dir.mkdir(exist_ok=True)
        backup_dir = backup_dir / bench_dir.name
        backup_dir.mkdir(exist_ok=True)

    assert verifier in ["abcrown_act", "abcrown_input", "neuralsat_act", "neuralsat_input", "pyrat", "marabou_vnncomp_2023", "marabou_vnncomp_2024"], \
        f"Verifier {verifier} not supported. Supported verifiers: abcrown_act, abcrown_input, " \
        f"neuralsat_act, neuralsat_input, pyrat, marabou_vnncomp_2023, marabou_vnncomp_2024"

    if "abcrown" in verifier:
        split_type = "hidden" if "act" in verifier else "input"
        verifier = VerifierABCROWN(
            name=verifier,
            bench_dir=bench_dir, results_dir=results_dir, split_type=split_type,
            backup_dir=backup_dir,
        )
    elif "neuralsat" in verifier:
        split_type = "hidden" if "act" in verifier else "input"
        verifier = VerifierNeuralSAT(
            name=verifier,
            bench_dir=bench_dir, results_dir=results_dir, split_type=split_type,
            backup_dir=backup_dir,
        )
    elif verifier == "pyrat":
        verifier = VerifierPyRAT(
            name="pyrat",
            bench_dir=bench_dir, results_dir=results_dir, backup_dir=backup_dir
            )
    elif "marabou" in verifier:
        verifier = VerifierMarabou(
            name=verifier, bench_dir=bench_dir, results_dir=results_dir, container_name=verifier,
            backup_dir=backup_dir)
    else:
        raise NameError(verifier)

    result_file = results_dir / f"{verifier.name}.csv"
    with result_file.open("w", newline="") as _:
        pass

    instances_file = bench_dir / f"instances{output_suffix}.csv"
    with instances_file.open("r") as f:
        reader = csv.reader(f)
        instances = [row for row in reader]
        try:
            for i, (model_path, property_path, timeout) in enumerate(instances):
                if select_instance is not None and i not in select_instance:
                    continue
                if override_timeout:
                    timeout = override_timeout
                ret, time = verifier.run(
                    i,
                    model_path=bench_dir / model_path,
                    property_path=bench_dir / property_path,
                    timeout=float(timeout),
                )
                with result_file.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        bench_dir,
                        model_path,
                        property_path,
                        ret, time
                    ])
                    print(
                        f"Model: {model_path}, Property: {property_path}, Result: {ret}, Time: {time}")
                    print(f"Results saved to {result_file}")
        except Exception as e:
            print("Error:", e)
            with (verifier.results_dir / f"{verifier.name}.err").open("a") as lf:
                lf.write(f"An error occurred: {e}\n")
            raise

    if backup_dir is not None:
        os.system(f"cp {result_file} {backup_dir}")


def main():
    for verifier in args.verifier:
        run_verifier(verifier=verifier, bench_dir=args.model_dir,
                        select_instance=args.select_instance,
                        override_timeout=args.override_timeout,
                        backup_dir=args.backup_dir)


if __name__ == "__main__":
    main()
