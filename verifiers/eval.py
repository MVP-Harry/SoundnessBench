import os
import argparse
import torch
import csv

def is_unsat(result):
    return any(keyword in result for keyword in ['unsat', 'safe', 'true'])

def is_sat(result):
    return any(keyword in result for keyword in ['sat', 'unsafe', 'false'])

def count_ratio(subset, check_fn):
    if len(subset) == 0:
        return 0.0
    return round(sum(1 for row in subset if check_fn(row[3])) / len(subset) * 100, 2)

def analyze_csv(file_path, model_name, num_clean, verifier_name):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader if len(row) == 5]

    num_unverifiable = len(rows) - num_clean
    unverifiable_instance = rows[:num_unverifiable]
    clean_instance = rows[num_unverifiable:]

    clean_instance_unsat_ratio = count_ratio(clean_instance, is_unsat)
    clean_instance_sat_ratio = count_ratio(clean_instance, is_sat)
    unverifiable_instance_unsat_ratio = count_ratio(unverifiable_instance, is_unsat)
    unverifiable_instance_sat_ratio = count_ratio(unverifiable_instance, is_sat)
    has_unsound = any(is_unsat(row[3]) for row in unverifiable_instance)

    return [
        model_name,
        verifier_name,
        clean_instance_unsat_ratio,
        clean_instance_sat_ratio,
        unverifiable_instance_unsat_ratio,
        unverifiable_instance_sat_ratio,
        str(has_unsound)
    ]

def main():
    parser = argparse.ArgumentParser(description="Process VNN data and calculate statistics.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--csv_file", type=str, default=None, help="Path to a specific verifier result .csv file")
    parser.add_argument('--num_examples', type=int, default=10, help='It should be the same as num_examples during generation and training')
    args = parser.parse_args()

    model_name = os.path.basename(args.model_dir)
    result_dir = os.path.join(args.model_dir, 'results')
    results = []

    num_clean = args.num_examples

    if args.csv_file:
        if not os.path.exists(args.csv_file):
            raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
        verifier_name = os.path.splitext(os.path.basename(args.csv_file))[0]
        result = analyze_csv(args.csv_file, model_name, num_clean, verifier_name)
        results.append(result)
        output_path = os.path.join(result_dir, f"{verifier_name}_stat.csv")
    else:
        verifier_names = [
            'abcrown_act', 'abcrown_input', 'neuralsat_act', 'neuralsat_input',
            'pyrat', 'marabou_vnncomp_2023', 'marabou_vnncomp_2024'
        ]
        for verifier_name in verifier_names:
            file_path = os.path.join(result_dir, f"{verifier_name}.csv")
            if not os.path.exists(file_path):
                continue
            result = analyze_csv(file_path, model_name, num_clean, verifier_name)
            results.append(result)
        output_path = os.path.join(result_dir, 'stat.csv')

    # Write output CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'model_name',
            'verifier',
            'clean_instance_verified_ratio',
            'clean_instance_falsified_ratio',
            'unverifiable_instance_verified_ratio',
            'unverifiable_instance_falsified_ratio',
            'has_unsound'
        ])
        writer.writerows(results)

    print(f"Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
