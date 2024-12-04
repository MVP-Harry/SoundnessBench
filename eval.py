import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Process VNN data and calculate statistics.")
parser.add_argument("file_path", type=str, help="Path to the input CSV file.")
parser.add_argument("output_path", type=str, help="Path to save the output CSV file.")
args = parser.parse_args()

columns = ['benchmark', 'model', 'property', 'prepare_time', 'result', 'time']
data = pd.read_csv(args.file_path, header=None, names=columns)
grouped = data.groupby(['benchmark', 'model'])

results = []

for (_, model), group in grouped:
    group = group.sort_index(ascending=False)
    clean_instance = group.iloc[:10]
    unverifiable_instance = group.iloc[10:]
    if len(group) < 10:
        continue
    
    clean_instance_unsat_ratio = round(
                                    (clean_instance['result'].isin(['unsat', 'safe', 'true'])).mean() * 100, 
                                    2
                                )
    clean_instance_sat_ratio = round(
                                    (clean_instance['result'].isin(['sat', 'unsafe', 'false'])).mean() * 100, 
                                    2
                                )

    unverifiable_instance_unsat_ratio = round(
                                    (clean_instance['result'].isin(['unsat', 'safe', 'true'])).mean() * 100, 
                                    2
                                )
    unverifiable_instance_sat_ratio = round(
                                    (clean_instance['result'].isin(['sat', 'unsafe', 'false'])).mean() * 100, 
                                    2
                                )

    has_unsound = (unverifiable_instance['result'].isin(['unsat', 'safe', 'true'])).any()

    results.append({
        'model_name': model,
        'clean_instance_verified_ratio': clean_instance_unsat_ratio,
        'clean_instance_falsified_ratio': clean_instance_sat_ratio,
        'unverifiable_instance_verified_ratio': unverifiable_instance_unsat_ratio,
        'unverifiable_instance_falsified_ratio': unverifiable_instance_sat_ratio,
        'has_unsound': has_unsound
    })

results_df = pd.DataFrame(results)

results_df.to_csv(args.output_path, index=False)
print(f"Analysis complete. Results saved to {args.output_path}")
