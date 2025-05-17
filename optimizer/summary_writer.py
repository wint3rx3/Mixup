import csv
import json

def write_summary_csv(input_path="data/memory.jsonl", output_path="data/summary.csv"):
    with open(input_path, encoding="utf-8") as f:
        memory = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=memory[0].keys())
        writer.writeheader()
        for row in memory:
            writer.writerow(row)
