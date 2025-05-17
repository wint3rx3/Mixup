import json

def extract_failed_cases(input_path="data/results.jsonl", output_path="data/errors.jsonl"):
    with open(input_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    failed = [
        item for item in lines
        if item.get("target") and item.get("prediction") != item.get("target")
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in failed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
