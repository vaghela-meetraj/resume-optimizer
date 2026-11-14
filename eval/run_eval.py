import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.nlp import NLPProcessor
from app.services.skills import normalize_skill


def evaluate(output_file="eval/results/baseline.json"):
    print("[*] Loading NLP Processor...")
    processor = NLPProcessor()

    labels_path = "eval/labels.json"
    if not os.path.exists(labels_path):
        print(f"[!] Error: {labels_path} not found.")
        sys.exit(1)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    file_paths = {}
    for filename in os.listdir("eval/data/jds"):
        file_paths[filename] = os.path.join("eval/data/jds", filename)
    for filename in os.listdir("eval/data/resumes"):
        file_paths[filename] = os.path.join("eval/data/resumes", filename)

    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("\n" + "=" * 90)
    print(f"{'Document':<26} | {'GT':<4} | {'Ext':<4} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<7} | {'F1':<6}")
    print("=" * 90)

    for doc_name, path in sorted(file_paths.items()):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        extracted_raw = processor.extract_skills_nlp(text)
        # Normalize extracted skills for consistent matching
        extracted = {normalize_skill(s).lower() for s in extracted_raw}

        # Normalize ground truth
        gt = {normalize_skill(s).lower() for s in labels.get(doc_name, [])}

        tp = len(extracted & gt)
        fp = len(extracted - gt)
        fn = len(gt - extracted)

        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        doc_result = {
            "document": doc_name,
            "ground_truth_count": len(gt),
            "extracted_count": len(extracted),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 1),
            "recall": round(recall, 1),
            "f1": round(f1, 1),
            "false_positives": sorted(list(extracted - gt)),
            "false_negatives": sorted(list(gt - extracted))
        }
        results.append(doc_result)

        print(f"{doc_name:<26} | {len(gt):<4} | {len(extracted):<4} | {tp:<4} | {fp:<4} | {fn:<4} | {precision:6.1f}%  | {recall:5.1f}% | {f1:5.1f}%")

    # Global Micro-F1 (standard MLOps metric)
    micro_precision = (total_tp / (total_tp + total_fp)) * 100 if (total_tp + total_fp) > 0 else 0.0
    micro_recall = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    macro_f1 = sum(r["f1"] for r in results) / len(results) if results else 0.0

    print("=" * 90)
    print(f"{'OVERALL (Micro-Average)':<26} | {total_tp + total_fn:<4} | {total_tp + total_fp:<4} | {total_tp:<4} | {total_fp:<4} | {total_fn:<4} | {micro_precision:6.1f}%  | {micro_recall:5.1f}% | {micro_f1:5.1f}%")
    print(f"{'OVERALL (Macro F1)':<26} | {'-':<4} | {'-':<4} | {'-':<4} | {'-':<4} | {'-':<4} | {'-':<9} | {'-':<7} | {macro_f1:5.1f}%")
    print("=" * 90 + "\n")

    summary = {
        "micro_precision": round(micro_precision, 1),
        "micro_recall": round(micro_recall, 1),
        "micro_f1": round(micro_f1, 1),
        "macro_f1": round(macro_f1, 1),
        "documents": results
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Evaluation results saved to {output_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval/results/baseline.json", help="Path to save results JSON")
    args = parser.parse_args()
    evaluate(args.output)
