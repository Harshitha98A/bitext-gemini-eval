import json
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from models_gemini import call_model
from metrics import score_example   #our file
DATA_PATH = Path("data/qa_test.jsonl")


def load_dataset(path: Path) -> List[Dict]:
    """
    Load the Q&A test dataset from a JSONL file.
    Each line looks like:
      {"id": ..., "question": "...", "reference_answer": "...", ...}
    """
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    #  only using first 5 examples for now
    examples = examples[:5]
    return examples


'''def eval_model_on_dataset(model_name: str, dataset: List[Dict]) -> Dict[str, float]:
    """
    Run one model across the entire dataset and compute average metrics.
    """
    scores = []
    for ex in tqdm(dataset, desc=f"Evaluating {model_name}"):
        q = ex["question"]
        ref = ex["reference_answer"]

        pred = call_model(model_name, q)
        s = score_example(pred, ref)
        scores.append(s)

    avg_exact = sum(s["exact_match"] for s in scores) / len(scores)
    avg_f1 = sum(s["f1"] for s in scores) / len(scores)

    return {
        "avg_exact_match": avg_exact,
        "avg_f1": avg_f1,
        "n": len(scores),
    }'''

def eval_model_on_dataset(model_name: str, dataset: List[Dict]) -> Dict[str, float]:
    """
    Run one model across the entire dataset and compute average metrics.
    """
    scores = []
    #change: add enumerate() so we get idx (0, 1, 2, ...)
    for idx, ex in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        q = ex["question"]
        ref = ex["reference_answer"]

        pred = call_model(model_name, q)

        '''# print the first example to inspect
        if idx == 0:
            print("\n--- SAMPLE CHECK ---")
            print("Question:         ", q)
            print("Reference answer: ", ref)
            print("Model answer:     ", pred)
            print("--------------------\n")'''

        s = score_example(pred, ref)
        scores.append(s)

    avg_exact = sum(s["exact_match"] for s in scores) / len(scores)
    avg_f1 = sum(s["f1"] for s in scores) / len(scores)

    return {
        "avg_exact_match": avg_exact,
        "avg_f1": avg_f1,
        "n": len(scores),
    }



def main():
    dataset = load_dataset(DATA_PATH)

    results: Dict[str, Dict[str, float]] = {}

    # For now we compare the two prompt-based Gemini configs
    for model_name in ["gemini_alt"]:
        res = eval_model_on_dataset(model_name, dataset)
        results[model_name] = res

    print("\n=== Evaluation Summary ===")
    for model_name, res in results.items():
        print(
            f"{model_name:12s} | "
            f"ExactMatch: {res['avg_exact_match']:.3f} | "
            f"F1: {res['avg_f1']:.3f} | "
            f"N={res['n']}"
        )


if __name__ == "__main__":
    main()
