import pandas as pd
import json
from pathlib import Path

CSV_PATH = "bittext_cs.csv"   # Kaggle file-Q&A

def main():
    # 1. Read the Kaggle CSV
    df = pd.read_csv(CSV_PATH)
    print("Columns in CSV:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    # We assumed these columns based on dataset description:
    # - instruction: user request
    # - response: assistant answer
    # - category, intent: metadata
    required_cols = ["instruction", "response"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV")

    # 2. Creating test set for evaluation 
    df_test = df.sample(n=80, random_state=42) if len(df) > 80 else df.copy()

    test_path = Path("data/qa_test.jsonl")
    test_path.parent.mkdir(exist_ok=True)

    with test_path.open("w", encoding="utf-8") as f:
        for i, row in df_test.iterrows():
            obj = {
                "id": int(i),
                "question": str(row["instruction"]),
                "reference_answer": str(row["response"]),
                "category": str(row.get("category", "")),
                "intent": str(row.get("intent", "")),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df_test)} examples to {test_path}")

    # 3. Creating train set for tuning (bit larger subset)
    n_train = min(300, len(df))
    df_train = df.sample(n=n_train, random_state=7)

    train_tune_path = Path("data/qa_train_tune.jsonl")

    with train_tune_path.open("w", encoding="utf-8") as f:
        for _, row in df_train.iterrows():
            obj = {
                "input_text": str(row["instruction"]),
                "output_text": str(row["response"]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df_train)} examples to {train_tune_path}")


if __name__ == "__main__":
    main()
