from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ml" / "data" / "raw"
MODEL_DIR = ROOT / "ml" / "models"
REPORT_DIR = ROOT / "reports"

DATASETS = [
    "amazon_cells_labelled.txt",
    "imdb_labelled.txt",
    "yelp_labelled.txt",
]


def load_dataset():
    records = []

    for filename in DATASETS:
        matches = list(DATA_DIR.rglob(filename))
        if not matches:
            raise FileNotFoundError(
                f"Dataset file not found: {filename} inside {DATA_DIR}"
            )

        with matches[0].open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.rsplit("\t", 1)
                if len(parts) != 2:
                    continue

                text, label = parts
                records.append({
                    "text": text.strip(),
                    "label": int(label),
                    "source": filename.replace("_labelled.txt", ""),
                })

    df = pd.DataFrame(records)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "label"])
    return df


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    if df["label"].nunique() != 2:
        raise ValueError("Expected binary sentiment labels: 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    print("Training MEGAN AI sentiment classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)
    report = classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )

    joblib.dump(model, MODEL_DIR / "sentiment_pipeline.joblib")

    metrics = {
        "project": "MEGAN AI",
        "task": "Binary sentiment classification",
        "dataset": "UCI Sentiment Labelled Sentences",
        "total_samples_after_cleaning": int(len(df)),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "split": "80% train / 20% test",
        "random_state": 42,
        "accuracy": round(float(accuracy), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
        "source_counts": {
            str(key): int(value)
            for key, value in df["source"].value_counts().items()
        },
    }

    with (REPORT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nTraining completed.")
    print(f"Samples used: {len(df)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1 score: {f1:.4f}")
    print("\nConfusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"\nModel saved to: {MODEL_DIR / 'sentiment_pipeline.joblib'}")
    print(f"Metrics saved to: {REPORT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
