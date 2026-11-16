import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from soutnance_1.config import OUTPUT_DIR


def evaluate_model():
    X_test, y_test = joblib.load(OUTPUT_DIR / "test.joblib")
    model = joblib.load(OUTPUT_DIR / "best_model.joblib")

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)

    (OUTPUT_DIR / "classification_report.txt").write_text(report)

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    evaluate_model()
