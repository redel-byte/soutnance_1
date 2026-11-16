import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from soutnance_1.config import DATA_PATH, OUTPUT_DIR, TARGET, RANDOM_STATE


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET, "Student_ID"])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    joblib.dump((X_train, y_train), OUTPUT_DIR / "train.joblib")
    joblib.dump((X_test, y_test), OUTPUT_DIR / "test.joblib")


if __name__ == "__main__":
    prepare_data()
