import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from soutnance_1.config import OUTPUT_DIR, RANDOM_STATE


def train_model():
    X_train, y_train = joblib.load(OUTPUT_DIR / "train.joblib")
    preprocessor = joblib.load(OUTPUT_DIR / "preprocessor.joblib")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE
        ),
    }

    param_spaces = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"]
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__class_weight": [None, "balanced"]
        },
        "KNN": {
            "classifier__n_neighbors": [3, 5, 9, 15],
            "classifier__weights": ["uniform", "distance"]
        }
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []
    best_score = -1
    best_model = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", model)
        ])

        search = GridSearchCV(
            pipeline,
            param_spaces[name],
            cv=cv,
            scoring="f1_macro",
            refit=True,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        results.append({
            "Model": name,
            "F1 Macro": search.best_score_,
            "Best Parameters": search.best_params_
        })

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_

    results = pd.DataFrame(results).sort_values(
        "F1 Macro", ascending=False
    )

    results.to_csv(OUTPUT_DIR / "training_results.csv", index=False)
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")

    print(results)
    return best_model


if __name__ == "__main__":
    train_model()
