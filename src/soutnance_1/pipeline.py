import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
    StratifiedKFold
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, OrdinalEncoder

DATA_PATH = Path(os.getenv("STUDENT_DATA_PATH", "data/raw/student-mental.csv"))
OUTPUT_DIR = Path(os.getenv("STUDENT_RESULTS_ROOT", "artifacts"))
TARGET = "Mental_Health_Risk"
RANDOM_STATE = 22
Z_SCORE_THRESHOLD = 3


# 1. Prepare data
def prepare_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET, "Student_ID"])
    y = df[TARGET]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# 2. Create preprocessing pipeline
def preprocess_data(X):
    # numeric = X.select_dtypes(include="number").columns
    # categorical = X.select_dtypes(exclude="number").columns
    ordinal_categories = {
    "Year_of_Study": [1, 2, 3, 4, 5],
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
    }
    ordinal_columns = list(ordinal_categories)
    nominal_columns = ["Gender", "Part_Time_Job", "Previous_Mental_Health_History"]
    numerical_columns = X.drop(columns=ordinal_columns).select_dtypes(include="number").columns
    return ColumnTransformer(transformers=[
        ("numeric", make_pipeline(
            StandardScaler(),FunctionTransformer(np.clip,kw_args={"a_min": -Z_SCORE_THRESHOLD, "a_max": Z_SCORE_THRESHOLD},feature_names_out="one-to-one"),
            SimpleImputer(strategy="median")), numerical_columns),
        ("ordinal", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(categories=list(ordinal_categories.values()),handle_unknown="use_encoded_value",unknown_value=-1)), ordinal_columns),
        ("nominal", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), nominal_columns)
        ])


# 3. Compare and optimize models
def train_model(X_train, y_train, tuning=False):
    preprocessor = preprocess_data(X_train)

    scoring = "f1_macro"
    results = []
    searches = {}
    tuning_results = []

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
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
        "classifier__weights": ["uniform", "distance"],
    }}
    
    cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE)

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])
        if tuning:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_spaces[name],
                cv=cv,
                scoring=scoring,
                refit=True,
                n_jobs=-1,
                error_score="raise"
            )
            search.fit(X_train,y_train)
            searches[name] = search
            tuning_results.append({
                "Model": name,
                "F1 Macro": search.best_score_,
                "Best Parameters": search.best_params_,
                "Best Estimator": search.best_estimator_
            })
            
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        results.append({"Model": name, "F1 Macro": scores.mean(),"std":scores.std()})
    return pd.DataFrame(results).sort_values("F1 Macro", ascending=False ,ignore_index=True) if not tuning else pd.DataFrame(tuning_results).sort_values("F1 Macro", ascending=False, ignore_index=True)

# 4. Evaluate on unseen test data
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Confusion matrix on the held-out test set")
    plt.show()


# Run complete pipeline
def main():
    X_train, X_test, y_train, y_test = prepare_data()

    model = train_model(X_train, y_train, tuning=True).loc[0, "Best Estimator"]

    print(model,type(model))
    
    evaluate_model(model, X_test, y_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "best_model.joblib")


if __name__ == "__main__":
    main()
