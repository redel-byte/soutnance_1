import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, clone
from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
    OrdinalEncoder,
    OneHotEncoder
)

from soutnance_1.config import OUTPUT_DIR, Z_SCORE_THRESHOLD


def preprocess_data():
    X_train, _ = joblib.load(OUTPUT_DIR / "train.joblib")

    ordinal_categories = {
        # "Year_of_Study": [1, 2, 3, 4, 5],
        "Physical_Activity": ["Low", "Moderate", "High"],
        "Diet_Quality": ["Poor", "Average", "Good"],
    }

    ordinal = list(ordinal_categories)
    nominal = ["Gender", "Part_Time_Job", "Previous_Mental_Health_History"]
    numerical = X_train.drop(
        columns=["Year_of_Study" , *nominal]
    ).select_dtypes(include="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            FunctionTransformer(
                np.clip,
                kw_args={
                    "a_min": -Z_SCORE_THRESHOLD,
                    "a_max": Z_SCORE_THRESHOLD
                },
                feature_names_out="one-to-one"
            )
        ), numerical),

        ("ordinal", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(
                categories=list(ordinal_categories.values()),
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        ), ordinal),

        ("nominal", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ), nominal)
    ])

    joblib.dump(clone(preprocessor), OUTPUT_DIR / "preprocessor.joblib")


if __name__ == "__main__":
    preprocess_data()
