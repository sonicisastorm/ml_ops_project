import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def main():
    # ====== EDA / Data Loading ======
    df = pd.read_parquet("multisim_dataset.parquet")  # Change path if needed
    print("✅ Dataset loaded. Shape:", df.shape)
    print("📊 Dataset preview:\n", df.head())

    # ====== Basic Data Cleaning ======
    # Drop duplicates if any
    df.drop_duplicates(inplace=True)

    # Remove completely empty columns (if present)
    df.dropna(axis=1, how="all", inplace=True)

    # ====== Feature Engineering 1 ======
    # Example: convert datetime columns to features
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
    df.drop(columns=datetime_cols, inplace=True)

    # ====== Feature Engineering 2 ======
    # Example: simple interaction feature
    num_cols_for_interactions = df.select_dtypes(include=["number"]).columns
    if len(num_cols_for_interactions) >= 2:
        df["feature_sum"] = df[num_cols_for_interactions[0]] + df[num_cols_for_interactions[1]]

    # ====== Split Features & Target ======
    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    # ====== Preprocessing Pipelines ======
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    # ====== Model Definition ======
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    # ====== Full Pipeline ======
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # ====== Train-Test Split ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ====== Train Model ======
    clf.fit(X_train, y_train)

    # ====== Evaluation ======
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    # ====== Save Model ======
    joblib.dump(clf, "models/model.pkl")
    print("✅ Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
