import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# 1. Load dataset
df = pd.read_parquet("multisim_dataset.parquet")  # Change path if needed

# 2. Split into features/target
X = df.drop("target", axis=1)
y = df["target"]

# 3. Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# 4. Preprocessing pipelines
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

# 5. Define model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# 6. Full pipeline
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 8. Fit model
clf.fit(X_train, y_train)

# 9. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

# 10. Save trained pipeline
joblib.dump(clf, "models/model.pkl")
print("✅ Model saved to models/model.pkl")
