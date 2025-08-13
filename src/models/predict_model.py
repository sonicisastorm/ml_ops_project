import joblib
import pandas as pd

# 1. Load saved model
clf = joblib.load("models/model.pkl")

# 2. Load new data (same structure as training features)
df_new = pd.read_parquet("new_data.parquet")  # Change file name as needed

# 3. Predict
predictions = clf.predict(df_new)
print("Predictions:", predictions)
