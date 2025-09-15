import pandas as pd
import joblib

test_df = pd.read_csv("dataset_past_month.csv")
model = joblib.load("bubble_risk_model.pkl")

X_test = test_df

predictions = model.predict(X_test)
test_df["Predicted_Bubble_Risk"] = predictions

test_df.to_csv("predicted_bubble_risk.csv", index=False)
print("Predictions saved to predicted_bubble_risk.csv")
print(test_df.head())
