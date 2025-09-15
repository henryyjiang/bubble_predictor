import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import joblib
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("dataset_2020.csv")

target_col = "Bubble_Risk"
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

base_models = [
    ("rf", RandomForestRegressor(random_state=42)),
    ("gbr", GradientBoostingRegressor(random_state=42))
]

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=GradientBoostingRegressor(random_state=42),
    passthrough=True
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", stacking_model)
])

param_grid = {
    "model__rf__n_estimators": [50, 100],
    "model__gbr__n_estimators": [50, 100],
    "model__final_estimator__n_estimators": [50, 100]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

joblib.dump(grid_search.best_estimator_, "bubble_risk_model.pkl")

y_pred = grid_search.predict(X_test)
print("Predictions on test set:", y_pred[:10])

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.4f}, Test R2: {r2:.4f}")
