import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
DATA_PATH = "data/Retail_and_wherehouse_Sale.csv"
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip()

print("Dataset loaded successfully")
print("Columns:", df.columns.tolist())

# Use RETAIL SALES as target
target_col = "RETAIL SALES"

# Keep only simple useful columns
required_columns = [
    "YEAR",
    "MONTH",
    "ITEM TYPE",
    "RETAIL TRANSFERS",
    "WAREHOUSE SALES",
    "RETAIL SALES"
]

df = df[required_columns]

# Remove missing values
df = df.dropna()

# Features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Convert ITEM TYPE into numbers
X = pd.get_dummies(X, columns=["ITEM TYPE"], drop_first=True)

# Save feature columns and target column
joblib.dump(list(X.columns), "models/feature_columns.pkl")
joblib.dump(target_col, "models/target_col.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model... please wait")

# Train model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed")

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nSales Forecasting Model")
print("-----------------------")
print(f"Target Column Used: {target_col}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, "models/sales_model.pkl")

print("\nModel saved successfully as models/sales_model.pkl")