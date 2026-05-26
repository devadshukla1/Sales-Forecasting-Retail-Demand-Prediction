import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# Page setup
st.set_page_config(
    page_title="Retail Demand Prediction",
    page_icon="📈",
    layout="wide"
)

# File path
DATA_PATH = "data/Retail_and_wherehouse_Sale.csv"

# Load model files
model = joblib.load("models/sales_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
target_col = joblib.load("models/target_col.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# App title
st.title("📈 Sales Forecasting / Retail Demand Prediction")

st.write(
    "This dashboard predicts retail demand using a simple machine learning model. "
    "It also shows sales trends, seasonality, actual vs predicted comparison, and business insights."
)

# Sidebar
st.sidebar.header("Enter Details for Prediction")

input_values = {}

for col in df.columns:
    if col == target_col:
        continue

    if col == "YEAR":
        input_values[col] = st.sidebar.number_input(
            "YEAR",
            min_value=2000,
            max_value=2026,
            value=2020,
            step=1
        )

    elif col == "MONTH":
        input_values[col] = st.sidebar.number_input(
            "MONTH",
            min_value=1,
            max_value=12,
            value=6,
            step=1
        )

    elif df[col].dtype == "object":
        unique_values = sorted(df[col].dropna().unique().tolist())
        input_values[col] = st.sidebar.selectbox(col, unique_values)

    else:
        min_value = float(df[col].min())
        max_value = float(df[col].max())
        mean_value = float(df[col].mean())

        input_values[col] = st.sidebar.number_input(
            col,
            min_value=min_value,
            max_value=max_value,
            value=mean_value
        )

# Prediction input dataframe
input_df = pd.DataFrame([input_values])

input_encoded = pd.get_dummies(input_df, drop_first=True)

for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[feature_columns]

# Prediction
if st.sidebar.button("Predict Sales"):
    prediction = model.predict(input_encoded)[0]

    st.subheader("Forecast Result")
    st.metric("Predicted Retail Sales", f"{prediction:,.2f}")

    st.subheader("Business Recommendation")

    if prediction > df[target_col].mean():
        st.success("Predicted demand is above average. Keep enough stock ready.")
    else:
        st.info("Predicted demand is below average. Avoid overstocking.")

# Monthly trend
st.subheader("Monthly Sales Trend")

monthly_sales = df.groupby("MONTH")[target_col].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(monthly_sales["MONTH"], monthly_sales[target_col], marker="o")
ax1.set_xlabel("Month")
ax1.set_ylabel(target_col)
ax1.set_title("Monthly Sales Trend")
st.pyplot(fig1)

# Seasonality analysis
st.subheader("Seasonality Analysis")

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8]:
        return "Monsoon"
    else:
        return "Festival"

df["Season"] = df["MONTH"].apply(get_season)

season_sales = df.groupby("Season")[target_col].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.bar(season_sales["Season"], season_sales[target_col])
ax2.set_xlabel("Season")
ax2.set_ylabel(f"Average {target_col}")
ax2.set_title("Average Sales by Season")
st.pyplot(fig2)

# Actual vs predicted
st.subheader("Actual vs Predicted Sales")

model_df = df.copy()
model_df = model_df.dropna()

X_all = model_df.drop(target_col, axis=1)
X_all = pd.get_dummies(X_all, drop_first=True)

for col in feature_columns:
    if col not in X_all.columns:
        X_all[col] = 0

X_all = X_all[feature_columns]

model_df["Predicted Sales"] = model.predict(X_all)

sample_df = model_df.head(50)

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(sample_df.index, sample_df[target_col], label="Actual Sales", marker="o")
ax3.plot(sample_df.index, sample_df["Predicted Sales"], label="Predicted Sales", marker="o")
ax3.set_xlabel("Records")
ax3.set_ylabel(target_col)
ax3.set_title("Actual vs Predicted Sales")
ax3.legend()
st.pyplot(fig3)

# Business insights
st.subheader("Business Insights")

total_sales = df[target_col].sum()
average_sales = df[target_col].mean()
highest_sales = df[target_col].max()

best_month = monthly_sales.loc[monthly_sales[target_col].idxmax(), "MONTH"]
best_item_type = df.groupby("ITEM TYPE")[target_col].sum().idxmax()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Retail Sales", f"{total_sales:,.0f}")

with col2:
    st.metric("Average Retail Sales", f"{average_sales:,.0f}")

with col3:
    st.metric("Best Month", int(best_month))

st.write(f"- Best performing item type is **{best_item_type}**.")
st.write("- Monthly trend helps understand demand movement over time.")
st.write("- Seasonality analysis helps identify high-demand periods.")
st.write("- Actual vs predicted graph helps compare model output with real values.")
st.write("- Business can use this forecast to plan inventory and avoid overstocking.")

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head(50), use_container_width=True)