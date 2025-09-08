import streamlit as st
import pandas as pd
import numpy as np

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI Financial Suite", layout="wide")

# =========================
# Custom CSS for Styling
# =========================
st.markdown(
    """
    <style>
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        h1 {
            color: #00BFFF;
            font-weight: bold;
            text-align: center;
            font-size: 2.5em;
        }
        h2 {
            color: #FFD700;
            text-align: center;
            font-size: 1.5em;
        }
        h3 {
            color: #FF69B4;
            font-size: 1.2em;
        }
        .stButton>button {
            background-color: #00BFFF;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1E90FF;
            transform: scale(1.05);
        }
        .stDownloadButton>button {
            background-color: #32CD32;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            transition: 0.3s;
        }
        .stDownloadButton>button:hover {
            background-color: #228B22;
            transform: scale(1.05);
        }
        .css-1d391kg {
            background: linear-gradient(180deg, #001F3F, #001122);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Dashboard Header
# =========================
st.markdown("<h1>💰 AI Financial Suite</h1>", unsafe_allow_html=True)
st.markdown("<h2>Your 4-in-1 AI Assistant for Smarter Workflows</h2>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("📊 Financial Tools")
menu = st.sidebar.radio(
    "Choose a module:",
    ["Accounts Reconciliation", "Cash Flow Forecasting", "Invoice Processing", "Expense Categorization"]
)

# =========================
# 1. Accounts Reconciliation
# =========================
if menu == "Accounts Reconciliation":
    st.markdown("<h3>📑 Accounts Reconciliation</h3>", unsafe_allow_html=True)
    st.write("Upload two CSV files (**Bank Statement** & **Internal Records**) to reconcile transactions automatically.")

    bank_file = st.file_uploader("Upload Bank Statement", type=["csv"], key="bank")
    internal_file = st.file_uploader("Upload Internal Records", type=["csv"], key="internal")

    if bank_file and internal_file:
        bank_df = pd.read_csv(bank_file)
        internal_df = pd.read_csv(internal_file)

        st.write("### 📘 Bank Statement")
        st.dataframe(bank_df)

        st.write("### 📗 Internal Records")
        st.dataframe(internal_df)

        unmatched = pd.concat([bank_df, internal_df]).drop_duplicates(keep=False)

        st.write("### 🔎 Unmatched Transactions")
        st.dataframe(unmatched)

        st.download_button("⬇️ Download Unmatched Records", unmatched.to_csv(index=False).encode("utf-8"), "unmatched.csv")

# =========================
# 2. Cash Flow Forecasting
# =========================
elif menu == "Cash Flow Forecasting":
    st.markdown("<h3>📈 Cash Flow Forecasting</h3>", unsafe_allow_html=True)
    st.write("Upload past transactions to **forecast your cash flow** for the next 6 months.")

    file = st.file_uploader("Upload Transactions CSV", type=["csv"], key="cashflow")

    if file:
        df = pd.read_csv(file)

        st.write("### 📘 Uploaded Data")
        st.dataframe(df)

        if "Amount" in df.columns:
            monthly_cashflow = df.groupby(df.index // 30)["Amount"].sum()
            forecast = monthly_cashflow.rolling(3).mean().shift(1).fillna(method="bfill")

            st.write("### 🔮 6-Month Forecast")
            forecast_df = pd.DataFrame({
                "Month": np.arange(1, len(forecast) + 1),
                "Forecasted Cash Flow": forecast.values
            })
            st.dataframe(forecast_df)

            st.download_button("⬇️ Download Forecast", forecast_df.to_csv(index=False).encode("utf-8"), "cashflow_forecast.csv")

# =========================
# 3. Invoice Processing
# =========================
elif menu == "Invoice Processing":
    st.markdown("<h3>🧾 Invoice Processing</h3>", unsafe_allow_html=True)
    st.write("Upload **invoices CSV file** to extract and organize payment details automatically.")

    file = st.file_uploader("Upload Invoices CSV", type=["csv"], key="invoice")

    if file:
        invoices = pd.read_csv(file)
        st.write("### 📘 Uploaded Invoices")
        st.dataframe(invoices)

        if {"Invoice ID", "Amount", "Status"}.issubset(invoices.columns):
            pending = invoices[invoices["Status"].str.lower() == "pending"]

            st.write("### ⏳ Pending Invoices")
            st.dataframe(pending)

            st.download_button("⬇️ Download Pending Invoices", pending.to_csv(index=False).encode("utf-8"), "pending_invoices.csv")

# =========================
# 4. Expense Categorization
# =========================
elif menu == "Expense Categorization":
    st.markdown("<h3>💳 Expense Categorization</h3>", unsafe_allow_html=True)
    st.write("Upload **expenses CSV file** and categorize your spending into smart buckets.")

    file = st.file_uploader("Upload Expenses CSV", type=["csv"], key="expenses")

    if file:
        expenses = pd.read_csv(file)
        st.write("### 📘 Uploaded Expenses")
        st.dataframe(expenses)

        if "Description" in expenses.columns and "Amount" in expenses.columns:
            def categorize(desc):
                desc = desc.lower()
                if "rent" in desc or "lease" in desc:
                    return "Housing"
                elif "salary" in desc or "wages" in desc:
                    return "Payroll"
                elif "utility" in desc or "electricity" in desc:
                    return "Utilities"
                elif "travel" in desc or "flight" in desc:
                    return "Travel"
                else:
                    return "Miscellaneous"

            expenses["Category"] = expenses["Description"].apply(categorize)

            st.write("### 📊 Categorized Expenses")
            st.dataframe(expenses)

            st.download_button("⬇️ Download Categorized Expenses", expenses.to_csv(index=False).encode("utf-8"), "categorized_expenses.csv")
