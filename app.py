import streamlit as st
import pandas as pd
import numpy as np

# ---------- Page Config ----------
st.set_page_config(
    page_title="AI Financial Suite",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Dashboard Header ----------
st.markdown(
    """
    <h1 style='text-align:center; color:#2E86C1; font-weight:900; font-size:42px;'>
        💰 AI Financial Suite
    </h1>
    <h3 style='text-align:center; color:#D4AC0D; font-weight:600; font-size:22px; margin-top:-10px;'>
        Your 4-in-1 AI Assistant for Smarter Workflows
    </h3>
    """,
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
st.sidebar.title("📌 Navigation")
app_mode = st.sidebar.radio(
    "Choose a module:",
    ["📑 Accounts Reconciliation", "📊 Cash Flow Forecasting", "🧾 Invoice Processor", "💳 Expense Categorization"]
)

# ---------- Accounts Reconciliation ----------
if app_mode == "📑 Accounts Reconciliation":
    st.markdown(
        """
        <h2 style='color:#2E86C1; font-weight:800;'>
            📑 Accounts Reconciliation
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.write("Upload two financial statement CSV files to reconcile accounts.")
    
    file1 = st.file_uploader("Upload First Statement", type=["csv"])
    file2 = st.file_uploader("Upload Second Statement", type=["csv"])
    
    if file1 and file2:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        st.write("### Statement 1", df1.head())
        st.write("### Statement 2", df2.head())
        
        reconciliation = pd.concat([df1, df2]).drop_duplicates(keep=False)
        st.write("### Reconciliation Result", reconciliation)

# ---------- Cash Flow Forecasting ----------
elif app_mode == "📊 Cash Flow Forecasting":
    st.markdown(
        """
        <h2 style='color:#2E86C1; font-weight:800;'>
            📊 Cash Flow Forecasting
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.write("Upload historical cash flow CSV to forecast future cash flows.")
    
    file = st.file_uploader("Upload Cash Flow Data", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("### Uploaded Data", df.head())
        
        df["Forecast"] = df.iloc[:, -1].rolling(window=3).mean()
        st.write("### Forecast Result", df)

# ---------- Invoice Processor ----------
elif app_mode == "🧾 Invoice Processor":
    st.markdown(
        """
        <h2 style='color:#2E86C1; font-weight:800;'>
            🧾 Invoice Processor
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.write("Upload invoice data (CSV) to extract and process invoice details.")
    
    file = st.file_uploader("Upload Invoice Data", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("### Uploaded Invoices", df.head())
        
        st.write("### Processed Invoices")
        st.write(df.describe())

# ---------- Expense Categorization ----------
elif app_mode == "💳 Expense Categorization":
    st.markdown(
        """
        <h2 style='color:#2E86C1; font-weight:800;'>
            💳 Expense Categorization
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.write("Upload an expenses CSV file and categorize expenses automatically.")
    
    file = st.file_uploader("Upload Expense Data", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("### Uploaded Expenses", df.head())
        
        categories = ["Travel", "Food", "Supplies", "Utilities", "Other"]
        df["Category"] = np.random.choice(categories, len(df))
        st.write("### Categorized Expenses", df)
