# app.py
import io
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Optional: IsolationForest if scikit-learn installed
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------- Page config ----------
st.set_page_config(page_title="AI Financial Suite", layout="wide", initial_sidebar_state="expanded")

# ---------- Global styling ----------
st.markdown(
    """
    <style>
    .reportview-container, .main, header, .stApp {
        background-color: #071026;
        color: #e6eef8;
    }
    .card {
        background: #081426;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.6);
        border: 1px solid rgba(255,255,255,0.03);
        margin-bottom: 18px;
    }
    .header-title{ font-size:34px; font-weight:800; color:#7dd3fc; margin:0; }
    .header-sub{ font-size:16px; color:#fbbf24; margin:0; }
    .agent-title { font-size:22px; font-weight:800; color:#93c5fd; margin-bottom:4px; }
    .agent-sub { font-size:14px; color:#fbcfe8; margin-bottom:10px; }
    .stButton>button { background-image: linear-gradient(90deg,#06b6d4,#7c3aed); color:white; font-weight:700; border-radius:8px; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div {
        background-color:#051226; color:#e6eef8; border-radius:6px; padding:8px;
    }
    .small-muted { color:#9ca3af; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
try:
    logo = Image.open("logo.png")
    st.image(logo, width=90)
except Exception:
    pass

st.markdown(
    """
    <div style="text-align:center; padding:20px; border-radius:10px;
                background: linear-gradient(90deg, #0ea5e9, #6366f1);
                box-shadow: 0 6px 18px rgba(0,0,0,0.45); margin-bottom:12px;">
      <h1 class="header-title">💰 AI Financial Suite</h1>
      <p class="header-sub">Accounts Reconciliation • Cash Flow Forecasting • Invoice Processing • Expense Categorization</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.title("📌 Agents")
agent = st.sidebar.selectbox(
    "Choose an agent",
    [
        "Accounts Reconciliation",
        "Cash Flow Forecasting",
        "Invoice Processor",
        "Expense Categorization"
    ],
)
st.sidebar.markdown("---")
st.sidebar.info("Upload CSV/XLSX files or paste data. These are demo tools — secure real data in production.")

# ---------- Helpers ----------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def read_csv_or_excel(u):
    if hasattr(u, "read"):
        name = getattr(u, "name", "")
        if name.lower().endswith(".csv"):
            return pd.read_csv(u)
        else:
            return pd.read_excel(u)
    else:
        # path-like
        if str(u).lower().endswith(".csv"):
            return pd.read_csv(u)
        else:
            return pd.read_excel(u)

# ---------- Agent: Accounts Reconciliation ----------
if agent == "Accounts Reconciliation":
    st.markdown('<div class="agent-title">🧾 Accounts Reconciliation</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-sub">Upload Bank statement and Ledger; auto-match amounts and flag mismatches.</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("Upload two files: 1) Bank statement (Date, Description, Amount). 2) Ledger / Books (Date, Description, Amount).")
    bank_file = st.file_uploader("Upload Bank Statement (CSV/XLSX)", type=['csv','xlsx'], key='bank')
    ledger_file = st.file_uploader("Upload Ledger / Book (CSV/XLSX)", type=['csv','xlsx'], key='ledger')

    if bank_file and ledger_file:
        try:
            bank = read_csv_or_excel(bank_file)
            ledger = read_csv_or_excel(ledger_file)
        except Exception as e:
            st.error(f"Could not read files: {e}")
            bank = ledger = None

        if bank is not None and ledger is not None:
            st.subheader("Previews")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Bank (sample)**")
                st.dataframe(bank.head())
            with c2:
                st.markdown("**Ledger (sample)**")
                st.dataframe(ledger.head())

            def find_amount_col(df):
                for c in df.columns:
                    if c.lower() in ('amount','amt','value','transaction_amount','debit','credit'):
                        return c
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        return c
                return None

            bank_amt = find_amount_col(bank)
            ledger_amt = find_amount_col(ledger)

            st.markdown("### Matching settings")
            bank_amt = st.selectbox("Bank amount column", options=["-- none --"] + list(bank.columns), index=1 if bank_amt else 0)
            ledger_amt = st.selectbox("Ledger amount column", options=["-- none --"] + list(ledger.columns), index=1 if ledger_amt else 0)
            tolerance = st.number_input("Amount tolerance for matching (absolute)", min_value=0.0, value=0.01, step=0.01)

            if st.button("Run Reconciliation"):
                try:
                    b = bank.copy()
                    l = ledger.copy()
                    # coerce numeric
                    b['__amt'] = pd.to_numeric(b[bank_amt], errors='coerce').round(2)
                    l['__amt'] = pd.to_numeric(l[ledger_amt], errors='coerce').round(2)

                    # create helper keys: absolute amounts and rounded
                    b['__abs'] = b['__amt'].abs()
                    l['__abs'] = l['__amt'].abs()

                    # match by amount within tolerance using left join on rounded amount
                    merged = pd.merge(b.reset_index().rename(columns={'index':'bank_idx'}),
                                      l.reset_index().rename(columns={'index':'ledger_idx'}),
                                      left_on='__amt', right_on='__amt', how='left', suffixes=('_bank','_ledger'), indicator=True)
                    # mark close matches by tolerance where direct equals not found
                    # fallback: for any unmatched bank row, find ledger rows within tolerance
                    unmatched_bank = merged[merged['_merge']=='left_only'].copy()
                    # attempt tolerant matching
                    tolerant_matches = []
                    for i, row in unmatched_bank.iterrows():
                        amt = row['__amt']
                        candidates = l[ (l['__amt'].notna()) & (l['__amt'].sub(amt).abs() <= tolerance) ]
                        if not candidates.empty:
                            # pick first candidate
                            cand = candidates.iloc[0]
                            tolerant_matches.append({
                                'bank_idx': row['bank_idx'],
                                'ledger_idx': cand.name,
                                'bank_amount': amt,
                                'ledger_amount': cand['__amt']
                            })
                    # Prepare outputs
                    direct_matched = merged[merged['_merge']=='both']
                    unmatched_bank_rows = merged[merged['_merge']=='left_only']
                    st.markdown("### Results")
                    st.write(f"Direct matches: {len(direct_matched)}")
                    st.write(f"Unmatched bank rows: {len(unmatched_bank_rows)}")
                    if tolerant_matches:
                        st.write(f"Tolerant matches found: {len(tolerant_matches)}")
                        st.dataframe(pd.DataFrame(tolerant_matches))
                    st.markdown("Unmatched bank sample")
                    st.dataframe(unmatched_bank_rows.head(50))
                    st.download_button("Download unmatched bank rows (CSV)", data=df_to_csv_bytes(unmatched_bank_rows), file_name="unmatched_bank_rows.csv", mime="text/csv")
                    st.success("Reconciliation complete — review unmatched rows and tolerant matches.")
                except Exception as e:
                    st.error(f"Reconciliation failed: {e}")
    else:
        st.info("Upload both Bank statement and Ledger to run reconciliation.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Agent: Cash Flow Forecasting ----------
elif agent == "Cash Flow Forecasting":
    st.markdown('<div class="agent-title">🔮 Cash Flow Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-sub">Upload historical transactions to forecast future cash flow (monthly) using simple trend + moving average.</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload transactional CSV/XLSX with Date, Revenue, Expenses columns", type=['csv','xlsx'])
    paste = st.expander("Or paste CSV text (optional)")
    pasted = paste.text_area("Paste CSV content", height=180)
    horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=36, value=6)

    df = None
    if uploaded is not None:
        try:
            df = read_csv_or_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif pasted.strip():
        try:
            df = pd.read_csv(io.StringIO(pasted))
        except Exception as e:
            st.error(f"Could not parse pasted CSV: {e}")

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(8))

        # detect columns
        cols_lower = [c.lower() for c in df.columns]
        date_col = next((c for c in df.columns if c.lower() in ['date','transaction_date','txn_date']), None)
        rev_col = next((c for c in df.columns if c.lower() in ['revenue','income','sales','amount','credit']), None)
        exp_col = next((c for c in df.columns if c.lower() in ['expenses','expense','costs','debit']), None)

        if not date_col or (not rev_col and not exp_col):
            st.error("Couldn't detect Date and Revenue/Expenses columns. Please ensure your CSV includes them.")
        else:
            try:
                df['__date'] = pd.to_datetime(df[date_col], errors='coerce')
            except Exception:
                df['__date'] = pd.to_datetime(df[date_col], errors='coerce')

            df['__rev'] = pd.to_numeric(df[rev_col], errors='coerce').fillna(0) if rev_col else 0.0
            df['__exp'] = pd.to_numeric(df[exp_col], errors='coerce').fillna(0) if exp_col else 0.0
            df = df.dropna(subset=['__date'])
            df = df.set_index('__date')
            monthly = df.resample('M')[['__rev','__exp']].sum()
            monthly['net'] = monthly['__rev'] - monthly['__exp']
            st.markdown("### Historical (monthly)")
            st.dataframe(monthly.tail(12))

            # forecasting: rolling mean + linear trend
            window = 3
            monthly['net_ma'] = monthly['net'].rolling(window=window, min_periods=1).mean()
            last_ma = monthly['net_ma'].iloc[-1] if not monthly['net_ma'].empty else 0.0

            # linear trend on net_ma
            x = np.arange(len(monthly))
            y = monthly['net_ma'].fillna(0).values
            forecast_values = []
            if len(x) > 1 and np.std(y) > 0:
                coeff = np.polyfit(x, y, 1)
                slope = coeff[0]; intercept = coeff[1]
                for i in range(1, int(horizon)+1):
                    forecast_values.append(intercept + slope*(len(x)-1+i))
            else:
                forecast_values = [last_ma]*int(horizon)

            future_idx = pd.date_range(start=monthly.index[-1] + pd.offsets.MonthBegin(1), periods=int(horizon), freq='M')
            forecast_df = pd.DataFrame({'forecast_net': forecast_values}, index=future_idx)

            # Plot results
            st.markdown("### Forecast (net cash)")
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(monthly.index, monthly['net'], label='Historical Net', marker='o')
            ax.plot(forecast_df.index, forecast_df['forecast_net'], label='Forecast', marker='o', linestyle='--')
            ax.set_ylabel('Amount')
            ax.legend()
            st.pyplot(fig)

            out = pd.concat([monthly[['__rev','__exp','net']], forecast_df], axis=0, sort=False)
            st.download_button("Download forecast (CSV)", data=df_to_csv_bytes(out.reset_index().rename(columns={'index':'date'})), file_name="cashflow_forecast.csv", mime="text/csv")
            st.success("Forecast generated. Use with caution — this is a simple model for prototyping.")
    else:
        st.info("Upload transactions CSV/XLSX or paste CSV to run forecasting.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Agent: Invoice Processor ----------
elif agent == "Invoice Processor":
    st.markdown('<div class="agent-title">📄 Invoice Processor</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-sub">Upload invoice CSVs to extract invoice totals, due dates, vendor summary and aging.</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload invoices CSV/XLSX (InvoiceID, Vendor, Date, DueDate, Amount, Status)", type=['csv','xlsx'])
    pasted = st.expander("Or paste CSV text (optional)")
    pasted_txt = pasted.text_area("Paste invoice CSV", height=160)

    df = None
    if uploaded is not None:
        try:
            df = read_csv_or_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif pasted_txt.strip():
        try:
            df = pd.read_csv(io.StringIO(pasted_txt))
        except Exception as e:
            st.error(f"Could not parse pasted CSV: {e}")

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(12))
        cols_lower = [c.lower() for c in df.columns]
        amount_col = next((c for c in df.columns if c.lower() in ['amount','amt','total','invoice_amount']), None)
        due_col = next((c for c in df.columns if c.lower() in ['duedate','due_date','due']), None)
        status_col = next((c for c in df.columns if c.lower() == 'status'), None)

        df['__amount'] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0) if amount_col else 0
        if due_col:
            df['__due'] = pd.to_datetime(df[due_col], errors='coerce')
        else:
            df['__due'] = pd.NaT
        today = pd.Timestamp.today().normalize()
        df['__days_past_due'] = (today - df['__due']).dt.days

        def age_bucket(days):
            if pd.isna(days): return 'No due date'
            if days <= 0: return 'Current'
            if days <= 30: return '1-30'
            if days <= 60: return '31-60'
            if days <= 90: return '61-90'
            return '90+'
        df['aging_bucket'] = df['__days_past_due'].apply(age_bucket)

        st.markdown("### Invoice Summary")
        if status_col:
            outstanding = df[~df[status_col].astype(str).str.lower().isin(['paid','paid '])]['__amount'].sum()
        else:
            outstanding = df['__amount'].sum()
        st.write(f"Total outstanding (approx): {outstanding:,.2f}")

        st.markdown("### Aging distribution")
        ag = df.groupby('aging_bucket')['__amount'].sum().reindex(['Current','1-30','31-60','61-90','90+','No due date']).fillna(0)
        fig, ax = plt.subplots(figsize=(6,3))
        ag.plot(kind='bar', ax=ax)
        ax.set_ylabel('Amount')
        st.pyplot(fig)

        st.dataframe(df[['InvoiceID','Vendor','__amount','aging_bucket','__due']].head(20))
        st.download_button("Download invoice summary (CSV)", data=df_to_csv_bytes(df), file_name="invoice_summary.csv", mime="text/csv")
    else:
        st.info("Upload or paste invoice CSV/XLSX to process.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Agent: Expense Categorization ----------
elif agent == "Expense Categorization":
    st.markdown('<div class="agent-title">🗂️ Expense Categorization</div>', unsafe_allow_html=True)
    st.markdown('<div class="agent-sub">Upload expenses; auto-classify into categories (Travel, Utilities, Salaries, Marketing, Office, Other).</div>', unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload expense CSV/XLSX (Date, Description, Amount)", type=['csv','xlsx'])
    pasted = st.expander("Or paste CSV text (optional)")
    pasted_txt = pasted.text_area("Paste expense CSV", height=160)

    df = None
    if uploaded is not None:
        try:
            df = read_csv_or_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif pasted_txt.strip():
        try:
            df = pd.read_csv(io.StringIO(pasted_txt))
        except Exception as e:
            st.error(f"Could not parse pasted CSV: {e}")

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(12))
        # find description and amount columns
        desc_col = next((c for c in df.columns if c.lower() in ['description','desc','note']), df.columns[1] if len(df.columns)>1 else df.columns[0])
        amt_col = next((c for c in df.columns if c.lower() in ['amount','amt','value']), df.columns[-1])

        def simple_category(desc):
            d = str(desc).lower()
            if any(k in d for k in ['flight','hotel','uber','taxi','travel','airline','conference']):
                return 'Travel'
            if any(k in d for k in ['salary','payroll','wages']):
                return 'Salaries'
            if any(k in d for k in ['electric','utility','water','gas','utilities']):
                return 'Utilities'
            if any(k in d for k in ['ads','google','facebook','campaign','marketing','seo']):
                return 'Marketing'
            if any(k in d for k in ['office','stationery','supplies','ink','paper']):
                return 'Office'
            if any(k in d for k in ['software','saas','subscription','license']):
                return 'Software'
            return 'Other'

        df['__amount'] = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        df['category'] = df[desc_col].apply(simple_category)

        st.markdown("### Categorized Preview")
        st.dataframe(df[[desc_col, amt_col, 'category']].head(30).rename(columns={desc_col:'Description', amt_col:'Amount'}))

        # aggregate
        agg = df.groupby('category')['__amount'].sum().sort_values(ascending=False)
        st.markdown("### Spending by Category")
        fig, ax = plt.subplots(figsize=(6,3))
        agg.plot(kind='bar', ax=ax)
        ax.set_ylabel('Amount')
        st.pyplot(fig)

        st.download_button("Download categorized expenses (CSV)", data=df_to_csv_bytes(df), file_name="expenses_categorized.csv", mime="text/csv")
        st.success("Expense categorization complete. Rules are heuristic — review & adjust for your chart of accounts.")
    else:
        st.info("Upload or paste an expense CSV/XLSX to begin categorization.")
    st.markdown("</div>", unsafe_allow_html=True)
