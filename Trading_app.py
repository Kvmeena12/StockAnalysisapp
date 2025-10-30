import streamlit as st

st.set_page_config(
    page_title="ForecastX: Time Series Stock Insights (ARIMA & SARIMA)",
    page_icon="📈",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.title("Hi, Trader 👋")
    st.markdown("<div style='margin-bottom:10px'>Welcome to your time series toolkit! 🚀</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("➔ Select a feature from the menu to begin.")
    st.markdown("---")
    st.write("Made  using Streamlit.")

# --- Enhanced Landing: Interactive UX ---
st.markdown("""
    <div style='padding:2.5rem; text-align:center; background: linear-gradient(90deg,#f4f6fb,#8ecae6,#219ebc16); border-radius: 8px 8px 0 0;'>
        <h1 style='color:#023047;font-size:2.5rem;margin-bottom:0.2em;'>ForecastX</h1>
        <h3 style='color:#219ebc;'><em>Stock Insights with ARIMA & SARIMA</em></h3>
        <p style='color:#333;font-size:1.15rem;margin-bottom:0.5em;'>
            Unlock data-driven stock analysis and AI forecasts—using classic time series models for smarter decisions.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Interactive Feature Selector ---
st.markdown("#### What do you want to explore today?")
col1, col2 = st.columns(2)
with col1:
    if st.button("🔎 Stock Analysis: Trends & Stats", use_container_width=True):
        st.switch_page("pages/stock_analysis.py")  # Assumes multipage
with col2:
    if st.button("🤖 Predict with ARIMA/SARIMA", use_container_width=True):
        st.switch_page("pages/stock_prediction.py")

st.markdown("""
<style>
.feature-row {display: flex; gap:1.7em; justify-content:center; margin-top:2em;margin-bottom:2em;flex-wrap:wrap;}
.card {background:#f7fafc; border-radius:1.2em;padding:1.5em 1.3em 1em 1.3em; box-shadow:0 4px 14px 0 #22313f0c; max-width:300px; min-width:210px; text-align:center;}
.card-emoji {font-size:1.8rem;margin-bottom:0.3em;}
@media (max-width:800px) {.feature-row{flex-direction:column;align-items:center;}}
</style>
<div class='feature-row'>
    <div class='card'><div class='card-emoji'>📈</div><b>Real-Time Analysis</b><p>Visualize key price trends and seasonality effortlessly.</p></div>
    <div class='card'><div class='card-emoji'>🤖</div><b>AI Forecasts</b><p>Run ARIMA or SARIMA for next 30-day stock predictions.</p></div>
    <div class='card'><div class='card-emoji'>📉</div><b>User Uploads</b><p>Upload your own data for custom model insights.</p></div>
    <div class='card'><div class='card-emoji'>💡</div><b>Transparent Outputs</b><p>See model diagnostics and interpret forecasts clearly.</p></div>
</div>
""", unsafe_allow_html=True)

### Guided Call-To-Action

st.success("💡 Tip: Start with **Stock Analysis** to understand history, or go straight to **Prediction** for forecasts. All powered by ARIMA and SARIMA only!")

# --- How It Works Explained ---
st.markdown("### How This Works (Interactive Guidance)")
with st.expander("View details"):
    st.markdown("""
    1. **Stock Analysis:**  
       - Select a stock, see recent price trends, auto-decompose seasonal effects, and interact with visual tools.
    2. **Prediction (ARIMA/SARIMA):**  
       - Choose a model, set forecast length, see results compared to historical data.  
       - Get easy-to-read charts and summary tables (with model accuracy/diagnostics).
    """)

st.info("To begin, use the sidebar or click a button above. All forecasts are powered by ARIMA or SARIMA for full transparency and reliability.")

# --- Persistent Footer ---
# --- Footer ---
st.markdown("""
    <style>
        footer {position: fixed; left: 0; bottom: 0; width: 100%; background: #1976D2; color: white;
                text-align: center; padding: 10px; z-index: 100;}
    </style>
    <footer>
        <p>Powered by Streamlit and Designed by Kvmeena</p>
    </footer>
    """, unsafe_allow_html=True)
