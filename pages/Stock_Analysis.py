import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import ta
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Stock Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Page Title Centered ---
st.markdown(
    """
    <style>
    h1 {text-align: center; margin-top:1em;}
    </style>
    """, unsafe_allow_html=True
)
st.title("📊 Stock Analysis Dashboard")

# --- User Input Section ---
with st.container():
    col1, col2, col3 = st.columns([3, 3, 3])
    today = datetime.date.today()
    with col1:
        ticker = st.text_input("Enter Stock Ticker", "TSLA")
    with col2:
        start_date = st.date_input("Start Date", datetime.date(today.year - 1, today.month, today.day))
    with col3:
        end_date = st.date_input("End Date", today)

# --- Data Fetch & Company Info ---
stock = yf.Ticker(ticker)
info = stock.info
currency = info.get('currency', 'USD')

st.markdown(
    f"""
    <div style="background: linear-gradient(90deg, #1976D2 60%, #42a5f5 100%); color:white; padding:12px; border-radius:8px;">
        <b>{info.get('shortName', ticker)}</b>
    </div>
    <div style="padding: 8px 0 12px 0; font-size:1.1em;">
        {info.get('longBusinessSummary', 'No Company Summary Available')}
    </div>
    """, unsafe_allow_html=True
)
col_sector, col_emp, col_site = st.columns([3, 2, 3])
col_sector.markdown(f"<b>Sector:</b><br>{info.get('sector','N/A')}", unsafe_allow_html=True)
col_emp.markdown(f"<b>Full Time Employees:</b><br>{info.get('fullTimeEmployees','N/A')}", unsafe_allow_html=True)
col_site.markdown(f"<b>Website:</b><br><a href='{info.get('website','')}' target='_blank'>{info.get('website','N/A')}</a>", unsafe_allow_html=True)

# --- Financial Metrics Card Pair ---
card_style = "background-color:#f6f9fd;box-shadow:0px 2px 15px 0px #d3e6ec;padding:18px 16px 7px 16px;border-radius:12px;margin-bottom:12px;font-size:1.05em;"
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div style='{card_style}'><b>Key Metrics</b>", unsafe_allow_html=True)
    key_metrics = {
        "Market Cap": info.get('marketCap', 'N/A'),
        "Beta": info.get('beta', 'N/A'),
        "EPS": info.get('trailingEps', 'N/A'),
        "PE Ratio": info.get('trailingPE', 'N/A')
    }
    st.table(pd.DataFrame({'Value': list(key_metrics.values())}, index=key_metrics.keys()))
with col2:
    st.markdown(f"<div style='{card_style}'><b>Financial Ratios</b>", unsafe_allow_html=True)
    fin_ratios = {
        "Quick Ratio": info.get('quickRatio', 'N/A'),
        "Profit Margins": info.get('profitMargins', 'N/A'),
        "Debt to Equity": info.get('debtToEquity', 'N/A'),
        "Return on Equity": info.get('returnOnEquity', 'N/A')
    }
    st.table(pd.DataFrame({'Value': list(fin_ratios.values())}, index=fin_ratios.keys()))

# --- Additional Financial Data (side by side) ---
extra_data = {
    "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
    "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
    "Dividend Yield": info.get('dividendYield', 'N/A'),
    "Forward PE Ratio": info.get('forwardPE', 'N/A'),
    "Avg Volume": info.get('averageVolume', 'N/A'),
    "Current Price": info.get('currentPrice', 'N/A')
}
datas = list(extra_data.items())
st.markdown(f"<div style='{card_style}'><b>Additional Data</b></div>", unsafe_allow_html=True)
ex1, ex2, ex3 = st.columns(3)
for i, (label, val) in enumerate(datas):
    if i < 2:
        ex1.metric(label, val)
    elif i < 4:
        ex2.metric(label, val)
    else:
        ex3.metric(label, val)

# --- Advanced Price Chart with Plotly ---
chart_data = stock.history(period="1y")
if not chart_data.empty:
    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_data.index, open=chart_data['Open'], high=chart_data['High'],
        low=chart_data['Low'], close=chart_data['Close'],
        name='Candlestick'
    ))
    fig.update_layout(
        title=f"{ticker} - Last 1 Year Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No historical data available for chart.")

# --- Advanced Technical Indicators Chart ---
st.markdown(f"<div style='{card_style}'><b>Technical Indicators</b></div>", unsafe_allow_html=True)
if not chart_data.empty:
    macd = ta.trend.macd(chart_data['Close'])
    rsi = ta.momentum.rsi(chart_data['Close'])
    bollinger = ta.volatility.BollingerBands(chart_data['Close'], window=20, window_dev=2)
    chart_data['MACD'] = macd
    chart_data['RSI'] = rsi
    chart_data['Bollinger_High'] = bollinger.bollinger_hband()
    chart_data['Bollinger_Low'] = bollinger.bollinger_lband()
    # Build multi-subplot
    import plotly.subplots as sp
    fig2 = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02, row_heights=[0.5, 0.25, 0.25],
                            subplot_titles=("Closing Price w/ Bollinger Bands", "MACD", "RSI"))
    # Close & BB
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'], name='Close', line=dict(color='royalblue')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Bollinger_High'], name='Bollinger High', line=dict(color='lightblue')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Bollinger_Low'], name='Bollinger Low', line=dict(color='lightblue')), row=1, col=1)
    # MACD
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MACD'], name='MACD', line=dict(color='darkgreen')), row=2, col=1)
    # RSI
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['RSI'], name='RSI', line=dict(color='orange')), row=3, col=1)
    fig2.update_layout(template='plotly_white', height=720, showlegend=False)
    # Highlight overbought/oversold
    fig2.add_shape(type="rect", xref="x", yref="y3", x0=chart_data.index[0], y0=70, x1=chart_data.index[-1], y1=100,
                  fillcolor="red", opacity=0.12, layer="below", line_width=0, row=3, col=1)
    fig2.add_shape(type="rect", xref="x", yref="y3", x0=chart_data.index[0], y0=0, x1=chart_data.index[-1], y1=30,
                  fillcolor="green", opacity=0.12, layer="below", line_width=0, row=3, col=1)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("No chart data available to compute technical indicators.")

# --- Data Table ---
st.markdown(f"<div style='{card_style}'><b>Stock Data Table</b></div>", unsafe_allow_html=True)
data_table = stock.history(start=start_date, end=end_date)
if not data_table.empty:
    st.dataframe(data_table.reset_index(), use_container_width=True)
else:
    st.info("No stock data found for the selected dates.")

# --- Footer ---
st.markdown("""
    <style>
        footer {position: fixed; left: 0; bottom: 0; width: 100%; background: #1976D2; color: white;
                text-align: center; padding: 10px; z-index: 100;}
    </style>
    <footer>
        <p>Stock Analysis Dashboard | Powered by Streamlit and Designed by Kvmeena</p>
    </footer>
    """, unsafe_allow_html=True)
