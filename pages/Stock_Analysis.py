import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import ta

st.set_page_config(
    page_title="Stock Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    h1 {text-align: center; margin-top:1em;}
    .css-1aumxhk {background: #f3f8fe;}
    .css-ffhzg2, .css-1kyxreq, .css-1uixxvy {background: #f2f6fa !important;}
    </style>""", unsafe_allow_html=True)
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

stock = yf.Ticker(ticker)
info = stock.info

# Dynamic metrics with fallback
def info_metric(key, fallback='N/A', factor=1):
    v = info.get(key)
    if v is None: return fallback
    if factor != 1:
        try: v = round(float(v)*factor, 2)
        except: pass
    return v

# --- Animated Banner & Summary ---
st.markdown(f"""
    <div style="background: linear-gradient(90deg,#1976D2,#42a5f5); 
                color:white; padding:18px; border-radius:12px; 
                animation: bounce 2s infinite alternate;">
        <b>{info.get('shortName', ticker)}</b> | {info.get('exchange','N/A')}<br>
        <span style="font-size:1.15em;">{info.get('longBusinessSummary', 'No Company Summary Available')}</span>
    </div>
    <style>
    @keyframes bounce {{0%{{transform:translateY(0);}}100%{{transform:translateY(-5px);}}}}
    </style>
    """, unsafe_allow_html=True)

col_sector, col_emp, col_site = st.columns([2, 1, 3])
col_sector.markdown(f"<b>Sector:</b><br>{info.get('sector','N/A')}", unsafe_allow_html=True)
col_emp.markdown(f"<b>Employees:</b><br>{info_metric('fullTimeEmployees')}", unsafe_allow_html=True)
col_site.markdown(f"<b>Website:</b><br><a href='{info.get('website','')}' target='_blank'>{info.get('website','N/A')}</a>", unsafe_allow_html=True)

# --- Extended Financial Metrics (Card) ---
fin_metrics = {
    "Market Cap": info_metric('marketCap'),
    "Current Price": info_metric('currentPrice'),
    "Outstanding Shares": info_metric('sharesOutstanding'),
    "Book Value/Share": info_metric('bookValue'),
    "Trailing P/E": info_metric('trailingPE'),
    "Forward P/E": info_metric('forwardPE'),
    "PEG Ratio": info_metric('pegRatio'),
    "Price/Sales": info_metric('priceToSalesTrailing12Months'),
    "Price/Book": info_metric('priceToBook'),
    "Beta": info_metric('beta'),
    "52-Week High": info_metric('fiftyTwoWeekHigh'),
    "52-Week Low": info_metric('fiftyTwoWeekLow'),
    "Dividend Yield": info_metric('dividendYield', factor=100, fallback='—'),  # as %
    "ROE": info_metric('returnOnEquity', factor=100, fallback='—'),            # as %
    "ROA": info_metric('returnOnAssets', factor=100, fallback='—'),            # as %
    "Profit Margin": info_metric('profitMargins', factor=100, fallback='—'),   # as %
    "Operating Margin": info_metric('operatingMargins', factor=100, fallback='—'), # as %
    "Debt/Equity": info_metric('debtToEquity'),
    "Quick Ratio": info_metric('quickRatio'),
    "Current Ratio": info_metric('currentRatio')
}
st.markdown("<div style='background:#e3ebf7;padding:0.6em 1em 0.3em 1em; border-radius:10px;font-weight:600;'>Key Financial Metrics</div>", unsafe_allow_html=True)
st.table(pd.DataFrame(fin_metrics.items(), columns=['Metric', 'Value']))

# --- Live Price & Animated Gauge ---
import plotly.graph_objs as go
current_price = info_metric('currentPrice', fallback=None)
fifty_two_wk_high = info_metric('fiftyTwoWeekHigh', fallback=None)
fifty_two_wk_low = info_metric('fiftyTwoWeekLow', fallback=None)
if current_price and fifty_two_wk_high and fifty_two_wk_low:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(current_price),
        delta={'reference': float(fifty_two_wk_high), 'increasing':{'color':'red'}, 'decreasing':{'color':'green'}},
        gauge={
            'axis': {'range': [fifty_two_wk_low, fifty_two_wk_high]},
            'bar': {'color': "royalblue"},
            'steps': [
                {'range': [fifty_two_wk_low, current_price], 'color': "#b6e0fe"},
                {'range': [current_price, fifty_two_wk_high], 'color': "#e3ebf7"}
            ],
            'threshold': {
                'line': {'color': "orange", 'width': 4},
                'thickness': 0.8,
                'value': float(current_price)
            }
        }
    ))
    gauge.update_layout(height=250, margin=dict(l=35, r=35, t=40, b=20),
        paper_bgcolor='#f7fafd', font=dict(color="black", size=14))
    st.plotly_chart(gauge, use_container_width=True)

# --- Interactive, Animated Price Chart ---
chart_data = stock.history(period="1y")
if not chart_data.empty:
    # Add moving averages for extra info
    chart_data['MA20'] = chart_data['Close'].rolling(window=20).mean()
    chart_data['MA50'] = chart_data['Close'].rolling(window=50).mean()

    fig = go.Figure()
    # OHLC animation
    fig.add_trace(go.Candlestick(
        x=chart_data.index, open=chart_data['Open'], high=chart_data['High'],
        low=chart_data['Low'], close=chart_data['Close'],
        name='Candlestick',
        increasing_line_color='lime', decreasing_line_color='red', opacity=0.85,
        hovertext=[f"Date: {d.date()}<br>Close: {c}" for d, c in zip(chart_data.index, chart_data['Close'])]
    ))
    # Moving averages
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data['MA20'], mode='lines', name='MA 20', line=dict(color='cyan', width=1.8, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data['MA50'], mode='lines', name='MA 50', line=dict(color='orange', width=1.8, dash='dot')
    ))

    # Animation effect via frames (replay toggle top-right)
    frames = [go.Frame(
            data=[go.Candlestick(x=chart_data.index[:k], open=chart_data['Open'][:k], high=chart_data['High'][:k],
                                 low=chart_data['Low'][:k], close=chart_data['Close'][:k])],
            name=str(k)
        ) for k in range(15, len(chart_data.index), 7)
    ]
    fig.frames = frames
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 60, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 0}}],
            }]},
        ],
        title=f"{ticker} - Last 1 Year Animated Price Trend",
        template="plotly_dark",
        xaxis=dict(rangeslider=dict(visible=False)), showlegend=True,
        autosize=True, margin=dict(l=20,r=20,t=60,b=25)
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No historical data available for chart.")

# --- Technical Indicators Combined Animation ---
st.markdown("<div style='background:#e3ebf7;padding:0.6em 1em 0.3em 1em; border-radius:10px;font-weight:600;'>Interactive Technical Indicators</div>", unsafe_allow_html=True)
if not chart_data.empty:
    macd = ta.trend.macd(chart_data['Close'])
    rsi = ta.momentum.rsi(chart_data['Close'])
    bollinger = ta.volatility.BollingerBands(chart_data['Close'], window=20, window_dev=2)
    chart_data['MACD'] = macd
    chart_data['RSI'] = rsi
    chart_data['Bollinger_High'] = bollinger.bollinger_hband()
    chart_data['Bollinger_Low'] = bollinger.bollinger_lband()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'],
                              name='Close', line=dict(color='royalblue', width=1.8)))
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Bollinger_High'], name='BB High', line=dict(color='lightblue', width=1, dash='dot')))
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Bollinger_Low'], name='BB Low', line=dict(color='lightblue', width=1, dash='dot')))
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MA20'], name='MA 20', line=dict(color='orange', dash='dash')))
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MA50'], name='MA 50', line=dict(color='green', dash='dot')))
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MACD'], name='MACD', line=dict(color='magenta')))
    fig2.add_trace(go.Scatter(x=chart_data.index, y=chart_data['RSI'], name='RSI', line=dict(color='darkred', dash='dash')))

    fig2.update_layout(
        template="plotly_white",
        title="Price, Moving Averages, MACD, RSI, Bollinger Bands (Animated Hover)",
        showlegend=True, autosize=True,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='x unified',
        hoverlabel={"namelength": -1}
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("No chart data available to compute technical indicators.")

# --- Enhanced Data Table ---
st.markdown("<div style='background:#e3ebf7;padding:0.6em 1em 0.3em 1em; border-radius:10px;font-weight:600;'>Stock Data Table</div>", unsafe_allow_html=True)
data_table = stock.history(start=start_date, end=end_date)
if not data_table.empty:
    st.dataframe(data_table.reset_index(), use_container_width=True, height=350)
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
