import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

# --- Data and Model Utility Functions ---

def get_data(ticker, years=2):
    period_str = f"{years}y"
    stock_data = yf.download(ticker, period=period_str, interval="1d", progress=False, auto_adjust=True)
    stock_data = stock_data[['Close']].dropna()
    stock_data.index = pd.to_datetime(stock_data.index)
    if stock_data.index.freq is None:
        stock_data = stock_data.asfreq('B')
    return stock_data

def model_evaluation(ts, order, model_type="ARIMA", seasonal_order=None, window=7, test_size=30):
    ts_processed = ts.rolling(window=window).mean().dropna()
    if ts_processed.index.freq is None:
        ts_processed = ts_processed.asfreq('B')
    min_required = window + test_size
    if len(ts_processed) < min_required:
        return "not_enough_data", None, None, None
    train = ts_processed.iloc[:-test_size]
    test = ts_processed.iloc[-test_size:]
    try:
        if model_type == "ARIMA":
            model = ARIMA(train, order=order)
            fit = model.fit()
            fc = fit.get_forecast(steps=test_size)
        elif model_type == "SARIMA":
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            fit = model.fit(disp=False)
            fc = fit.get_forecast(steps=test_size)
        else:
            return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}, None, None, None
        pred = np.asarray(fc.predicted_mean).reshape(-1)
        actual = np.asarray(test).reshape(-1)
        ci = fc.conf_int()
        ci_df = pd.DataFrame(ci)
        ci_df.columns = ["Lower CI", "Upper CI"] if len(ci.columns) == 2 else ci.columns
        L = min(len(pred), len(actual))
        pred = pred[:L]
        actual = actual[:L]
        mask = ~np.isnan(pred) & ~np.isnan(actual)
        if not np.any(mask):
            return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}, pred, ci_df, test.index
        mse = mean_squared_error(actual[mask], pred[mask])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual[mask], pred[mask])
        nonzero = np.abs(actual[mask]) > 1e-8
        if np.any(nonzero):
            mape = np.mean(np.abs((actual[mask][nonzero] - pred[mask][nonzero]) / actual[mask][nonzero])) * 100
        else:
            mape = np.nan
        return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}, pred, ci_df, test.index
    except Exception as e:
        print("Model evaluation error:", e)
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}, None, None, None

def model_forecast(ts, order, steps=30, model_type="ARIMA", seasonal_order=None, window=7):
    ts_processed = ts.rolling(window=window).mean().dropna()
    if ts_processed.index.freq is None:
        ts_processed = ts_processed.asfreq('B')
    last_actual = ts_processed.index[-1]
    try:
        if model_type == "ARIMA":
            model = ARIMA(ts_processed, order=order)
            fit = model.fit()
            fc = fit.get_forecast(steps=steps)
        elif model_type == "SARIMA":
            model = SARIMAX(ts_processed, order=order, seasonal_order=seasonal_order)
            fit = model.fit(disp=False)
            fc = fit.get_forecast(steps=steps)
        else:
            raise ValueError("Model type must be ARIMA or SARIMA.")
        pred = np.asarray(fc.predicted_mean).reshape(-1)
        ci = fc.conf_int()
        ci_df = pd.DataFrame(ci)
        ci_df.columns = ["Lower CI", "Upper CI"] if len(ci.columns) == 2 else ci.columns
        forecast_index = pd.date_range(start=last_actual + pd.Timedelta(days=1), periods=steps, freq='B')
        return forecast_index, pred, ci_df
    except Exception as e:
        print("Forecasting error:", e)
        return None, None, None

def arima_sarima_grid_search(ts, model_type, window, test_size, param_grid, season_grid=None, m_season=7, show_table=False):
    ts_processed = ts.rolling(window=window).mean().dropna()
    if ts_processed.index.freq is None:
        ts_processed = ts_processed.asfreq('B')
    if len(ts_processed) < (window + test_size):
        return None, None
    best_score = float('inf')
    best_order = None
    best_season = None
    best_metrics = None
    results = []
    for order in param_grid:
        if model_type == "SARIMA":
            for season in season_grid:
                so = (season[0], season[1], season[2], m_season)
                score, _, _, _ = model_evaluation(
                    ts, order=order, model_type="SARIMA", seasonal_order=so, window=window, test_size=test_size
                )
                current_score = score['rmse'] if score else float('inf')
                results.append({
                    "order": order, "seasonal_order": so,
                    "mse": score['mse'], "rmse": score['rmse'], "mae": score['mae'], "mape": score['mape']
                })
                if current_score < best_score:
                    best_score, best_order, best_season, best_metrics = current_score, order, so, score
        else:
            score, _, _, _ = model_evaluation(
                ts, order=order, model_type="ARIMA", seasonal_order=None, window=window, test_size=test_size
            )
            current_score = score['rmse'] if score else float('inf')
            results.append({
                "order": order, "mse": score['mse'], "rmse": score['rmse'],
                "mae": score['mae'], "mape": score['mape'], "seasonal_order": ""
            })
            if current_score < best_score:
                best_score, best_order, best_season, best_metrics = current_score, order, None, score
    results_df = pd.DataFrame(results)
    if show_table:
        st.write("**Grid Search Results:**")
        st.dataframe(results_df.style.format({
            "mse": "{:.4f}", "rmse": "{:.4f}", "mae": "{:.4f}", "mape": "{:.2f}"
        }))
    return (best_order, best_season), best_metrics

# ------------- Streamlit App --------------

st.set_page_config(page_title="Advanced Stock Prediction", layout="wide")
st.title("Advanced Stock Price Prediction")

with st.sidebar:
    st.header("Model & Parameters")
    mode = st.radio("Modeling Mode", ["Manual", "Hyperparameter Tuning (Grid Search)"])
    model_choice = st.selectbox("Select Model", ["ARIMA", "SARIMA"])
    period_years = st.slider("Training Data (years)", 3, 10, 3)
    window = st.slider("Rolling Mean Window", 4, 14, 7)
    p = st.number_input("AR or lag order (p)", 0, 5, 1)
    d = st.number_input("Differencing (d)", 0, 2, 1)
    q = st.number_input("MA order (q)", 0, 5, 1)
    if model_choice == "SARIMA":
        sp = st.number_input("Seasonal period (s)", min_value=2, max_value=52, value=7)
        P = st.number_input("Seasonal AR (P)", 0, 2, 0)
        D = st.number_input("Seasonal Differencing (D)", 0, 2, 0)
        Q = st.number_input("Seasonal MA (Q)", 0, 2, 0)
        seasonal_order = (P, D, Q, sp)
    else:
        seasonal_order = None

# Main input
ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
test_size = 30

if ticker:
    close_price = get_data(ticker, years=period_years)
    order = (int(p), int(d), int(q))
    grid_result = None

    ## --- Hyperparameter tuning mode ---
    if mode == "Hyperparameter Tuning (Grid Search)":
        st.info("Searching parameter grid for lowest validation RMSE... (approx 1 min for demo grid)",
            icon="🔎"
        )
        if model_choice == "ARIMA":
            param_grid = list(itertools.product(range(0, 3), range(0, 2), range(0, 3)))
            best_params, best_metrics = arima_sarima_grid_search(
                close_price['Close'],
                model_type="ARIMA",
                window=window,
                test_size=test_size,
                param_grid=param_grid,
                show_table=True
            )
            if not best_params:
                st.error("Not enough data for search or all models failed.")
                st.stop()
            order = best_params[0]
            seasonal_order = None
            st.success(f"Best ARIMA order: {order} | RMSE={best_metrics['rmse']:.4f}")
        else:  # SARIMA
            param_grid = list(itertools.product(range(0, 3), range(0, 2), range(0, 3)))
            season_grid = list(itertools.product(range(0, 2), range(0, 2), range(0, 2)))
            best_params, best_metrics = arima_sarima_grid_search(
                close_price['Close'],
                model_type="SARIMA",
                window=window,
                test_size=test_size,
                param_grid=param_grid,
                season_grid=season_grid,
                m_season=int(sp),
                show_table=True
            )
            if not best_params:
                st.error("Not enough data for search or all models failed.")
                st.stop()
            order = best_params[0]
            seasonal_order = best_params[1]
            st.success(f"Best SARIMA order: {order} | seasonal_order: {seasonal_order} | RMSE={best_metrics['rmse']:.4f}")

    # --- Model Validation block ---
    min_required = window + test_size
    model_metrics, val_pred, val_ci, val_index = model_evaluation(
        close_price['Close'],
        order=order,
        model_type="SARIMA" if model_choice == "SARIMA" else "ARIMA",
        seasonal_order=seasonal_order if model_choice == "SARIMA" else None,
        window=window,
        test_size=test_size,
    )

    if model_metrics == "not_enough_data":
        st.error(f"Not enough data after {window}-day rolling window for evaluation. "
                 f"At least {min_required} points are needed after rolling. "
                 f"Select a longer period or smaller rolling window.")
        st.write(f"Data length after rolling: {len(close_price['Close'].rolling(window).mean().dropna())}")
        st.stop()

    st.write("### Model Validation Metrics (Last 30 Days)")
    st.write(f"**MSE:** {model_metrics.get('mse', float('nan')):.4f}")
    st.write(f"**RMSE:** {model_metrics.get('rmse', float('nan')):.4f}")
    st.write(f"**MAE:** {model_metrics.get('mae', float('nan')):.4f}")
    st.write(f"**MAPE:** {model_metrics.get('mape', float('nan')):.2f}%")

    st.markdown("### Additional Analytics")
    latest_close = close_price['Close'].iloc[-1]
    if isinstance(latest_close, pd.Series):
        latest_close = latest_close.iloc[0]
    st.write(f"**Latest Close:** {float(latest_close):.2f}")

    mean_val = close_price['Close'].mean()
    if isinstance(mean_val, pd.Series):
        mean_val = mean_val.iloc[0]
    st.write(f"**Mean (Past Period):** {float(mean_val):.2f}")

    std_val = close_price['Close'].std()
    if isinstance(std_val, pd.Series):
        std_val = std_val.iloc[0]
    st.write(f"**Volatility (Std Dev):** {float(std_val):.2f}")

    # ---- Actual validation period forecast preview/download ----
    val_forecast_df = pd.DataFrame({
        "Date": val_index,
        "Forecast": val_pred,
        "Lower CI": val_ci['Lower CI'].values.ravel(),
        "Upper CI": val_ci['Upper CI'].values.ravel(),
    })
    val_forecast_df.reset_index(drop=True, inplace=True)
    st.write("### Holdout Period Forecast (validation set)")
    st.dataframe(val_forecast_df)
    st.download_button(
        "Download Validation Forecast CSV",
        val_forecast_df.to_csv(index=False).encode('utf-8'),
        "validation_forecast_with_ci.csv",
        "text/csv"
    )

    # ---- Forecast into the future and download ----
    forecast_index, pred, ci_df = model_forecast(
        close_price['Close'],
        order=order,
        steps=30,
        model_type="SARIMA" if model_choice == "SARIMA" else "ARIMA",
        seasonal_order=seasonal_order if model_choice == "SARIMA" else None,
        window=window
    )
    if forecast_index is not None:
        forecast_df = pd.DataFrame({
            "Date": forecast_index,
            "Forecast": pred,
            "Lower CI": ci_df['Lower CI'].values.ravel(),
            "Upper CI": ci_df['Upper CI'].values.ravel(),
        })
        forecast_df.reset_index(drop=True, inplace=True)
        st.write("### Next 30 Days Forecast")
        st.dataframe(forecast_df)
        st.download_button(
            "Download Forecast CSV",
            forecast_df.to_csv(index=False).encode('utf-8'),
            "forecast_with_ci.csv",
            "text/csv"
        )
        # ---- Plot history, validation, and forecast ----
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(close_price.index, close_price['Close'], label='Actual', color='blue')
        # Validation (holdout) forecast
        ax.plot(val_forecast_df['Date'], val_forecast_df['Forecast'], color='green', label="Validation Forecast")
        ax.fill_between(val_forecast_df['Date'], val_forecast_df['Lower CI'], val_forecast_df['Upper CI'],
                        color='green', alpha=0.2, label='Validation CI')
        # Out-of-sample forecast
        ax.plot(forecast_df['Date'], forecast_df['Forecast'], color='red', label='Future Forecast')
        ax.fill_between(forecast_df['Date'], forecast_df['Lower CI'], forecast_df['Upper CI'],
                        color='red', alpha=0.1, label='Future CI')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title(f"Actual, Validation, and Forecast for {ticker}")
        ax.legend()
        st.pyplot(fig)
    # --- Footer ---
st.markdown("""
    <style>
        footer {position: fixed; left: 0; bottom: 0; width: 100%; background: #1976D2; color: white;
                text-align: center; padding: 10px; z-index: 100;}
    </style>
    <footer>
        <p>Stock Prediction Dashboard | Powered by Streamlit and Designed by Kvmeena</p>
    </footer>
    """, unsafe_allow_html=True)
