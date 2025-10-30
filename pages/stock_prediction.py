import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import warnings
warnings.filterwarnings('ignore')

# --- Data and Model Utility Functions ---

def get_data(ticker, years=3):
    """Get stock data with proper frequency handling for SARIMA"""
    if years < 1:
        years = 1
    elif years > 10:
        years = 10
    
    period_str = f"{years}y"
    stock_data = yf.download(ticker, period=period_str, interval="1d", progress=False, auto_adjust=True)
    stock_data = stock_data[['Close']].dropna()
    stock_data.index = pd.to_datetime(stock_data.index)
    
    # Use calendar days for proper seasonal pattern alignment
    if stock_data.index.freq is None:
        stock_data = stock_data.asfreq('D')
    
    return stock_data

def model_evaluation(ts, order, model_type="ARIMA", seasonal_order=None, window=7, test_size=30):
    """Evaluate model with improved error handling"""
    ts_processed = ts.rolling(window=window).mean().dropna()
    
    if ts_processed.index.freq is None:
        ts_processed = ts_processed.asfreq('D')
    
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
            if seasonal_order is None or seasonal_order[3] <= 1:
                # Fallback to ARIMA if seasonal period is invalid
                model = ARIMA(train, order=order)
                fit = model.fit()
            else:
                model = SARIMAX(train, order=order, seasonal_order=seasonal_order, 
                              enforce_stationarity=False, enforce_invertibility=False)
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
        
        mask = ~np.isnan(pred) & ~np.isnan(actual) & np.isfinite(pred) & np.isfinite(actual)
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
        st.error(f"Model evaluation error: {e}")
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}, None, None, None

def model_forecast(ts, order, steps=30, model_type="ARIMA", seasonal_order=None, window=7):
    """Generate forecasts with improved error handling"""
    ts_processed = ts.rolling(window=window).mean().dropna()
    
    if ts_processed.index.freq is None:
        ts_processed = ts_processed.asfreq('D')
    
    last_actual = ts_processed.index[-1]
    
    try:
        if model_type == "ARIMA":
            model = ARIMA(ts_processed, order=order)
            fit = model.fit()
            fc = fit.get_forecast(steps=steps)
        elif model_type == "SARIMA":
            if seasonal_order is None or seasonal_order[3] <= 1:
                # Fallback to ARIMA if seasonal period is invalid
                model = ARIMA(ts_processed, order=order)
                fit = model.fit()
            else:
                model = SARIMAX(ts_processed, order=order, seasonal_order=seasonal_order,
                              enforce_stationarity=False, enforce_invertibility=False)
                fit = model.fit(disp=False)
            fc = fit.get_forecast(steps=steps)
        else:
            raise ValueError("Model type must be ARIMA or SARIMA.")
        
        pred = np.asarray(fc.predicted_mean).reshape(-1)
        ci = fc.conf_int()
        ci_df = pd.DataFrame(ci)
        ci_df.columns = ["Lower CI", "Upper CI"] if len(ci.columns) == 2 else ci.columns
        
        # Generate proper date range for forecasting
        forecast_index = pd.date_range(start=last_actual + pd.Timedelta(days=1), periods=steps, freq='D')
        
        return forecast_index, pred, ci_df
    
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        return None, None, None

def arima_sarima_grid_search(ts, model_type, window, test_size, param_grid, season_grid=None, m_season=5, show_table=False):
    """Enhanced grid search with better seasonal patterns"""
    ts_processed = ts.rolling(window=window).mean().dropna()
    
    if ts_processed.index.freq is None:
        ts_processed = ts_processed.asfreq('D')
    
    if len(ts_processed) < (window + test_size + (m_season * 2 if model_type == "SARIMA" else 0)):
        return None, None
    
    best_score = float('inf')
    best_order = None
    best_season = None
    best_metrics = None
    results = []
    
    for order in param_grid:
        if model_type == "SARIMA":
            for season in season_grid:
                # Skip invalid seasonal patterns
                if season == (0, 0, 0):  # No seasonality - use ARIMA
                    continue
                    
                so = (season[0], season[1], season[2], m_season)
                score, _, _, _ = model_evaluation(
                    ts, order=order, model_type="SARIMA", seasonal_order=so, 
                    window=window, test_size=test_size
                )
                
                if score and not np.isnan(score.get('rmse', np.nan)):
                    current_score = score['rmse']
                    results.append({
                        "order": order, "seasonal_order": so,
                        "mse": score['mse'], "rmse": score['rmse'], 
                        "mae": score['mae'], "mape": score['mape']
                    })
                    
                    if current_score < best_score:
                        best_score, best_order, best_season, best_metrics = current_score, order, so, score
        else:
            score, _, _, _ = model_evaluation(
                ts, order=order, model_type="ARIMA", seasonal_order=None, 
                window=window, test_size=test_size
            )
            
            if score and not np.isnan(score.get('rmse', np.nan)):
                current_score = score['rmse']
                results.append({
                    "order": order, "mse": score['mse'], "rmse": score['rmse'],
                    "mae": score['mae'], "mape": score['mape'], "seasonal_order": ""
                })
                
                if current_score < best_score:
                    best_score, best_order, best_season, best_metrics = current_score, order, None, score
    
    if results:
        results_df = pd.DataFrame(results)
        if show_table:
            st.write("**Grid Search Results:**")
            st.dataframe(results_df.style.format({
                "mse": "{:.4f}", "rmse": "{:.4f}", "mae": "{:.4f}", "mape": "{:.2f}"
            }))
    
    return (best_order, best_season), best_metrics

# ------------- Streamlit App --------------

st.set_page_config(page_title="Advanced Stock Prediction", layout="wide")
st.title("📈 Advanced Stock Price Prediction")

with st.sidebar:
    st.header("🔧 Model Configuration")
    
    # Mode selection
    mode = st.radio("Modeling Mode", ["Manual", "Hyperparameter Tuning (Grid Search)"])
    
    # Model selection
    model_choice = st.selectbox("Select Model", ["ARIMA", "SARIMA"])
    
    # Data parameters
    period_years = st.slider("Training Data (years)", 1, 10, 3)
    window = st.slider("Rolling Mean Window", 4, 21, 7)
    
    # Model parameters
    st.subheader("Model Parameters")
    p = st.number_input("AR or lag order (p)", 0, 5, 1)
    d = st.number_input("Differencing (d)", 0, 2, 1)
    q = st.number_input("MA order (q)", 0, 5, 1)
    
    # Seasonal parameters for SARIMA
    if model_choice == "SARIMA":
        st.subheader("Seasonal Parameters")
        sp = st.number_input("Seasonal period (s)", min_value=2, max_value=52, value=5)
        st.caption("💡 Use 5 for weekly, 21 for monthly patterns")
        
        P = st.number_input("Seasonal AR (P)", 0, 2, 1)
        D = st.number_input("Seasonal Differencing (D)", 0, 1, 0)
        Q = st.number_input("Seasonal MA (Q)", 0, 2, 1)
        seasonal_order = (P, D, Q, sp)
    else:
        seasonal_order = None
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        test_size = st.slider("Validation Size (days)", 20, 60, 30)
        forecast_steps = st.slider("Forecast Period (days)", 15, 60, 30)

# Main input
st.subheader("📊 Stock Selection")
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
with col2:
    if st.button("🔄 Refresh Data"):
        st.rerun()

if ticker:
    try:
        # Get data
        close_price = get_data(ticker, years=period_years)
        
        if close_price.empty:
            st.error("No data found for this ticker. Please check the symbol.")
            st.stop()
        
        order = (int(p), int(d), int(q))
        
        # Display data info
        st.info(f"📈 Loaded {len(close_price)} days of data for {ticker}")
        
        ## --- Hyperparameter tuning mode ---
        if mode == "Hyperparameter Tuning (Grid Search)":
            st.subheader("🔍 Hyperparameter Optimization")
            
            with st.spinner("Searching optimal parameters... ⏳"):
                if model_choice == "ARIMA":
                    param_grid = list(itertools.product(range(0, 4), range(0, 2), range(0, 4)))
                    best_params, best_metrics = arima_sarima_grid_search(
                        close_price['Close'],
                        model_type="ARIMA",
                        window=window,
                        test_size=test_size,
                        param_grid=param_grid,
                        show_table=True
                    )
                    
                    if not best_params or not best_metrics:
                        st.error("Grid search failed. Try different parameters or more data.")
                        st.stop()
                    
                    order = best_params[0]
                    seasonal_order = None
                    st.success(f"✅ Best ARIMA order: {order} | RMSE: {best_metrics['rmse']:.4f}")
                
                else:  # SARIMA
                    param_grid = list(itertools.product(range(0, 3), range(0, 2), range(0, 3)))
                    # Enhanced seasonal grid with meaningful patterns
                    season_grid = [
                        (1, 0, 0),  # Seasonal AR only
                        (0, 0, 1),  # Seasonal MA only  
                        (1, 0, 1),  # Common SARIMA pattern
                        (1, 1, 1),  # Full seasonal
                        (2, 0, 1),  # Advanced pattern
                    ]
                    
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
                    
                    if not best_params or not best_metrics:
                        st.error("SARIMA grid search failed. Try ARIMA or adjust parameters.")
                        st.stop()
                    
                    order = best_params[0]
                    seasonal_order = best_params[1]
                    st.success(f"✅ Best SARIMA: {order}x{seasonal_order} | RMSE: {best_metrics['rmse']:.4f}")

        # --- Model Validation ---
        st.subheader("📊 Model Performance")
        
        model_metrics, val_pred, val_ci, val_index = model_evaluation(
            close_price['Close'],
            order=order,
            model_type=model_choice,
            seasonal_order=seasonal_order if model_choice == "SARIMA" else None,
            window=window,
            test_size=test_size,
        )

        if model_metrics == "not_enough_data":
            min_required = window + test_size
            st.error(f"❌ Insufficient data. Need {min_required}+ points after {window}-day rolling window.")
            st.write(f"Available: {len(close_price['Close'].rolling(window).mean().dropna())} points")
            st.stop()

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MSE", f"{model_metrics.get('mse', 0):.4f}")
        with col2:
            st.metric("RMSE", f"{model_metrics.get('rmse', 0):.4f}")
        with col3:
            st.metric("MAE", f"{model_metrics.get('mae', 0):.4f}")
        with col4:
            st.metric("MAPE", f"{model_metrics.get('mape', 0):.2f}%")

        # Additional analytics
        st.subheader("📈 Stock Analytics")
        latest_close = float(close_price['Close'].iloc[-1])
        mean_val = float(close_price['Close'].mean())
        std_val = float(close_price['Close'].std())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Close", f"${latest_close:.2f}")
        with col2:
            st.metric("Average Price", f"${mean_val:.2f}")
        with col3:
            st.metric("Volatility (σ)", f"${std_val:.2f}")

        # Validation forecast results
        if val_pred is not None and val_ci is not None:
            val_forecast_df = pd.DataFrame({
                "Date": val_index,
                "Forecast": val_pred,
                "Lower CI": val_ci['Lower CI'].values.ravel(),
                "Upper CI": val_ci['Upper CI'].values.ravel(),
            })
            
            st.subheader("🔍 Validation Period Results")
            st.dataframe(val_forecast_df.style.format({
                "Forecast": "${:.2f}", 
                "Lower CI": "${:.2f}", 
                "Upper CI": "${:.2f}"
            }))
            
            csv_val = val_forecast_df.to_csv(index=False)
            st.download_button(
                "📥 Download Validation Results",
                csv_val.encode('utf-8'),
                f"{ticker}_validation_forecast.csv",
                "text/csv"
            )

        # Future forecast
        forecast_index, pred, ci_df = model_forecast(
            close_price['Close'],
            order=order,
            steps=forecast_steps,
            model_type=model_choice,
            seasonal_order=seasonal_order if model_choice == "SARIMA" else None,
            window=window
        )
        
        if forecast_index is not None and pred is not None:
            forecast_df = pd.DataFrame({
                "Date": forecast_index,
                "Forecast": pred,
                "Lower CI": ci_df['Lower CI'].values.ravel(),
                "Upper CI": ci_df['Upper CI'].values.ravel(),
            })
            
            st.subheader(f"🔮 {forecast_steps}-Day Forecast")
            st.dataframe(forecast_df.style.format({
                "Forecast": "${:.2f}", 
                "Lower CI": "${:.2f}", 
                "Upper CI": "${:.2f}"
            }))
            
            csv_forecast = forecast_df.to_csv(index=False)
            st.download_button(
                "📥 Download Forecast",
                csv_forecast.encode('utf-8'),
                f"{ticker}_forecast_{forecast_steps}days.csv",
                "text/csv"
            )

            # Enhanced visualization
            st.subheader("📊 Forecast Visualization")
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Historical data
            ax.plot(close_price.index, close_price['Close'], 
                   label='Historical Price', color='#1f77b4', linewidth=2)
            
            # Validation forecast
            if val_pred is not None:
                ax.plot(val_forecast_df['Date'], val_forecast_df['Forecast'], 
                       color='#ff7f0e', label="Validation Forecast", linewidth=2)
                ax.fill_between(val_forecast_df['Date'], 
                              val_forecast_df['Lower CI'], 
                              val_forecast_df['Upper CI'],
                              color='#ff7f0e', alpha=0.2, label='Validation CI')
            
            # Future forecast
            ax.plot(forecast_df['Date'], forecast_df['Forecast'], 
                   color='#d62728', label='Future Forecast', linewidth=2)
            ax.fill_between(forecast_df['Date'], 
                          forecast_df['Lower CI'], 
                          forecast_df['Upper CI'],
                          color='#d62728', alpha=0.2, label='Future CI')
            
            # Styling
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Price ($)", fontsize=12)
            ax.set_title(f"{ticker} Stock Price Forecast | {model_choice}{order if model_choice=='ARIMA' else str(order)+'x'+str(seasonal_order)}", 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add current price line
            ax.axhline(y=latest_close, color='gray', linestyle='--', alpha=0.7, 
                      label=f'Current: ${latest_close:.2f}')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Forecast insights
            st.subheader("💡 Forecast Insights")
            forecast_change = ((pred[-1] - latest_close) / latest_close) * 100
            
            if forecast_change > 0:
                st.success(f"📈 Predicted {forecast_steps}-day return: +{forecast_change:.2f}%")
            else:
                st.error(f"📉 Predicted {forecast_steps}-day return: {forecast_change:.2f}%")
            
            st.info(f"🎯 Target price: ${pred[-1]:.2f} (Range: ${ci_df['Lower CI'].iloc[-1]:.2f} - ${ci_df['Upper CI'].iloc[-1]:.2f})")
            
    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {str(e)}")
        st.write("Please try a different ticker or adjust parameters.")

# --- Enhanced Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>📊 Advanced Stock Prediction Dashboard</b></p>
    <p>Powered by ARIMA/SARIMA Models | Built with Streamlit | Designed by Kvmeena</p>
    <p><i>⚠️ For educational purposes only. Not financial advice.</i></p>
</div>
""", unsafe_allow_html=True)