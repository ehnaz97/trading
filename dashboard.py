import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rug import Rug

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Stock Dashboard with Rug")

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
period = st.sidebar.selectbox("History Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox("Candle Interval", ["1h", "4h", "1d", "1wk", "1mo"], index=2)

# Bollinger Settings
st.sidebar.subheader("Bollinger Band Settings")
window = st.sidebar.number_input("Window", value=20)
std_dev = st.sidebar.number_input("Std Dev", value=2.0)

if st.sidebar.button("Analyze Stock"):
    
    # --- 1. Get Fundamental Data with Rug ---
    try:
        rug_stock = Rug(symbol)
        
        # Fetching info using Rug methods
        basic_info = rug_stock.get_basic_info()
        price_change = rug_stock.get_current_price_change()
        
        # Display Fundamental Info in Columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Company Name", basic_info.get("name", symbol))
        with col2:
            st.metric("Current Price", f"${basic_info.get('price', 'N/A')}")    
        with col3:
            # Rug returns change as string often, checking format
            change_val = price_change.get("change")
            percent_val = price_change.get("changePercent")
            st.metric("Change", f"{change_val}", f"{percent_val}%")
            

    except Exception as e:
        st.error(f"Could not retrieve data from Rug: {e}")

    # --- 2. Get Historical Data & Calc Bollinger Bands (yfinance) ---
    # Rug doesn't seem to support historical candles, so we use yfinance for the chart
    try:        
        # Mapping intervals to yfinance format if needed, but 1h, 1d, 1wk, 1mo are standard
        # Note: 4h is not always standard in yfinance free tier (usually 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        # yfinance valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # We will map '4h' to something else or check if it works, usually 1h is safe. 
        # yfinance doesn't natively support '4h'. We can try '1h' and resample, but for simplicity let's stick to standard supported ones.
        # Actually yfinance doesn't support 4h. The closest is 1h. I will add a note or handle it.
        # Let's adjust the list to supported ones: 1h, 1d, 5d, 1wk, 1mo
        
        # However, user explicitly asked for 1h, 4h. I will try to fetch 1h and resample for 4h if needed, 
        # but simpler to just pass supported intervals. 
        # yfinance free API often restricts intraday data (1h) to the last 730 days.
        
        yf_interval = interval
        if interval == '4h':
             # yfinance doesn't support 4h directly. We use 1h and could resample, 
             # but to keep it simple and working let's warn or fallback to 1h
             # Or we can just request 1h and let pandas resample.
             yf_interval = '1h'

        df = yf.download(symbol, period=period, interval=yf_interval, progress=False)
        
        # Flatten MultiIndex columns if present (common in new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if interval == '4h' and not df.empty:
             # Resample to 4H
             df_resampled = df.resample('4h').agg({
                 'Open': 'first',
                 'High': 'max',
                 'Low': 'min',
                 'Close': 'last',
                 'Volume': 'sum'
             }).dropna()
             df = df_resampled

        if not df.empty:
            # Calculate Bollinger Bands
            # Using min_periods=1 to allow calculation from the start (expanding window)
            # This ensures bands are visible even for the first 'window' periods
            df['SMA'] = df['Close'].rolling(window=window, min_periods=1).mean()
            df['STD'] = df['Close'].rolling(window=window, min_periods=1).std()
            
            df['Upper'] = df['SMA'] + (df['STD'] * std_dev)
            df['Lower'] = df['SMA'] - (df['STD'] * std_dev)

            # --- RSI Calculation ---
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # --- 3. Plotting ---
            # Create subplots: Row 1 = Price, Row 2 = RSI, Row 3 = Volume
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05, 
                subplot_titles=(f'{symbol} Price & Bollinger Bands', 'RSI', 'Volume'),
                row_width=[0.2, 0.2, 0.6],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # Note: Plotly subplots are indexed 1-based. Row 1 is Top.
            # Row 1: Price
            # Row 2: RSI
            # Row 3: Volume

            # --- Row 1: Price & Bollinger Bands ---
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='rgba(0, 150, 255, 0.3)', width=1), name='Upper BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='rgba(0, 150, 255, 0.3)', width=1), fill='tonexty', name='Lower BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], line=dict(color='orange', width=1.5), name='SMA'), row=1, col=1)

            # --- Row 2: RSI ---
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
            # RSI Reference Lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # --- Row 3: Volume ---
            # Color volume bars based on price change (Green if Close > Open, else Red)
            colors = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=3, col=1)

            fig.update_layout(
                height=900, 
                xaxis_rangeslider_visible=False, # Disable default candlestick slider (conflicts with subplots sometimes)
                template="plotly_dark",
                showlegend=False # Optional: hide legend to save space, or keep it
            )
            
            # Enable Range Slider on the bottom plot (Volume) to control all x-axes
            fig.update_xaxes(rangeslider_visible=True, row=3, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No historical price data found.")

    except Exception as e:
        st.error(f"Error calculating technicals: {e}")
