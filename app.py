import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# ==========================================
# 0. Page Config & Setup
# ==========================================
st.set_page_config(
    page_title="BHMM Energy Quant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

warnings.filterwarnings("ignore")

# Custom CSS for "Pro" look
st.markdown("""
    <style>
    /* Metric Box Styling */
    div[data-testid="stMetric"] {
        background-color: #1a1c24; /* Darker background */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2d303e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #a0a0a0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #1a1c24;
        border-radius: 5px;
    }
    
    /* Clean up padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Sidebar Configuration
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()
    
    st.subheader("Universe Selection")
    DEFAULT_WATCHLIST = {
        "Brent Crude Oil (BZ)": "BZ=F",
        "WTI Crude Oil (CL)": "CL=F",
        "Henry Hub Gas (NG)": "NG=F",
        "Dutch TTF Gas (TTF)": "TTF=F"
    }
    selected_assets = st.multiselect(
        "Select Assets", 
        options=list(DEFAULT_WATCHLIST.keys()),
        default=list(DEFAULT_WATCHLIST.keys())
    )
    
    st.subheader("Model Parameters")
    n_components = st.slider("Hidden States (Regimes)", 2, 5, 3)
    window_size = st.number_input("Volatility Window (Days)", value=21)
    iter_num = st.number_input("Training Iterations", value=1000)
    
    st.subheader("Backtest Settings")
    lookback_years = st.slider("Lookback Period (Years)", 1, 10, 4)
    transaction_cost = st.number_input("Transaction Cost (bps)", value=2) / 10000
    
    start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    st.divider()
    st.caption("v1.0.2 | Powered by HMM & Plotly")

# ==========================================
# 2. Core Logic (Cached)
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker, start, end, window):
    """Fetch and preprocess market data."""
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        
        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except:
                pass 
                
        if len(df) < 252: return None

        # Feature Engineering
        # Ensure we have 'Close'
        if 'Close' not in df.columns and len(df.columns) > 0:
             # Fallback if column names are messy
             pass 

        data = df[['Close']].copy()
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Log_Ret'].rolling(window=window).std()
        
        data.dropna(inplace=True)
        return data
    except Exception as e:
        return None

def train_bayesian_hmm(df, n_comps, n_iter):
    """Train Gaussian HMM and calculate Bayesian posterior expected returns."""
    scale = 100.0
    X = df[['Log_Ret', 'Volatility']].values * scale
    
    try:
        model = GaussianHMM(n_components=n_comps, covariance_type="full", n_iter=n_iter, 
                           random_state=42, tol=0.01, min_covar=0.001)
        model.fit(X)
    except:
        return None, None

    hidden_states = model.predict(X)
    
    # Sort states by volatility (Low Vol -> High Vol)
    state_vol_means = []
    for i in range(n_comps):
        avg_vol = X[hidden_states == i, 1].mean()
        state_vol_means.append((i, avg_vol))
    
    sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
    
    # Remap posteriors
    posterior_probs = model.predict_proba(X)
    sorted_probs = np.zeros_like(posterior_probs)
    for old_i, new_i in mapping.items():
        sorted_probs[:, new_i] = posterior_probs[:, old_i]
    
    df['Regime'] = np.array([mapping[s] for s in hidden_states])
    
    # Calculate state priors (historical mean return per state)
    state_means = []
    for i in range(n_comps):
        mean_ret = df[df['Regime'] == i]['Log_Ret'].mean()
        state_means.append(mean_ret)
    state_means = np.array(state_means)
    
    # Remap transition matrix
    new_transmat = np.zeros_like(model.transmat_)
    for i in range(n_comps):
        for j in range(n_comps):
            new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
            
    # Bayesian Forecasting (Expected Return at T+1)
    next_day_probs = np.dot(sorted_probs, new_transmat)
    df['Bayes_Exp_Ret'] = np.dot(next_day_probs, state_means)
    
    return df, sorted_probs

def run_backtest_logic(df, cost):
    """Execute strategy backtest."""
    threshold = 0.0005
    
    df['Signal'] = 0
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    df.loc[df['Bayes_Exp_Ret'] < -threshold, 'Signal'] = -1
    
    # Position logic (Shift 1 for realistic execution)
    df['Position'] = df['Signal'].shift(1).fillna(0)
    trades = df['Position'].diff().abs()
    t_cost = trades * cost
    
    df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - t_cost
    df['Cum_Bench'] = (1 + df['Log_Ret']).cumprod()
    df['Cum_Strat'] = (1 + df['Strategy_Ret']).cumprod()
    
    # Metrics
    total_ret = df['Cum_Strat'].iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / len(df)) - 1
    if df['Strategy_Ret'].std() == 0:
        sharpe = 0
    else:
        sharpe = (df['Strategy_Ret'].mean() * 252) / (df['Strategy_Ret'].std() * np.sqrt(252))
    
    roll_max = df['Cum_Strat'].cummax()
    drawdown = (df['Cum_Strat'] - roll_max) / roll_max
    max_dd = drawdown.min()
    
    return df, {
        "Total Return": total_ret,
        "CAGR": annual_ret,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }

# ==========================================
# 3. Main Dashboard UI
# ==========================================

st.title("⚡ BHMM Energy Regime Quant")
st.markdown(f"**Bayesian HMM Strategy** | Window: `{start_date}` to `{end_date}`")

# Action Button
if st.button("🚀 SCAN MARKETS", use_container_width=True, type="primary"):
    
    results_summary = {}
    regime_data = {}
    
    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "📈 Asset Details", "🧩 Correlation Matrix"])
    
    with st.spinner("Processing Market Data & Training Models..."):
        processed_data = []

        for name in selected_assets:
            ticker = DEFAULT_WATCHLIST[name]
            
            # 1. Fetch
            df = get_data(ticker, start_date, end_date, window_size)
            if df is None:
                st.error(f"Failed to fetch data for {name}")
                continue
            
            # 2. Train
            df, probs = train_bayesian_hmm(df.copy(), n_components, iter_num)
            if df is None:
                st.error(f"Model convergence failed for {name}")
                continue
                
            # 3. Backtest
            df, metrics = run_backtest_logic(df, transaction_cost)
            
            # Signal Logic
            last_signal_val = df['Signal'].iloc[-1]
            if last_signal_val == 1:
                last_signal = "LONG"
                signal_color = "#00e676" # Bright Green
            elif last_signal_val == -1:
                last_signal = "SHORT"
                signal_color = "#ff1744" # Bright Red
            else:
                last_signal = "NEUTRAL"
                signal_color = "#b0bec5" # Grey
            
            processed_data.append({
                "name": name,
                "ticker": ticker,
                "df": df,
                "metrics": metrics,
                "last_signal": last_signal,
                "signal_color": signal_color
            })
            
            regime_data[name] = df['Regime']
            results_summary[name] = metrics

    # --- Tab 1: Overview Dashboard ---
    with tab1:
        st.subheader("📡 Live Signal Monitor")
        
        # Responsive Grid for Cards
        cols = st.columns(len(processed_data)) if len(processed_data) <= 4 else st.columns(2)
        
        for idx, item in enumerate(processed_data):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                with st.container():
                    st.markdown(f"**{item['name']}**")
                    # Signal Display
                    st.markdown(f"<h2 style='color:{item['signal_color']}; margin:0; padding:0;'>{item['last_signal']}</h2>", unsafe_allow_html=True)
                    
                    # Expected Return Metric
                    bayes_ret = item['df']['Bayes_Exp_Ret'].iloc[-1] * 100
                    st.metric("Exp. Daily Alpha", f"{bayes_ret:.3f}%", 
                             delta_color="normal" if bayes_ret > 0 else "inverse")
                    
                    st.caption(f"Current Regime: **State {item['df']['Regime'].iloc[-1]}**")
                    st.divider()

        st.subheader("🏆 Strategy Performance")
        if results_summary:
            sum_df = pd.DataFrame(results_summary).T
            
            c1, c2 = st.columns(2)
            
            # Sharpe Chart
            fig_sharpe = go.Figure(go.Bar(
                x=sum_df.index, y=sum_df['Sharpe'], 
                marker_color='#00CC96', text=sum_df['Sharpe'].round(2), textposition='auto'
            ))
            fig_sharpe.update_layout(
                title="Sharpe Ratio", 
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10), height=250
            )
            c1.plotly_chart(fig_sharpe, use_container_width=True)
            
            # Drawdown Chart
            fig_dd = go.Figure(go.Bar(
                x=sum_df.index, y=sum_df['Max Drawdown']*100, 
                marker_color='#EF553B', text=(sum_df['Max Drawdown']*100).round(1).astype(str)+'%', textposition='auto'
            ))
            fig_dd.update_layout(
                title="Max Drawdown", 
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10), height=250
            )
            c2.plotly_chart(fig_dd, use_container_width=True)

    # --- Tab 2: Detailed Analysis ---
    with tab2:
        for item in processed_data:
            with st.expander(f"🔎 {item['name']} - Deep Dive", expanded=False):
                df = item['df']
                metrics = item['metrics']
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{metrics['Total Return']*100:.1f}%")
                m2.metric("CAGR", f"{metrics['CAGR']*100:.1f}%")
                m3.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
                m4.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.1f}%")
                
                # Plotly Chart
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.08, row_heights=[0.6, 0.4],
                    subplot_titles=("Price & Regime Detection", "Strategy Equity Curve")
                )
                
                # Price with Regime Coloring
                colors = ['#00E676', '#FFEA00', '#FF1744', '#D500F9'] # Green, Yellow, Red, Purple
                for i in range(n_components):
                    mask = df['Regime'] == i
                    if mask.any():
                        fig.add_trace(go.Scatter(
                            x=df.index[mask], y=df['Close'][mask],
                            mode='markers', marker=dict(size=4, color=colors[i % len(colors)]),
                            name=f"Regime {i} (Vol)"
                        ), row=1, col=1)
                
                # Ghost Line for continuity
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], 
                                        line=dict(color='rgba(255,255,255,0.15)', width=1), 
                                        showlegend=False, hoverinfo='skip'), row=1, col=1)

                # Equity Curve
                fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Bench'], name="Benchmark (B&H)", 
                                        line=dict(color='#78909c', dash='dot', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Strat'], name="BHMM Strategy", 
                                        line=dict(color='#00e5ff', width=2)), row=2, col=1)
                
                fig.update_layout(
                    template="plotly_dark",
                    height=500, 
                    margin=dict(l=10, r=10, t=30, b=10), 
                    legend=dict(orientation="h", y=1.05, x=0),
                    paper_bgcolor='rgba(0,0,0,0)', # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: Correlations ---
    with tab3:
        if len(regime_data) > 1:
            st.subheader("🧩 Cross-Asset Regime Correlation")
            st.info("Higher correlation implies assets tend to switch Regimes (e.g., Panic/Calm) simultaneously.")
            
            regime_df = pd.DataFrame(regime_data).dropna()
            corr_matrix = regime_df.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                colorscale='Viridis',
                showscale=False
            ))
            fig_corr.update_layout(
                template="plotly_dark",
                height=450, 
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Select at least 2 assets to view correlation analysis.")
            
    # Disclaimer Footer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This tool is for quantitative research purposes only. Past performance is not indicative of future results. Not financial advice.")

else:
    # Empty State
    st.info("👈 Please configure parameters in the sidebar and click **'🚀 SCAN MARKETS'** to begin.")
