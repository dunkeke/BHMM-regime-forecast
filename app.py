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
    page_title="BHMM Quant Terminal",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. High-End CSS Styling (Dark/Neon Theme)
# ==========================================
st.markdown("""
    <style>
    /* Global Background Override */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Metrics: Glassmorphism Effect */
    div[data-testid="stMetric"] {
        background-color: rgba(28, 31, 46, 0.8); /* Dark Blue-Grey with opacity */
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px 20px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border: 1px solid rgba(41, 98, 255, 0.5); /* Blue glow on hover */
        transform: translateY(-2px);
    }
    
    /* Metric Labels */
    div[data-testid="stMetricLabel"] {
        font-family: 'SF Pro Display', sans-serif;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b9bb4;
    }
    
    /* Metric Values */
    div[data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-weight: 600;
        color: #E0E0E0;
    }
    
    /* Expander Styling - Minimalist */
    .streamlit-expanderHeader {
        background-color: #161920;
        border-radius: 8px;
        border: 1px solid #2d303e;
        color: #E0E0E0;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #8b9bb4;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #2962FF;
        color: #2962FF;
        font-weight: bold;
    }
    
    /* Custom Button */
    div.stButton > button {
        background: linear-gradient(90deg, #2962FF 0%, #2979FF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 15px rgba(41, 98, 255, 0.6);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'SF Pro Display', sans-serif;
        letter-spacing: -0.5px;
    }
    
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Sidebar Configuration
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONFIG")
    st.divider()
    
    st.caption("UNIVERSE SELECTION")
    DEFAULT_WATCHLIST = {
        "Brent Crude (BZ)": "BZ=F",
        "WTI Crude (CL)": "CL=F",
        "Natural Gas (NG)": "NG=F",
        "Dutch TTF (TTF)": "TTF=F"
    }
    selected_assets = st.multiselect(
        "Assets", 
        options=list(DEFAULT_WATCHLIST.keys()),
        default=list(DEFAULT_WATCHLIST.keys()),
        label_visibility="collapsed"
    )
    
    st.caption("ALGORITHM PARAMS")
    n_components = st.slider("Regimes (Hidden States)", 2, 5, 3)
    window_size = st.number_input("Vol Window (Days)", value=21)
    
    st.caption("BACKTEST SETTINGS")
    lookback_years = st.slider("History (Years)", 1, 10, 4)
    transaction_cost = st.number_input("Cost (bps)", value=2) / 10000
    
    start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    st.divider()
    st.markdown("<div style='text-align: center; color: #555;'>QUANT LABS v2.0</div>", unsafe_allow_html=True)

# ==========================================
# 3. Core Logic (Cached)
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker, start, end, window):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass 
        if len(df) < 252: return None
        if 'Close' not in df.columns: pass 

        data = df[['Close']].copy()
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Log_Ret'].rolling(window=window).std()
        data.dropna(inplace=True)
        return data
    except: return None

def train_bayesian_hmm(df, n_comps, n_iter):
    scale = 100.0
    X = df[['Log_Ret', 'Volatility']].values * scale
    
    try:
        model = GaussianHMM(n_components=n_comps, covariance_type="full", n_iter=1000, 
                           random_state=42, tol=0.01, min_covar=0.001)
        model.fit(X)
    except: return None, None

    hidden_states = model.predict(X)
    
    # Sort states: Low Vol -> High Vol
    state_vol_means = [(i, X[hidden_states == i, 1].mean()) for i in range(n_comps)]
    sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
    
    # Remap
    posterior_probs = model.predict_proba(X)
    sorted_probs = np.zeros_like(posterior_probs)
    for old_i, new_i in mapping.items():
        sorted_probs[:, new_i] = posterior_probs[:, old_i]
    
    df['Regime'] = np.array([mapping[s] for s in hidden_states])
    
    # Priors & Transmat
    state_means = np.array([df[df['Regime'] == i]['Log_Ret'].mean() for i in range(n_comps)])
    new_transmat = np.zeros_like(model.transmat_)
    for i in range(n_comps):
        for j in range(n_comps):
            new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
            
    # Forecasting
    next_day_probs = np.dot(sorted_probs, new_transmat)
    df['Bayes_Exp_Ret'] = np.dot(next_day_probs, state_means)
    
    return df, sorted_probs

def run_backtest_logic(df, cost):
    threshold = 0.0005
    df['Signal'] = 0
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    df.loc[df['Bayes_Exp_Ret'] < -threshold, 'Signal'] = -1
    
    df['Position'] = df['Signal'].shift(1).fillna(0)
    t_cost = df['Position'].diff().abs() * cost
    df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - t_cost
    
    df['Cum_Bench'] = (1 + df['Log_Ret']).cumprod()
    df['Cum_Strat'] = (1 + df['Strategy_Ret']).cumprod()
    
    total_ret = df['Cum_Strat'].iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / len(df)) - 1
    sharpe = (df['Strategy_Ret'].mean() * 252) / (df['Strategy_Ret'].std() * np.sqrt(252)) if df['Strategy_Ret'].std() != 0 else 0
    max_dd = ((df['Cum_Strat'] - df['Cum_Strat'].cummax()) / df['Cum_Strat'].cummax()).min()
    
    return df, {"Total Return": total_ret, "CAGR": annual_ret, "Sharpe": sharpe, "Max Drawdown": max_dd}

# ==========================================
# 4. Main Dashboard UI
# ==========================================

st.title("💠 BHMM QUANT TERMINAL")
st.markdown(f"<div style='color: #8b9bb4; margin-bottom: 20px;'>Bayesian Regime Switching Model | <span style='color: #2962FF;'>{start_date}</span> to <span style='color: #2962FF;'>{end_date}</span></div>", unsafe_allow_html=True)

if st.button("INITIATE MARKET SCAN", use_container_width=True, type="primary"):
    
    results_summary = {}
    regime_data = {}
    
    tab1, tab2, tab3 = st.tabs(["MARKET RADAR", "ASSET ANALYSIS", "CORRELATION MAP"])
    
    with st.spinner("Processing Algorithms..."):
        processed_data = []

        for name in selected_assets:
            ticker = DEFAULT_WATCHLIST[name]
            df = get_data(ticker, start_date, end_date, window_size)
            if df is None: continue
            
            df, probs = train_bayesian_hmm(df.copy(), n_components, 1000)
            if df is None: continue
                
            df, metrics = run_backtest_logic(df, transaction_cost)
            
            # --- High-End Signal Styling ---
            last_signal_val = df['Signal'].iloc[-1]
            if last_signal_val == 1:
                last_signal = "LONG"
                signal_color = "#00E5FF" # Cyan / Electric Blue
                bg_gradient = "linear-gradient(135deg, rgba(0,229,255,0.1) 0%, rgba(0,0,0,0) 100%)"
                border_col = "#00E5FF"
            elif last_signal_val == -1:
                last_signal = "SHORT"
                signal_color = "#FF2E63" # Neon Red / Rose
                bg_gradient = "linear-gradient(135deg, rgba(255,46,99,0.1) 0%, rgba(0,0,0,0) 100%)"
                border_col = "#FF2E63"
            else:
                last_signal = "NEUTRAL"
                signal_color = "#7F8C8D" # Slate Grey
                bg_gradient = "linear-gradient(135deg, rgba(127,140,141,0.1) 0%, rgba(0,0,0,0) 100%)"
                border_col = "#555"
            
            processed_data.append({
                "name": name, "df": df, "metrics": metrics,
                "last_signal": last_signal, "signal_color": signal_color,
                "bg_gradient": bg_gradient, "border_col": border_col
            })
            regime_data[name] = df['Regime']
            results_summary[name] = metrics

    # --- Tab 1: Market Radar ---
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(len(processed_data)) if len(processed_data) <= 4 else st.columns(2)
        
        for idx, item in enumerate(processed_data):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                # Custom HTML Card
                st.markdown(f"""
                <div style="
                    background: {item['bg_gradient']};
                    border-left: 3px solid {item['border_col']};
                    border-radius: 4px;
                    padding: 15px;
                    margin-bottom: 15px;
                    border: 1px solid rgba(255,255,255,0.05);
                ">
                    <div style="color: #8b9bb4; font-size: 0.8rem; letter-spacing: 1px;">{item['name'].upper()}</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {item['signal_color']}; margin: 5px 0;">
                        {item['last_signal']}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                        <span style="color: #aaa; font-size: 0.9rem;">Exp. Alpha</span>
                        <span style="color: #fff; font-weight: bold;">{item['df']['Bayes_Exp_Ret'].iloc[-1]*100:.3f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #aaa; font-size: 0.9rem;">Regime State</span>
                        <span style="color: #fff; font-weight: bold;">{item['df']['Regime'].iloc[-1]}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### STRATEGY BENCHMARKING")
        if results_summary:
            sum_df = pd.DataFrame(results_summary).T
            c1, c2 = st.columns(2)
            
            # Sharpe
            fig_s = go.Figure(go.Bar(
                x=sum_df.index, y=sum_df['Sharpe'], 
                marker=dict(color=sum_df['Sharpe'], colorscale='Tealgrn'),
                text=sum_df['Sharpe'].round(2), textposition='auto'
            ))
            fig_s.update_layout(
                title=dict(text="RISK ADJUSTED RETURN (SHARPE)", font=dict(size=12, color='#8b9bb4')),
                template="plotly_dark", height=250, margin=dict(l=0,r=0,t=40,b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Roboto Mono", size=10),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            c1.plotly_chart(fig_s, use_container_width=True)
            
            # Drawdown
            fig_d = go.Figure(go.Bar(
                x=sum_df.index, y=sum_df['Max Drawdown']*100, 
                marker=dict(color=sum_df['Max Drawdown'], colorscale='Redor_r'),
                text=(sum_df['Max Drawdown']*100).round(1).astype(str)+'%', textposition='auto'
            ))
            fig_d.update_layout(
                title=dict(text="MAX DRAWDOWN RISK", font=dict(size=12, color='#8b9bb4')),
                template="plotly_dark", height=250, margin=dict(l=0,r=0,t=40,b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Roboto Mono", size=10),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            c2.plotly_chart(fig_d, use_container_width=True)

    # --- Tab 2: Asset Analysis ---
    with tab2:
        for item in processed_data:
            with st.expander(f"{item['name']} // DEEP DIVE", expanded=False):
                df, metrics = item['df'], item['metrics']
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{metrics['Total Return']*100:.1f}%")
                m2.metric("CAGR", f"{metrics['CAGR']*100:.1f}%")
                m3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
                m4.metric("Max DD", f"{metrics['Max Drawdown']*100:.1f}%")
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
                                   row_heights=[0.6, 0.4], subplot_titles=("REGIME DETECTION", "EQUITY CURVE"))
                
                # Sophisticated Colors for Regimes
                # 0: Calm (Teal), 1: Trans (Gold), 2: Volatile (Purple/Red)
                regime_colors = ['#00E5FF', '#FFD740', '#D500F9', '#FF1744'] 
                
                for i in range(n_components):
                    mask = df['Regime'] == i
                    if mask.any():
                        fig.add_trace(go.Scatter(
                            x=df.index[mask], y=df['Close'][mask],
                            mode='markers', marker=dict(size=3, color=regime_colors[i % 4], opacity=0.8),
                            name=f"Regime {i}"
                        ), row=1, col=1)
                
                # Thin white line for context
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], 
                                        line=dict(color='rgba(255,255,255,0.1)', width=1), showlegend=False), row=1, col=1)

                # Equity Curve
                fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Bench'], name="Benchmark", 
                                        line=dict(color='#546E7A', dash='dot', width=1.5)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Strat'], name="Strategy", 
                                        line=dict(color='#2962FF', width=2)), row=2, col=1)
                
                # Area under curve
                fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Strat'], fill='tozeroy', 
                                        fillcolor='rgba(41, 98, 255, 0.1)', line=dict(width=0), showlegend=False), row=2, col=1)

                fig.update_layout(
                    template="plotly_dark", height=500, margin=dict(l=0,r=0,t=40,b=20),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Roboto Mono", size=10),
                    legend=dict(orientation="h", y=1.1, x=0, font=dict(color="#8b9bb4"))
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: Correlation ---
    with tab3:
        if len(regime_data) > 1:
            st.markdown("#### REGIME CO-MOVEMENT MATRIX")
            corr_matrix = pd.DataFrame(regime_data).dropna().corr()
            
            fig_c = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
                colorscale='Viridis', showscale=False
            ))
            fig_c.update_layout(
                template="plotly_dark", height=400, margin=dict(l=20,r=20,t=20,b=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Roboto Mono", size=10)
            )
            st.plotly_chart(fig_c, use_container_width=True)
        else: st.warning("Need >1 Assets")

    st.markdown("---")
    st.markdown("<div style='text-align:center; color: #555; font-size: 0.8rem;'>QUANTITATIVE RESEARCH ONLY | NOT FINANCIAL ADVICE</div>", unsafe_allow_html=True)

else:
    st.info("System Ready. Initialize Scan from Sidebar.")
    st.caption("⚠️ **Disclaimer:** This tool is for quantitative research purposes only. Past performance is not indicative of future results. Not financial advice.")

else:
    # Empty State
    st.info("👈 Please configure parameters in the sidebar and click **'🚀 SCAN MARKETS'** to begin.")

