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
# 0. 页面配置与初始设置
# ==========================================
st.set_page_config(
    page_title="BHMM 能源风格预测",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed" # 手机端默认折叠侧边栏
)

warnings.filterwarnings("ignore")

# 自定义CSS优化手机显示
st.markdown("""
    <style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 侧边栏配置 (控制面板)
# ==========================================
with st.sidebar:
    st.header("⚙️ 模型参数设置")
    
    st.subheader("资产池选择")
    DEFAULT_WATCHLIST = {
        "布伦特原油 (Brent)": "BZ=F",
        "WTI 原油": "CL=F",
        "天然气 (Henry Hub)": "NG=F",
        "荷兰天然气 (TTF)": "TTF=F"
    }
    selected_assets = st.multiselect(
        "选择要分析的标的", 
        options=list(DEFAULT_WATCHLIST.keys()),
        default=list(DEFAULT_WATCHLIST.keys())
    )
    
    st.subheader("HMM 模型参数")
    n_components = st.slider("隐状态数量 (Regimes)", 2, 5, 3)
    window_size = st.number_input("波动率窗口 (天)", value=21)
    iter_num = st.number_input("训练迭代次数", value=1000)
    
    st.subheader("回测参数")
    lookback_years = st.slider("回测年限", 1, 10, 4)
    transaction_cost = st.number_input("单边交易成本 (bps)", value=2) / 10000
    
    start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

# ==========================================
# 2. 核心逻辑函数 (带缓存)
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker, start, end, window):
    """获取并预处理数据 (缓存1小时)"""
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        
        # 处理 yfinance MultiIndex 问题
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except:
                pass # 某些版本可能不需要
                
        # 再次检查列名，确保只保留需要的
        if 'Close' not in df.columns:
            # 尝试修复列名 (如果只有一层但名字不对)
            if len(df.columns) > 0:
                # 这是一个简化的假设，视yfinance版本而定
                pass
        
        if len(df) < 252: return None

        # 提取核心数据
        data = df[['Close']].copy() # 只需要Close计算收益
        # 如果有 High/Low 更好，但为了稳健性只用Close计算LogRet
        
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Log_Ret'].rolling(window=window).std()
        
        data.dropna(inplace=True)
        return data
    except Exception as e:
        return None

def train_bayesian_hmm(df, n_comps, n_iter):
    """训练 HMM 模型"""
    scale = 100.0
    X = df[['Log_Ret', 'Volatility']].values * scale
    
    try:
        model = GaussianHMM(n_components=n_comps, covariance_type="full", n_iter=n_iter, 
                           random_state=42, tol=0.01, min_covar=0.001)
        model.fit(X)
    except:
        return None, None

    hidden_states = model.predict(X)
    
    # 状态排序：按波动率从小到大
    state_vol_means = []
    for i in range(n_comps):
        avg_vol = X[hidden_states == i, 1].mean()
        state_vol_means.append((i, avg_vol))
    
    sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
    
    # 重映射后验概率
    posterior_probs = model.predict_proba(X)
    sorted_probs = np.zeros_like(posterior_probs)
    for old_i, new_i in mapping.items():
        sorted_probs[:, new_i] = posterior_probs[:, old_i]
    
    df['Regime'] = np.array([mapping[s] for s in hidden_states])
    
    # 计算先验收益
    state_means = []
    for i in range(n_comps):
        mean_ret = df[df['Regime'] == i]['Log_Ret'].mean()
        state_means.append(mean_ret)
    state_means = np.array(state_means)
    
    # 重映射转移矩阵
    new_transmat = np.zeros_like(model.transmat_)
    for i in range(n_comps):
        for j in range(n_comps):
            new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
            
    # 计算贝叶斯预期收益 (Next Day)
    next_day_probs = np.dot(sorted_probs, new_transmat)
    df['Bayes_Exp_Ret'] = np.dot(next_day_probs, state_means)
    
    return df, sorted_probs

def run_backtest_logic(df, cost):
    """执行回测"""
    threshold = 0.0005
    
    df['Signal'] = 0
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    df.loc[df['Bayes_Exp_Ret'] < -threshold, 'Signal'] = -1
    
    df['Position'] = df['Signal'].shift(1).fillna(0)
    trades = df['Position'].diff().abs()
    t_cost = trades * cost
    
    df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - t_cost
    df['Cum_Bench'] = (1 + df['Log_Ret']).cumprod()
    df['Cum_Strat'] = (1 + df['Strategy_Ret']).cumprod()
    
    # 计算指标
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
# 3. 主界面逻辑
# ==========================================

st.title("🛢️ BHMM 能源风格预测")
st.caption(f"Bayesian HMM Regime Switching | 观测窗口: {start_date} ~ {end_date}")

if st.button("🚀 运行市场扫描", use_container_width=True, type="primary"):
    
    results_summary = {}
    regime_data = {}
    
    # 创建Tabs来展示不同内容
    tab1, tab2, tab3 = st.tabs(["📊 市场概览", "📈 个股详情", "🧩 风格相关性"])
    
    with st.spinner("正在训练贝叶斯隐马尔可夫模型..."):
        # 存储所有结果的列表
        processed_data = []

        for name in selected_assets:
            ticker = DEFAULT_WATCHLIST[name]
            
            # 1. 获取数据
            df = get_data(ticker, start_date, end_date, window_size)
            if df is None:
                st.error(f"{name} 数据获取失败或数据不足")
                continue
            
            # 2. 训练模型
            df, probs = train_bayesian_hmm(df.copy(), n_components, iter_num)
            if df is None:
                st.error(f"{name} 模型训练发散")
                continue
                
            # 3. 回测
            df, metrics = run_backtest_logic(df, transaction_cost)
            
            # 存储关键数据
            last_signal_val = df['Signal'].iloc[-1]
            last_signal = "看多 (Long)" if last_signal_val == 1 else ("看空 (Short)" if last_signal_val == -1 else "空仓 (Cash)")
            signal_color = "green" if last_signal_val == 1 else ("red" if last_signal_val == -1 else "gray")
            
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

    # --- Tab 1: 市场概览 (仪表盘) ---
    with tab1:
        st.markdown("### 🎯 实时信号面板")
        
        # 使用列布局显示卡片
        cols = st.columns(len(processed_data)) if len(processed_data) <= 4 else st.columns(2)
        
        for idx, item in enumerate(processed_data):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                with st.container():
                    st.markdown(f"**{item['name']}**")
                    st.markdown(f"<h3 style='color:{item['signal_color']}'>{item['last_signal']}</h3>", unsafe_allow_html=True)
                    
                    # 贝叶斯预期收益
                    bayes_ret = item['df']['Bayes_Exp_Ret'].iloc[-1] * 100
                    st.metric("预期日收益 (E)", f"{bayes_ret:.3f}%", 
                             delta_color="normal" if bayes_ret > 0 else "inverse")
                    
                    st.divider()
                    st.caption(f"当前状态: Regime {item['df']['Regime'].iloc[-1]}")

        st.markdown("### 🏆 策略绩效对比")
        if results_summary:
            sum_df = pd.DataFrame(results_summary).T
            
            # 两个柱状图并排
            c1, c2 = st.columns(2)
            
            fig_sharpe = go.Figure(go.Bar(
                x=sum_df.index, y=sum_df['Sharpe'], 
                marker_color='#00CC96', text=sum_df['Sharpe'].round(2), textposition='auto'
            ))
            fig_sharpe.update_layout(title="夏普比率 (Sharpe)", margin=dict(l=10, r=10, t=30, b=10), height=250)
            c1.plotly_chart(fig_sharpe, use_container_width=True)
            
            fig_dd = go.Figure(go.Bar(
                x=sum_df.index, y=sum_df['Max Drawdown']*100, 
                marker_color='#EF553B', text=(sum_df['Max Drawdown']*100).round(1).astype(str)+'%', textposition='auto'
            ))
            fig_dd.update_layout(title="最大回撤 (Drawdown)", margin=dict(l=10, r=10, t=30, b=10), height=250)
            c2.plotly_chart(fig_dd, use_container_width=True)

    # --- Tab 2: 个股详情 (交互图表) ---
    with tab2:
        for item in processed_data:
            with st.expander(f"📊 {item['name']} 详细分析", expanded=False):
                df = item['df']
                metrics = item['metrics']
                
                # 指标栏
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("总回报", f"{metrics['Total Return']*100:.1f}%")
                m2.metric("年化回报", f"{metrics['CAGR']*100:.1f}%")
                m3.metric("夏普比率", f"{metrics['Sharpe']:.2f}")
                m4.metric("最大回撤", f"{metrics['Max Drawdown']*100:.1f}%")
                
                # 绘图
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.05, row_heights=[0.6, 0.4],
                                   subplot_titles=("价格与Regime", "策略净值曲线"))
                
                # 价格与体制 (散点图)
                colors = ['#00ff00', '#ffcc00', '#ff0000', '#aa00ff', '#ffffff'] # 绿, 黄, 红...
                for i in range(n_components):
                    mask = df['Regime'] == i
                    if mask.any():
                        fig.add_trace(go.Scatter(
                            x=df.index[mask], y=df['Close'][mask],
                            mode='markers', marker=dict(size=4, color=colors[i % len(colors)]),
                            name=f"Regime {i} (Vol)"
                        ), row=1, col=1)
                
                # 价格线 (背景)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], 
                                        line=dict(color='rgba(255,255,255,0.2)', width=1), 
                                        showlegend=False, hoverinfo='skip'), row=1, col=1)

                # 净值曲线
                fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Bench'], name="买入持有", 
                                        line=dict(color='gray', dash='dot', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Strat'], name="BHMM策略", 
                                        line=dict(color='#00ffff', width=2)), row=2, col=1)
                
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10), 
                                 legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: 相关性分析 ---
    with tab3:
        if len(regime_data) > 1:
            st.markdown("### 🧩 跨品种体制共振 (Regime Correlation)")
            st.info("颜色越亮，表示两个品种越倾向于同时进入高波动或低波动状态。")
            
            regime_df = pd.DataFrame(regime_data).dropna()
            corr_matrix = regime_df.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                colorscale='Viridis'
            ))
            fig_corr.update_layout(height=400, width=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("请至少选择两个标的以查看相关性分析。")

else:
    st.info("👈 请在侧边栏调整参数，然后点击上方的 **'🚀 运行市场扫描'** 按钮开始分析。")
