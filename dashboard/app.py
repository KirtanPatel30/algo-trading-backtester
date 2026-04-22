"""
dashboard/app.py
Dark terminal-style trading dashboard — Bloomberg-inspired aesthetic.
Green-on-black, monospace fonts, sharp data-dense layout.
Pages: Command Center | Strategy Lab | P&L Curves | Trade Log
"""

import sys, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QUANT TERMINAL",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Bloomberg Terminal CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap');

:root {
    --bg:      #080c10;
    --panel:   #0d1117;
    --border:  #1c2a1c;
    --green:   #00ff88;
    --green2:  #00cc66;
    --red:     #ff3355;
    --amber:   #ffaa00;
    --blue:    #00aaff;
    --muted:   #4a5a4a;
    --text:    #c8dcc8;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background-color: #060a0e !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    color: var(--green) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

.metric-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--green);
    padding: 14px 18px;
    margin: 4px 0;
}
.metric-label {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 26px;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    color: var(--green);
}
.metric-value.red  { color: var(--red) !important; }
.metric-value.amber { color: var(--amber) !important; }

.ticker-badge {
    display: inline-block;
    background: #0a1a0a;
    border: 1px solid var(--green2);
    color: var(--green);
    padding: 2px 10px;
    font-size: 11px;
    letter-spacing: 0.15em;
    margin: 2px;
}

.section-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 16px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 13px;
    letter-spacing: 0.25em;
    color: var(--muted);
    text-transform: uppercase;
}

.stSelectbox > div, .stMultiSelect > div {
    background: var(--panel) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--green) !important;
    color: var(--green) !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
}
.stButton > button:hover {
    background: var(--green) !important;
    color: var(--bg) !important;
}

[data-testid="stDataFrame"] {
    background: var(--panel) !important;
}

div[data-testid="metric-container"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--green);
    padding: 10px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plot theme ────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#080c10",
    plot_bgcolor="#080c10",
    font=dict(family="Share Tech Mono", color="#c8dcc8", size=11),
    xaxis=dict(gridcolor="#1c2a1c", linecolor="#1c2a1c", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1c2a1c", linecolor="#1c2a1c", tickfont=dict(size=10)),
    legend=dict(bgcolor="#0d1117", bordercolor="#1c2a1c", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)
COLORS = {
    "Momentum":      "#00ff88",
    "Mean Reversion":"#00aaff",
    "Pairs Trading": "#ffaa00",
    "ML Signal":     "#ff66aa",
    "benchmark":     "#4a5a4a",
}

# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    p = PROCESSED_DIR / "metrics.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_equity():
    p = PROCESSED_DIR / "equity.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None

@st.cache_data
def load_trades():
    p = PROCESSED_DIR / "trades.csv"
    return pd.read_csv(p) if p.exists() else None

metrics_df = load_metrics()
equity_df  = load_equity()
trades_df  = load_trades()

STRATEGIES = ["Momentum","Mean Reversion","Pairs Trading","ML Signal"]
TICKERS    = ["AAPL","MSFT","GOOGL","TSLA","NVDA","SPY"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⬛ QUANT TERMINAL")
st.sidebar.markdown('<div class="section-header">navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["⬛ COMMAND CENTER", "🔬 STRATEGY LAB", "📈 P&L CURVES", "📋 TRADE LOG"],
                         label_visibility="collapsed")
st.sidebar.markdown('<div class="section-header">system status</div>', unsafe_allow_html=True)
data_ok = metrics_df is not None
st.sidebar.markdown(f"{'`● DATA LOADED`' if data_ok else '`○ NO DATA — RUN pipeline`'}")
if metrics_df is not None:
    st.sidebar.markdown(f"`STRATEGIES: {metrics_df['strategy'].nunique()}`")
    st.sidebar.markdown(f"`TICKERS: {metrics_df['ticker'].nunique()}`")
    st.sidebar.markdown(f"`BACKTESTS: {len(metrics_df)}`")

# ── PAGE: COMMAND CENTER ──────────────────────────────────────────────────────
if page == "⬛ COMMAND CENTER":
    st.markdown("# ⬛ Command Center")
    st.markdown('<div class="section-header">performance overview — all strategies</div>', unsafe_allow_html=True)

    if metrics_df is None:
        st.error("NO DATA. Run `python run_all.py` first.")
    else:
        # Top KPIs
        best_sharpe = metrics_df.loc[metrics_df["sharpe"].idxmax()]
        best_return = metrics_df.loc[metrics_df["ann_return"].idxmax()]
        best_winrate= metrics_df.loc[metrics_df["win_rate"].idxmax()]
        avg_sharpe  = metrics_df["sharpe"].mean()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("BEST SHARPE",    f"{best_sharpe['sharpe']:.3f}",
                  f"{best_sharpe['strategy']} / {best_sharpe['ticker']}")
        c2.metric("BEST ANN. RETURN", f"{best_return['ann_return']:.1f}%",
                  f"{best_return['strategy']} / {best_return['ticker']}")
        c3.metric("BEST WIN RATE",  f"{best_winrate['win_rate']:.1f}%",
                  f"{best_winrate['strategy']} / {best_winrate['ticker']}")
        c4.metric("AVG SHARPE",     f"{avg_sharpe:.3f}", "all strategies")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">sharpe ratio by strategy + ticker</div>', unsafe_allow_html=True)
            fig = px.bar(
                metrics_df.sort_values("sharpe", ascending=True),
                x="sharpe", y="ticker", color="strategy",
                color_discrete_map=COLORS, orientation="h",
                barmode="group",
            )
            fig.update_layout(**PLOT_LAYOUT, height=320,
                              title=dict(text="SHARPE RATIO", font=dict(size=11, color="#4a5a4a")))
            fig.add_vline(x=1.0, line_dash="dash", line_color="#ffaa00",
                          annotation_text="1.0 threshold")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">return vs drawdown scatter</div>', unsafe_allow_html=True)
            fig2 = px.scatter(
                metrics_df, x="max_drawdown", y="ann_return",
                color="strategy", symbol="strategy",
                color_discrete_map=COLORS, size_max=14,
                hover_data=["ticker","sharpe","win_rate"],
                text="ticker",
            )
            fig2.update_traces(textposition="top center", textfont=dict(size=9))
            fig2.update_layout(**PLOT_LAYOUT, height=320,
                               title=dict(text="RETURN vs DRAWDOWN", font=dict(size=11, color="#4a5a4a")),
                               xaxis_title="MAX DRAWDOWN (%)", yaxis_title="ANN. RETURN (%)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">full metrics table</div>', unsafe_allow_html=True)
        display = metrics_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
        st.dataframe(display.style.background_gradient(subset=["sharpe","ann_return"], cmap="Greens")
                              .background_gradient(subset=["max_drawdown"], cmap="Reds"),
                     use_container_width=True, height=300)

# ── PAGE: STRATEGY LAB ────────────────────────────────────────────────────────
elif page == "🔬 STRATEGY LAB":
    st.markdown("# 🔬 Strategy Lab")
    if metrics_df is None:
        st.error("NO DATA. Run `python run_all.py` first.")
    else:
        selected = st.selectbox("SELECT STRATEGY", STRATEGIES)
        st.markdown(f'<div class="section-header">{selected} — performance breakdown</div>', unsafe_allow_html=True)

        filt = metrics_df[metrics_df["strategy"] == selected].sort_values("sharpe", ascending=False)
        c1,c2,c3 = st.columns(3)
        c1.metric("AVG SHARPE",   f"{filt['sharpe'].mean():.3f}")
        c2.metric("AVG RETURN",   f"{filt['ann_return'].mean():.1f}%")
        c3.metric("AVG DRAWDOWN", f"{filt['max_drawdown'].mean():.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=filt["ticker"], y=filt["ann_return"],
                marker_color=[COLORS.get(selected,"#00ff88") if v > 0 else "#ff3355" for v in filt["ann_return"]],
                name="Ann. Return"
            ))
            fig.add_trace(go.Bar(
                x=filt["ticker"], y=filt["bh_return"],
                marker_color="#1c2a1c", name="Buy & Hold"
            ))
            fig.update_layout(**PLOT_LAYOUT, height=300, barmode="group",
                              title=dict(text="STRATEGY vs BUY & HOLD", font=dict(size=11,color="#4a5a4a")))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            metrics_list = ["sharpe","sortino","calmar","win_rate"]
            for ticker_row in filt.itertuples():
                vals = [getattr(ticker_row, m) for m in metrics_list]
                fig2.add_trace(go.Scatterpolar(
                    r=vals, theta=["Sharpe","Sortino","Calmar","Win Rate"],
                    fill="toself", name=ticker_row.ticker,
                    line=dict(color=COLORS.get(selected,"#00ff88"))
                ))
            fig2.update_layout(
                **PLOT_LAYOUT, height=300,
                polar=dict(
                    bgcolor="#080c10",
                    radialaxis=dict(visible=True, color="#4a5a4a"),
                    angularaxis=dict(color="#4a5a4a")
                ),
                title=dict(text="RISK RADAR", font=dict(size=11, color="#4a5a4a"))
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(filt.style.background_gradient(subset=["sharpe"], cmap="Greens"),
                     use_container_width=True)

# ── PAGE: P&L CURVES ─────────────────────────────────────────────────────────
elif page == "📈 P&L CURVES":
    st.markdown("# 📈 P&L Curves")
    if equity_df is None:
        st.error("NO DATA. Run `python run_all.py` first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.selectbox("TICKER", [t for t in TICKERS if t in equity_df["ticker"].values])
        with col2:
            strategies_avail = equity_df[equity_df["ticker"]==ticker]["strategy"].unique().tolist()
            selected_strats  = st.multiselect("STRATEGIES", strategies_avail, default=strategies_avail)

        filt = equity_df[(equity_df["ticker"]==ticker) & (equity_df["strategy"].isin(selected_strats))]

        # Equity curve
        fig = go.Figure()
        for strat in selected_strats:
            sub = filt[filt["strategy"]==strat].sort_values("date")
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["equity"],
                name=strat, line=dict(color=COLORS.get(strat,"#00ff88"), width=2)
            ))
        # Benchmark
        bench = equity_df[(equity_df["ticker"]==ticker)].sort_values("date").drop_duplicates("date")
        if not bench.empty:
            bh_equity = 10000 * (1 + bench["cum_bh"].values)
            fig.add_trace(go.Scatter(
                x=bench["date"], y=bh_equity,
                name="Buy & Hold", line=dict(color="#4a5a4a", width=1, dash="dot")
            ))
        fig.update_layout(**PLOT_LAYOUT, height=350,
                          title=dict(text=f"{ticker} — EQUITY CURVE ($10,000 start)",
                                     font=dict(size=12, color="#00ff88")),
                          yaxis_title="PORTFOLIO VALUE ($)")
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown chart
        fig2 = go.Figure()
        for strat in selected_strats:
            sub = filt[filt["strategy"]==strat].sort_values("date")
            if sub.empty: continue
            peak = sub["equity"].expanding().max()
            dd   = (sub["equity"] - peak) / (peak + 1e-9) * 100
            fig2.add_trace(go.Scatter(
                x=sub["date"], y=dd, name=strat,
                line=dict(color=COLORS.get(strat,"#00ff88"), width=1),
                fill="tozeroy", fillcolor=COLORS.get(strat,"#00ff88").replace("ff","22")
            ))
        fig2.update_layout(**PLOT_LAYOUT, height=220,
                           title=dict(text="DRAWDOWN (%)", font=dict(size=11,color="#4a5a4a")),
                           yaxis_title="DRAWDOWN (%)")
        st.plotly_chart(fig2, use_container_width=True)

# ── PAGE: TRADE LOG ───────────────────────────────────────────────────────────
elif page == "📋 TRADE LOG":
    st.markdown("# 📋 Trade Log")
    if trades_df is None:
        st.error("NO DATA. Run `python run_all.py` first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            t_filter = st.multiselect("TICKER",   TICKERS, default=["AAPL","MSFT","NVDA"])
        with col2:
            s_filter = st.multiselect("STRATEGY", STRATEGIES, default=STRATEGIES[:2])

        filt = trades_df[
            (trades_df["ticker"].isin(t_filter)) &
            (trades_df["strategy"].isin(s_filter))
        ].copy()
        filt["action"] = filt["signal"].apply(lambda x: "▲ LONG" if x > 0 else "▼ SHORT" if x < 0 else "— EXIT")
        filt["date"]   = pd.to_datetime(filt["date"]).dt.strftime("%Y-%m-%d")

        c1,c2,c3 = st.columns(3)
        c1.metric("TOTAL SIGNALS", len(filt))
        c2.metric("LONG ENTRIES",  (filt["signal"] > 0).sum())
        c3.metric("SHORT ENTRIES", (filt["signal"] < 0).sum())

        # Trade frequency chart
        if not filt.empty:
            filt["date_dt"] = pd.to_datetime(filt["date"])
            daily = filt.groupby(["date_dt","strategy"]).size().reset_index(name="trades")
            fig = px.bar(daily, x="date_dt", y="trades", color="strategy",
                         color_discrete_map=COLORS,
                         title="DAILY TRADE FREQUENCY")
            fig.update_layout(**PLOT_LAYOUT, height=220)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">recent signals</div>', unsafe_allow_html=True)
        st.dataframe(
            filt[["date","ticker","strategy","action","close"]].tail(200),
            use_container_width=True, height=300
        )
