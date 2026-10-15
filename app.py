import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.inference_engine import XAUUSDInferenceEngine

# ──────────────────────────────────────────────────────────────
# Konfigurasi halaman
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='XAU/USD Trading Terminal', page_icon='🪙', layout='wide'
)

INK = '#14161c'
SURFACE = '#1b1e26'
SURFACE_2 = '#222631'
LINE = '#2c313d'
TEXT = '#ece8df'
MUTED = '#9aa0ad'
GOLD = '#d9b45a'
BUY = '#6fcf97'
SELL = '#ef8a80'
BAND = '#7aa2f7'

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
}
.stApp { background-color: __INK__; color: __TEXT__; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: __SURFACE__;
    border-right: 1px solid __LINE__;
}
section[data-testid="stSidebar"] .side-title {
    font-family: 'Fraunces', serif; font-size: 1.25rem; font-weight: 600;
    color: __TEXT__; margin-bottom: 0.2rem;
}
section[data-testid="stSidebar"] .side-note {
    color: __MUTED__; font-size: 0.86rem; line-height: 1.5; margin-bottom: 1.2rem;
}

/* Header halaman */
.page-head { display: flex; justify-content: space-between; align-items: flex-end;
    gap: 1rem; flex-wrap: wrap; margin-bottom: 1.4rem; }
.page-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.1rem;
    line-height: 1.15; color: __TEXT__; margin: 0; letter-spacing: -0.01em; }
.page-sub { color: __MUTED__; font-size: 0.95rem; margin-top: 0.35rem; }
.time-pill { background: __SURFACE__; border: 1px solid __LINE__; color: __TEXT__;
    padding: 8px 14px; border-radius: 999px; font-size: 0.88rem; }
.time-pill b { color: __GOLD__; font-weight: 600; }

/* Ringkasan dalam bahasa manusia */
.summary { background: linear-gradient(135deg, __SURFACE__ 0%, __SURFACE_2__ 100%);
    border: 1px solid __LINE__; border-left: 4px solid var(--accent, __GOLD__);
    border-radius: 14px; padding: 18px 22px; margin-bottom: 1.2rem;
    font-size: 1.05rem; line-height: 1.6; color: __TEXT__; }
.summary .hint { display: block; color: __MUTED__; font-size: 0.86rem; margin-top: 6px; }

/* Kartu metrik */
.metric-card { background: __SURFACE__; border: 1px solid __LINE__; border-radius: 16px;
    padding: 16px 18px 18px; height: 100%; min-height: 118px; }
.metric-label { font-size: 0.86rem; color: __MUTED__; margin-bottom: 8px; }
.metric-value { font-size: 1.7rem; font-weight: 600; color: __TEXT__;
    font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
.metric-sub { font-size: 0.8rem; color: __MUTED__; margin-top: 6px; }
.bar { height: 6px; background: __SURFACE_2__; border-radius: 999px; margin-top: 12px; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 999px; }

/* Badge sinyal */
.badge { display: inline-block; padding: 7px 16px; border-radius: 999px;
    font-weight: 600; font-size: 1.02rem; border: 1px solid; }
.buy-badge { background: rgba(111,207,151,0.12); color: __BUY__; border-color: rgba(111,207,151,0.45); }
.sell-badge { background: rgba(239,138,128,0.12); color: __SELL__; border-color: rgba(239,138,128,0.45); }
.wait-badge { background: __SURFACE_2__; color: __MUTED__; border-color: __LINE__; }

/* Kartu bungkus grafik */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: __SURFACE__; border: 1px solid __LINE__ !important; border-radius: 18px;
    padding: 6px 10px;
}
.section-title { font-family: 'Fraunces', serif; font-size: 1.3rem; font-weight: 600;
    color: __TEXT__; margin: 4px 0 2px; }
.section-sub { color: __MUTED__; font-size: 0.9rem; margin-bottom: 4px; }

/* Tabel proyeksi */
.table-wrap { background: __SURFACE__; border: 1px solid __LINE__; border-radius: 18px;
    padding: 8px 8px 4px; overflow-x: auto; }
table.proj { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
table.proj th { text-align: right; font-weight: 500; color: __MUTED__; font-size: 0.86rem;
    padding: 12px 16px; border-bottom: 1px solid __LINE__; }
table.proj td { text-align: right; padding: 13px 16px; border-bottom: 1px solid __LINE__;
    font-size: 0.97rem; color: __TEXT__; }
table.proj tr:last-child td { border-bottom: none; }
table.proj th:first-child, table.proj td:first-child { text-align: left; }
table.proj td:first-child { font-weight: 600; }
table.proj tbody tr:hover td { background: __SURFACE_2__; }
table.proj td.expected { color: __GOLD__; font-weight: 600; }
table.proj td.low { color: __SELL__; }
table.proj td.high { color: __BUY__; }
table.proj td.pos { color: __BUY__; }
table.proj td.neg { color: __SELL__; }
table.proj td.na { color: __MUTED__; }
.foot-note { color: __MUTED__; font-size: 0.82rem; margin: 10px 4px 0; }
</style>
"""
for key, val in {
    '__INK__': INK,
    '__SURFACE_2__': SURFACE_2,
    '__SURFACE__': SURFACE,
    '__LINE__': LINE,
    '__TEXT__': TEXT,
    '__MUTED__': MUTED,
    '__GOLD__': GOLD,
    '__BUY__': BUY,
    '__SELL__': SELL,
}.items():
    CSS = CSS.replace(key, val)
st.markdown(CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Engine & data (DI-OPTIMASI UNTUK HEMAT RAM)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return XAUUSDInferenceEngine(
        model_name='xauusd_hybrid_m1_engine.joblib'
    )


@st.cache_data
def load_data():
    path = 'data/XAUUSD_2021_2026_1min_master.parquet'
    # BACA KOLOM PENTING SAJA UNTUK MENCEGAH STACK OVERFLOW / CRASH
    df = pd.read_parquet(
        path, columns=['open', 'high', 'low', 'close', 'spread_mean']
    )
    return df


try:
    engine = load_engine()
    df_m1 = load_data()
except Exception as e:
    st.error(f'System Initialization Error: {e}')
    st.stop()


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div class="side-title">Simulator historis</div>'
    '<div class="side-note">Pilih satu candle sebagai titik awal (t0). '
    'Model akan memprediksi 12 offset M1 berikutnya.</div>',
    unsafe_allow_html=True,
)

app_mode = st.sidebar.radio(
    'Mode Tampilan:',
    options=[
        'Simulator Historis (Dengan Aktual)',
        'Live Production (Tanpa Aktual)',
    ],
    index=0,
)

# AMBIL SAMPEL WAKTU AGAR SLIDER SMOOTH
step_sample = 5
available_dates = df_m1.index[120 + 100 : -60 : step_sample]

selected_timestamp = st.sidebar.select_slider(
    'Waktu candle (t0)',
    options=available_dates,
    value=available_dates[-100],
    format_func=lambda ts: ts.strftime('%d %b %Y, %H:%M'),
)

# ──────────────────────────────────────────────────────────────
# Inferensi
# ──────────────────────────────────────────────────────────────
c_idx = df_m1.index.get_loc(selected_timestamp)
df_slice = df_m1.iloc[c_idx - 119 : c_idx + 1]
res = engine.predict(df_slice)

df_future = df_m1.iloc[c_idx + 1 : c_idx + 13]
actual_prices = df_future['close'].values

is_signal = res['is_high_conf']
is_buy = res['signal'] == 'BUY'
accent = (BUY if is_buy else SELL) if is_signal else GOLD

atr_value = float(np.mean(df_slice['high'].values - df_slice['low'].values))

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="page-head"><div>
<h1 class="page-title">XAU/USD Quantitative Execution Terminal</h1>
<div class="page-sub">Proyeksi gelombang harga emas 60 menit ke depan, lengkap dengan rentang risikonya.</div>
</div>
<div class="time-pill">Titik awal: <b>{selected_timestamp.strftime('%d %b %Y, %H:%M')}</b></div></div>""",
    unsafe_allow_html=True,
)

final_pred = float(res['pred_path'][-1])
delta = final_pred - float(res['anchor_price'])
if is_signal:
    verb = 'beli' if is_buy else 'jual'
    summary_html = (
        f'Model melihat peluang <b>{verb}</b> dengan keyakinan '
        f"<b>{res['confidence_pct']:.1f}%</b>. Dari ${res['anchor_price']:.2f}, "
        f'harga diperkirakan bergerak ke sekitar <b>${final_pred:.2f}</b> '
        f'({delta:+.2f}) dalam satu jam.'
        f'<span class="hint">Selalu cek rentang risiko di grafik sebelum mengambil keputusan.</span>'
    )
else:
    summary_html = (
        f"Keyakinan model baru <b>{res['confidence_pct']:.1f}%</b>, belum cukup "
        f'untuk memberi sinyal. Untuk saat ini lebih aman menunggu.'
        f'<span class="hint">Proyeksi tetap ditampilkan di bawah sebagai bahan pengamatan.</span>'
    )
st.markdown(
    f'<div class="summary" style="--accent:{accent}">{summary_html}</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Baris metrik
# ──────────────────────────────────────────────────────────────
conf_width = max(0.0, min(float(res['confidence_pct']), 100.0))
col1, col2, col3, col4 = st.columns(4, gap='medium')

with col1:
    st.markdown(
        f"""<div class="metric-card">
<div class="metric-label">Harga acuan (t0)</div>
<div class="metric-value">${res['anchor_price']:.2f}</div>
<div class="metric-sub">Penutupan candle terakhir</div>
</div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""<div class="metric-card">
<div class="metric-label">Volatilitas (ATR)</div>
<div class="metric-value">${atr_value:.2f}</div>
<div class="metric-sub">Rata-rata pergerakan per candle</div>
</div>""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""<div class="metric-card">
<div class="metric-label">Tingkat keyakinan</div>
<div class="metric-value">{res['confidence_pct']:.2f}%</div>
<div class="bar"><span style="width:{conf_width}%; background:{accent}"></span></div>
</div>""",
        unsafe_allow_html=True,
    )

with col4:
    if is_signal:
        badge_class = 'buy-badge' if is_buy else 'sell-badge'
        status_html = (
            f'<span class="badge {badge_class}">Sinyal {res["signal"]}</span>'
        )
        status_sub = 'Keyakinan melewati ambang batas'
    else:
        status_html = '<span class="badge wait-badge">Belum ada sinyal</span>'
        status_sub = 'Menunggu peluang yang lebih jelas'

    st.markdown(
        f"""<div class="metric-card">
<div class="metric-label">Status eksekusi</div>
<div style="margin-top: 2px;">{status_html}</div>
<div class="metric-sub">{status_sub}</div>
</div>""",
        unsafe_allow_html=True,
    )

st.write('')

# ──────────────────────────────────────────────────────────────
# Grafik (LIGHTWEIGHT RENDER)
# ──────────────────────────────────────────────────────────────
future_times = df_future.index
fig = go.Figure()

# 1. Pita rentang risiko Q10 - Q90
fig.add_trace(
    go.Scatter(
        x=list(future_times) + list(future_times)[::-1],
        y=list(res['q90_path']) + list(res['q10_path'])[::-1],
        fill='toself',
        fillcolor='rgba(122, 162, 247, 0.14)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name='Rentang risiko (Q10–Q90)',
    )
)

# 2. Riwayat harga penutupan
fig.add_trace(
    go.Scatter(
        x=df_slice.index,
        y=df_slice['close'],
        mode='lines',
        name='Riwayat (M1)',
        line=dict(color=MUTED, width=1.6),
        hovertemplate='$%{y:.2f}<extra>Riwayat</extra>',
    )
)

# 3. Harga aktual (Mode Simulator)
if app_mode.startswith('Simulator'):
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=actual_prices,
            mode='lines+markers',
            name='Harga aktual',
            line=dict(color=TEXT, width=2, dash='dot'),
            marker=dict(size=6, color=TEXT),
            hovertemplate='$%{y:.2f}<extra>Aktual</extra>',
        )
    )

# 4. Proyeksi gelombang model
line_color = BUY if is_buy else SELL
fig.add_trace(
    go.Scatter(
        x=future_times,
        y=res['pred_path'],
        mode='lines+markers',
        name='Proyeksi model (M1 Hybrid)',
        line=dict(color=line_color, width=2.8),
        marker=dict(size=7, color=line_color, line=dict(color=SURFACE, width=1.5)),
        hovertemplate='$%{y:.2f}<extra>Proyeksi</extra>',
    )
)

# Penanda t0
t0 = df_slice.index[-1]
fig.add_shape(
    type='line',
    x0=t0,
    x1=t0,
    xref='x',
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color=GOLD, width=1, dash='dash'),
)
fig.add_annotation(
    x=t0,
    xref='x',
    y=1,
    yref='paper',
    text='t0',
    showarrow=False,
    yshift=10,
    font=dict(color=GOLD, size=12),
)

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color=TEXT, size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    height=470,
    hovermode='x unified',
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=LINE, font=dict(color=TEXT)),
    xaxis=dict(showgrid=False, linecolor=LINE, zeroline=False),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(44,49,61,0.7)',
        gridwidth=1,
        zeroline=False,
        tickprefix='$',
        side='right',
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.04,
        xanchor='left',
        x=0,
        font=dict(color=MUTED, size=12),
    ),
)

with st.container(border=True):
    sub_title_txt = (
        'Garis berwarna adalah prediksi gelombang M1 model, garis putus-putus'
        ' putih adalah harga yang benar-benar terjadi.'
        if app_mode.startswith('Simulator')
        else 'Garis berwarna adalah proyeksi gelombang pergerakan harga model.'
    )
    st.markdown(
        f'<div class="section-title">Proyeksi vs kenyataan</div>'
        f'<div class="section-sub">{sub_title_txt}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)

st.write('')

# ──────────────────────────────────────────────────────────────
# Tabel proyeksi per horizon
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">Rincian per horizon</div>'
    '<div class="section-sub">Perkiraan harga, batas risiko bawah dan atas, serta selisih dengan harga aktual.</div>',
    unsafe_allow_html=True,
)

horizon_mapping = [
    ('+5m', 0),
    ('+10m', 1),
    ('+15m', 2),
    ('+30m', 5),
    ('+45m', 8),
    ('+60m', 11)
]

rows_html = ''

for h_label, h_idx in horizon_mapping:
    pred_val = res['pred_path'][h_idx]
    q10_val = res['q10_path'][h_idx]
    q90_val = res['q90_path'][h_idx]

    if app_mode.startswith('Simulator'):
        act_val = (
            actual_prices[h_idx]
            if h_idx < len(actual_prices)
            else np.nan
        )
        if np.isnan(act_val):
            actual_txt, err_txt, err_cls = 'N/A', 'N/A', 'na'
        else:
            err = act_val - pred_val
            actual_txt = f'${act_val:.2f}'
            err_txt = f'${err:+.2f}'
            err_cls = 'pos' if err >= 0 else 'neg'
    else:
        actual_txt, err_txt, err_cls = '—', '—', 'na'

    rows_html += (
        f'<tr><td>{h_label}</td>'
        f'<td class="expected">${pred_val:.2f}</td>'
        f'<td class="low">${q10_val:.2f}</td>'
        f'<td class="high">${q90_val:.2f}</td>'
        f'<td>{actual_txt}</td>'
        f'<td class="{err_cls}">{err_txt}</td></tr>'
    )

st.markdown(
    '<div class="table-wrap"><table class="proj"><thead><tr>'
    '<th>Horizon</th><th>Perkiraan (USD)</th><th>Batas bawah (Q10)</th>'
    '<th>Batas atas (Q90)</th><th>Harga aktual</th><th>Selisih ($)</th>'
    '</tr></thead><tbody>' + rows_html + '</tbody></table></div>'
    '<div class="foot-note">Selisih = harga aktual dikurangi perkiraan model. '
    'Nilai positif berarti pasar berakhir lebih tinggi dari prediksi.</div>',
    unsafe_allow_html=True,
)