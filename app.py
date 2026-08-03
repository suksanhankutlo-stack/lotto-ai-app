import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
from datetime import datetime, timedelta
import copy
import warnings

# --- Machine Learning Modules ---
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. Setup Streamlit & Anti-Translate & CSS Styling
# ==============================================================================
st.set_page_config(page_title="ระบบวิเคราะห์เลขดับ PRO V4", page_icon="🛑", layout="wide")

# โค้ดบังคับไม่ให้ Chrome แปลภาษาอัตโนมัติ และ CSS ตกแต่งหน้าจอ
st.markdown("""
    <style>
        body { translate: no; }
        
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {visibility:hidden;}

        .stApp{
            background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
            color:white;
        }

        .main-title{
            text-align:center;
            font-size:42px;
            font-weight:900;
            color:#FFD700;
            text-shadow:2px 2px 15px orange;
            margin-top: 20px;
        }

        .sub-title{
            text-align:center;
            color:#DDDDDD;
            font-size:18px;
            margin-bottom:25px;
        }

        .box{
            background:#1e293b;
            border-radius:18px;
            padding:18px;
            border:1px solid #334155;
            box-shadow:0 0 18px rgba(0,255,255,.2);
            margin-bottom:15px;
        }

        .result{
            background:linear-gradient(90deg,#0ea5e9,#0284c7);
            border-radius:15px;
            padding:18px;
            color:white;
            font-size:22px;
            font-weight:bold;
            text-align:center;
            margin-top:20px;
            margin-bottom:15px;
        }

        .dead{
            background:#7f1d1d;
            color:white;
            padding:12px;
            border-radius:10px;
            font-size:20px;
            margin-top:10px;
            text-align:center;
            box-shadow:0 4px 6px rgba(0,0,0,0.3);
        }

        div.stButton>button{
            width:100%;
            height:60px;
            border-radius:15px;
            font-size:22px;
            font-weight:bold;
            background:linear-gradient(90deg,#ff9800,#ff5722);
            color:white;
            border: none;
        }

        div.stButton>button:hover{
            background:linear-gradient(90deg,#ffd54f,#ff9800);
            transform:scale(1.02);
            transition: 0.3s;
        }

        div[data-baseweb="select"]{
            background:#1e293b;
            border-radius:12px;
        }

        .metric{
            text-align:center;
            font-size:28px;
            font-weight:bold;
            color:#00E5FF;
        }
    </style>
    <meta name="google" content="notranslate">
""", unsafe_allow_html=True)

# Initialize Session State Cache
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'bt_cache' not in st.session_state:
    st.session_state.bt_cache = {}

# ==============================================================================
# 1. Web Scraper (พร้อมระบบ Retry)
# ==============================================================================
LOTTERY_URLS = {
    'หวยไทย': 'https://suksan18190.blogspot.com/2026/07/blog-post_07.html',
    'หวยธกส': 'https://suksan18190.blogspot.com/2026/07/blog-post_12.html',
    'หวยออมสิน': 'https://suksan18190.blogspot.com/2026/07/blog-post_525.html',
    'หวยลาว': 'https://suksan18190.blogspot.com/2026/07/blog-post.html',
    'หวยฮานอย': 'https://suksan18190.blogspot.com/2026/07/blog-post_08.html',
    'หวยมาเลย์': 'https://suksan18190.blogspot.com/2026/07/blog-post_10.html',
    'หวยหุ้นไทยเย็น': 'https://suksan18190.blogspot.com/2026/07/blog-post_11.html',
    'หวยหุ้นนิเคอิบ่าย': 'https://suksan18190.blogspot.com/2026/07/blog-post_412.html',
    'หวยหุ้นฮั่งเส็งบ่าย': 'https://suksan18190.blogspot.com/2026/07/blog-post_229.html',
    'หวยหุ้นจีนบ่าย': 'https://suksan18190.blogspot.com/2026/07/blog-post_162.html'
}

@st.cache_data(ttl=3600)
def fetch_data(lotto_name):
    if lotto_name not in LOTTERY_URLS: return None
    url = LOTTERY_URLS[lotto_name]
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        content = response.content
    except:
        return None

    try:
        soup = BeautifulSoup(content, 'html.parser')
        post_body = soup.find('div', class_=re.compile(r'post-body|entry-content'))
        if not post_body: return None

        text_content = post_body.get_text()
        pattern = r"\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d{2})"
        matches = re.findall(pattern, text_content)

        data = []
        for date_str, prize1, bot2 in matches:
            p1_str = str(prize1).zfill(3)
            bot2_str = str(bot2).zfill(2)
            data.append({
                'date': date_str,
                'draw_num': prize1,
                'hundred': int(p1_str[-3]),
                'ten': int(p1_str[-2]),
                'unit': int(p1_str[-1]),
                'bot_ten': int(bot2_str[0]),
                'bot_unit': int(bot2_str[1])
            })

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date').reset_index(drop=True)
    except Exception:
        return None

# ==============================================================================
# 2. Adaptive Feature Engineering (Cyclical Sin/Cos + Mod3)
# ==============================================================================
@st.cache_data
def build_features_adaptive(df, col, lags, rolls):
    df_feat = df.copy()
    n = len(df)
    df_feat['prev_val'] = df_feat[col].shift(1)

    df_feat['mirror'] = (df_feat['prev_val'] + 5) % 10
    df_feat['is_even'] = (df_feat['prev_val'] % 2 == 0).astype(int)
    df_feat['is_high'] = (df_feat['prev_val'] >= 5).astype(int)
    df_feat['mod3'] = (df_feat['prev_val'] % 3).fillna(0).astype(int)
    df_feat['weekday'] = df_feat['date'].dt.weekday

    df_feat['sin_num'] = np.sin(2 * np.pi * df_feat['prev_val'] / 10).fillna(0)
    df_feat['cos_num'] = np.cos(2 * np.pi * df_feat['prev_val'] / 10).fillna(0)
    df_feat['sin_weekday'] = np.sin(2 * np.pi * df_feat['weekday'] / 7).fillna(0)
    df_feat['cos_weekday'] = np.cos(2 * np.pi * df_feat['weekday'] / 7).fillna(0)

    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat[col].shift(lag)

    if 'lag_1' in df_feat.columns and 'lag_2' in df_feat.columns:
        df_feat['repeat_2'] = (df_feat['lag_1'] == df_feat['lag_2']).astype(int)
        if 'lag_3' in df_feat.columns:
            df_feat['repeat_3'] = ((df_feat['lag_1'] == df_feat['lag_2']) & (df_feat['lag_2'] == df_feat['lag_3'])).astype(int)

    for w in rolls:
        df_feat[f'rolling_mean_{w}'] = df_feat[col].shift(1).rolling(w).mean()
        df_feat[f'rolling_std_{w}'] = df_feat[col].shift(1).rolling(w).std()

    history = df_feat[col].values
    hc_windows = list(rolls)
    if n >= 500 and 50 not in hc_windows: hc_windows.append(50)

    stats_cols = {f'{typ}{w}_{d}': np.zeros(n) for typ in ['hot', 'cold'] for w in hc_windows for d in range(10) if not (typ=='cold' and w>=50)}
    skip_cols = {f'skip_{d}': np.full(n, 100) for d in range(10)}
    last_seen = {d: -1 for d in range(10)}

    for i in range(1, n):
        for d in range(10):
            skip = i - last_seen[d] if last_seen[d] != -1 else 100
            skip_cols[f'skip_{d}'][i] = skip

        for w in hc_windows:
            window_slice = history[max(0, i-w):i]
            for d in range(10):
                hot_count = np.sum(window_slice == d)
                stats_cols[f'hot{w}_{d}'][i] = hot_count
                if w < 50:
                    stats_cols[f'cold{w}_{d}'][i] = len(window_slice) - hot_count
        last_seen[history[i]] = i

    for key, val in skip_cols.items(): df_feat[key] = val
    for key, val in stats_cols.items(): df_feat[key] = val

    return df_feat.fillna(-1)

# ==============================================================================
# 3. Optimized Elimination System (PRO V4 - Ultimate Edition)
# ==============================================================================
class OptimizedEliminationSystemV4:
    def __init__(self, df, target_col, lotto_name):
        self.df = df.copy()
        self.target_col = target_col
        self.lotto_name = lotto_name
        n = len(self.df)

        if n >= 700:
            self.mode_name = "Mode 4 (700+ งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop, self.depth = 100, 20, 10, 6
            self.lags, self.rolls = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20]
            self.ai_weights = (1.0, 1.0, 1.0, 1.0)
        elif n >= 400:
            self.mode_name = "Mode 3 (400-699 งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop, self.depth = 100, 20, 10, 5
            self.lags, self.rolls = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20]
            self.ai_weights = (1.0, 0.9, 0.8, 1.0)
        elif n >= 200:
            self.mode_name = "Mode 2 (200-399 งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop, self.depth = 80, 15, 8, 4
            self.lags, self.rolls = [1, 2, 3, 5, 8], [3, 5, 10, 20]
            self.ai_weights = (1.0, 0.8, 0.6, 0.5)
        else:
            self.mode_name = "Mode 1 (100-199 งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop, self.depth = 60, 10, 5, 3
            self.lags, self.rolls = [1, 2, 3, 5], [3, 5, 10]
            self.ai_weights = (1.0, 0.8, 0.5, 0.15)

        if n < 100: self.test_size = min(5, max(0, n - 30))

        self.models = {
            'rf': RandomForestClassifier(n_estimators=self.trees, random_state=42, max_depth=self.depth, n_jobs=1),
            'et': ExtraTreesClassifier(n_estimators=self.trees, random_state=42, max_depth=self.depth, n_jobs=1),
            'hgb': HistGradientBoostingClassifier(random_state=42, max_iter=50),
            'xgb': XGBClassifier(n_estimators=50, max_depth=self.depth, tree_method="hist", verbosity=0, random_state=42, n_jobs=1)
        }

        self.model_weights_dict = {
            'rf': self.ai_weights[0], 'et': self.ai_weights[1],
            'hgb': self.ai_weights[2], 'xgb': self.ai_weights[3]
        }

        self.df_feat = build_features_adaptive(self.df, self.target_col, tuple(self.lags), tuple(self.rolls))

    def precompute_markov_adaptive(self, df_hist):
        seq = df_hist[self.target_col].values
        n = len(seq)
        if n < 5: return np.ones(10)/10.0

        L1, L2, L3 = seq[-1], seq[-2], seq[-3] if n >= 6 else -1

        mc1, tot1 = np.zeros(10), 0
        for i in range(1, len(seq)-1):
            if seq[i] == L1: mc1[seq[i+1]] += 1; tot1 += 1
        prob_o1 = mc1 / tot1 if tot1 > 0 else np.ones(10)/10.0

        if n < 200: return prob_o1

        mc2, tot2 = np.zeros(10), 0
        for i in range(2, len(seq)-1):
            if seq[i-1] == L2 and seq[i] == L1: mc2[seq[i+1]] += 1; tot2 += 1
        prob_o2 = mc2 / tot2 if tot2 > 0 else prob_o1

        if n < 500: return (0.6 * prob_o2) + (0.4 * prob_o1)

        mc3, tot3 = np.zeros(10), 0
        for i in range(3, len(seq)-1):
            if seq[i-2] == L3 and seq[i-1] == L2 and seq[i] == L1: mc3[seq[i+1]] += 1; tot3 += 1
        prob_o3 = mc3 / tot3 if tot3 > 0 else prob_o2

        return (0.5 * prob_o3) + (0.3 * prob_o2) + (0.2 * prob_o1)

    def calculate_freq_skip(self, df_hist, digit):
        col = self.target_col
        freq = (df_hist[col] == digit).sum() / max(len(df_hist), 1)
        matches = df_hist[df_hist[col] == digit]
        skip = len(df_hist) - matches.index[-1] - 1 if len(matches) > 0 else 100

        norm_freq = min(freq * 10, 1.0)
        norm_skip = max(1.0 - (skip / 30), 0.0)
        return (0.5 * norm_freq) + (0.5 * norm_skip)

    def run_backtest(self, X_train, y_train, df_hist_cut, test_size):
        cache_key = f"bt_{self.lotto_name}_{self.target_col}_{len(df_hist_cut)}_{test_size}_ultimate_fs"
        
        if cache_key in st.session_state.bt_cache:
            return st.session_state.bt_cache[cache_key]

        bt_train_X = X_train.iloc[:-test_size]
        bt_train_y = y_train.iloc[:-test_size]
        bt_test_X = X_train.iloc[-test_size:]
        bt_test_y = y_train.iloc[-test_size:].values

        ai_fails = 0
        trained_models = {}
        for name, model in self.models.items():
            m = copy.deepcopy(model)
            m.fit(bt_train_X, bt_train_y)
            trained_models[name] = m

        ai_preds = np.zeros((test_size, 10))
        total_ai_weight = sum(self.model_weights_dict.values())

        for name, m in trained_models.items():
            preds = m.predict_proba(bt_test_X)
            full_preds = np.zeros((test_size, 10))
            for idx, c in enumerate(m.classes_):
                full_preds[:, int(c)] = preds[:, idx]
            ai_preds += full_preds * self.model_weights_dict[name]
        ai_preds /= total_ai_weight

        for i in range(test_size):
            dead_7 = np.argsort(ai_preds[i])[:7]
            if bt_test_y[i] in dead_7: ai_fails += 1

        stat_fails = 0
        for i in range(test_size):
            curr_hist = df_hist_cut.iloc[:-(test_size - i)]
            mk = self.precompute_markov_adaptive(curr_hist)
            st_probs = np.zeros(10)
            for d in range(10):
                st_probs[d] = (0.5 * self.calculate_freq_skip(curr_hist, d)) + (0.5 * mk[d])
            dead_7 = np.argsort(st_probs)[:7]
            if bt_test_y[i] in dead_7: stat_fails += 1

        day_fails = 0
        for i in range(test_size):
            curr_hist = df_hist_cut.iloc[:-(test_size - i)]
            target_dow = df_hist_cut.iloc[-(test_size - i)]['date'].weekday()
            day_probs = np.zeros(10)
            day_df = curr_hist[curr_hist['date'].dt.weekday == target_dow]
            if len(day_df) > 0:
                counts = day_df[self.target_col].value_counts(normalize=True)
                for d in range(10): day_probs[d] = counts.get(d, 0.0)
            else: day_probs = np.ones(10)/10.0
            dead_7 = np.argsort(day_probs)[:7]
            if bt_test_y[i] in dead_7: day_fails += 1

        result = (ai_fails, stat_fails, day_fails)
        st.session_state.bt_cache[cache_key] = result
        return result

    def analyze(self, target_dow):
        df_work = self.df_feat
        df_hist = self.df
        data_size = len(df_hist)
        if data_size < 30: return None

        exclude = ['date', 'draw_num', 'hundred', 'ten', 'unit', 'bot_ten', 'bot_unit', self.target_col]
        feature_cols = [c for c in df_work.columns if c not in exclude]

        X, y = df_work[feature_cols], df_work[self.target_col]
        train_X, test_X = X.iloc[:-1], X.iloc[-1:]
        train_y = y.iloc[:-1]
        df_hist_cut = df_hist.iloc[:-1]

        # ✂️ Feature Selection
        selector = ExtraTreesClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=1)
        selector.fit(train_X, train_y)

        top_n = 40 if data_size >= 400 else (30 if data_size >= 200 else 20)
        top_n = min(top_n, len(feature_cols))

        important_indices = np.argsort(selector.feature_importances_)[::-1][:top_n]
        selected_features = [feature_cols[i] for i in important_indices]
        self.selected_feat_count = len(selected_features)

        train_X = train_X[selected_features]
        test_X = test_X[selected_features]

        if data_size < 200: w_ai, w_stat, w_day = 0.30, 0.50, 0.20
        elif data_size < 500: w_ai, w_stat, w_day = 0.40, 0.40, 0.20
        else: w_ai, w_stat, w_day = 0.50, 0.35, 0.15

        backtest_msg = ""
        if self.test_size > 0 and data_size > self.test_size + 30:
            ai_f, st_f, day_f = self.run_backtest(train_X, train_y, df_hist_cut, self.test_size)

            ai_score = max(0.1, 1.0 - (ai_f / self.test_size))**2
            st_score = max(0.1, 1.0 - (st_f / self.test_size))**2
            day_score = max(0.1, 1.0 - (day_f / self.test_size))**2

            w_ai_adj = w_ai * ai_score
            w_st_adj = w_stat * st_score
            w_day_adj = w_day * day_score

            total_adj = w_ai_adj + w_st_adj + w_day_adj
            w_ai, w_stat, w_day = w_ai_adj/total_adj, w_st_adj/total_adj, w_day_adj/total_adj

            backtest_msg = f" (BT-Score: AI {int((1-ai_f/self.test_size)*100)}% | Stat {int((1-st_f/self.test_size)*100)}% | Day {int((1-day_f/self.test_size)*100)}%)"

        last_date = df_hist['date'].iloc[-1].strftime('%Y-%m-%d')
        cache_key = f"{self.lotto_name}_{self.target_col}_{last_date}_ultimate_fs"
        ai_probs = np.zeros(10)

        if cache_key in st.session_state.model_cache:
            trained_models = st.session_state.model_cache[cache_key]
        else:
            trained_models = {}
            for name, model in self.models.items():
                model.fit(train_X, train_y)
                trained_models[name] = model
            st.session_state.model_cache[cache_key] = trained_models

        total_ai_weight = sum(self.model_weights_dict.values())
        for name, model in trained_models.items():
            preds = model.predict_proba(test_X)[0]
            model_probs = np.zeros(10)
            for idx, c in enumerate(model.classes_):
                model_probs[int(c)] = preds[idx]
            ai_probs += model_probs * self.model_weights_dict[name]

        ai_probs /= total_ai_weight
        ai_probs /= (ai_probs.sum() + 1e-9)

        stat_probs = np.zeros(10)
        markov_scores = self.precompute_markov_adaptive(df_hist_cut)
        for d in range(10):
            stat_probs[d] = (0.5 * self.calculate_freq_skip(df_hist_cut, d)) + (0.5 * markov_scores[d])
        stat_probs /= (stat_probs.sum() + 1e-9)

        day_probs = np.zeros(10)
        day_df = df_hist_cut[df_hist_cut['date'].dt.weekday == target_dow]
        if len(day_df) > 0:
            counts = day_df[self.target_col].value_counts(normalize=True)
            for d in range(10): day_probs[d] = counts.get(d, 0.0)
        else:
            day_probs = np.ones(10)/10.0

        final_probs = (w_ai * ai_probs) + (w_stat * stat_probs) + (w_day * day_probs)
        final_probs /= (final_probs.sum() + 1e-9)

        return {
            'ai': ai_probs, 'stat': stat_probs, 'day': day_probs,
            'final': final_probs, 'w_ai': w_ai, 'w_stat': w_stat, 'w_day': w_day,
            'bt_msg': backtest_msg
        }

# ==============================================================================
# ฟังก์ชันคำนวณวันออกงวดถัดไป
# ==============================================================================
def calculate_next_draw_date(last_date, lotto_name):
    """คำนวณวันออกงวดถัดไปให้ถูกต้อง รวมถึงวันหยุดพิเศษของไทย"""
    if any(k in lotto_name for k in ['ไทย', 'ธกส', 'ออมสิน']):
        year = last_date.year
        month = last_date.month
        day = last_date.day

        if month == 12 and day >= 16 and day < 30: return datetime(year, 12, 30)
        elif month == 12 and day == 30: return datetime(year + 1, 1, 17)
        elif month == 1 and day == 17: return datetime(year, 2, 1)
        elif month == 4 and day >= 16: return datetime(year, 5, 2)
        elif month == 5 and day == 2: return datetime(year, 5, 16)
        
        if day < 15: return datetime(year, month, 16)
        else:
            next_month = 1 if month == 12 else month + 1
            next_year = year + 1 if month == 12 else year
            return datetime(next_year, next_month, 1)
    else:
        return last_date + timedelta(days=1)


# ==============================================================================
# 4. Streamlit UI Elements
# ==============================================================================

# หัวแอปพลิเคชันแบบใหม่
st.markdown("""
<div class="main-title">
🛑 LOTTO AI PRO V4
</div>

<div class="sub-title">
Ultimate Candidate Elimination System<br>
AI + XGBoost + Markov + Feature Selection
</div>
""", unsafe_allow_html=True)

# กล่องเลือกข้อมูล
st.markdown('<div class="box">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    selected_lotto = st.selectbox("🎯 เลือกหวย:", list(LOTTERY_URLS.keys()))
with col2:
    day_options = {
        'อัตโนมัติ (คำนวณจากงวดล่าสุด)': None, 'วันจันทร์': 0, 'วันอังคาร': 1, 
        'วันพุธ': 2, 'วันพฤหัสบดี': 3, 'วันศุกร์': 4, 'วันเสาร์': 5, 'วันอาทิตย์': 6
    }
    selected_day_name = st.selectbox("📅 ออกวัน:", list(day_options.keys()))
    target_dow_input = day_options[selected_day_name]
st.markdown("</div>", unsafe_allow_html=True)


def get_dead_numbers(probs_array, k=7):
    return [(idx, probs_array[idx]) for idx in np.argsort(probs_array)[:k]]

def format_dead_output(dead_list):
    return " - ".join([str(num) for num, prob in dead_list])

# ปุ่มกดค้นหา
if st.button("🛑 ค้นหาเลขดับ PRO V4", type="primary", use_container_width=True):
    with st.status("กำลังเชื่อมต่อและประมวลผลข้อมูล...", expanded=True) as status:
        df = fetch_data(selected_lotto)
        
        if df is None or df.empty:
            status.update(label="❌ ไม่สามารถดึงข้อมูลได้ โปรดตรวจสอบการเชื่อมต่อ", state="error")
            st.stop()

        status.write("ข้อมูลพร้อมใช้งาน! เริ่มกระบวนการ Feature Selection...")
        
        sys_status = OptimizedEliminationSystemV4(df, 'hundred', selected_lotto)
        _ = sys_status.analyze(0)  # Dummy run to get selected_feat_count
        
        last_date = df['date'].iloc[-1]
        
        if target_dow_input is not None:
            days_ahead = target_dow_input - last_date.dayofweek
            if days_ahead <= 0: days_ahead += 7
            target_date = last_date + timedelta(days=days_ahead)
            target_dow = target_dow_input
        else:
            # เรียกใช้ฟังก์ชันคำนวณวันที่เราเพิ่งสร้าง
            target_date = calculate_next_draw_date(last_date, selected_lotto)
            target_dow = target_date.weekday()

        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
        data_size = len(df)
        
        store_final_probs = {}
        positions = {
            '💯 3 ตัวบน (ร้อย)': 'hundred', '🔟 3 ตัวบน (สิบ)': 'ten', '1️⃣ 3 ตัวบน (หน่วย)': 'unit',
            '🔽 2 ตัวล่าง (สิบ)': 'bot_ten', '⬇️ 2 ตัวล่าง (หน่วย)': 'bot_unit'
        }
        
        results_output = {}
        for pos_th, col_en in positions.items():
            status.write(f"กำลังสกัดเลขดับ: {pos_th} ...")
            system = OptimizedEliminationSystemV4(df, col_en, selected_lotto)
            res = system.analyze(target_dow)
            if res:
                store_final_probs[col_en] = res['final']
                results_output[pos_th] = res
        
        status.update(label="✨ ประมวลผลเลขดับเสร็จสิ้นสมบูรณ์!", state="complete", expanded=False)

    # --- Render Dashboard ---
    
    # การ์ดสรุปผล
    st.markdown(f"""
    <div class="result">
    🔮 ผลวิเคราะห์เลขดับ ประจำวัน{dow_names[target_dow]}<br>
    วันที่ {target_date.strftime('%d/%m/%Y')}
    </div>
    """, unsafe_allow_html=True)
    
    # กล่องแสดงเลขดับรวม
    if all(k in store_final_probs for k in ['hundred', 'ten', 'unit']):
        top_probs = (store_final_probs['hundred'] + store_final_probs['ten'] + store_final_probs['unit']) / 3.0
        st.markdown(f"""
        <div class="dead">
        🚫 ดับบนรวม (ร้อย-สิบ-หน่วย)<br>
        {format_dead_output(get_dead_numbers(top_probs, 7))}
        </div>
        """, unsafe_allow_html=True)

    if all(k in store_final_probs for k in ['bot_ten', 'bot_unit']):
        bot_probs = (store_final_probs['bot_ten'] + store_final_probs['bot_unit']) / 2.0
        st.markdown(f"""
        <div class="dead" style="background:#431407;">
        ⬇️ ดับล่างรวม (สิบ-หน่วย)<br>
        {format_dead_output(get_dead_numbers(bot_probs, 7))}
        </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # ตัวชี้วัดระบบแบบ 4 คอลัมน์ (Metrics)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 จำนวนงวด", f"{data_size}")
    c2.metric("✂️ Features", f"{sys_status.selected_feat_count}")
    c3.metric("🌲 Trees", f"{sys_status.trees}")
    c4.metric("⚙️ Mode", sys_status.mode_name.split()[0]) # เอาแค่คำว่า Mode X
    
    st.write("---")

    # การตั้งค่าระบบ
    with st.expander("⚙️ ดูรายละเอียดการตั้งค่าระบบและน้ำหนัก AI", expanded=False):
        weights_str = f"RF={sys_status.ai_weights[0]} | ET={sys_status.ai_weights[1]} | HGB={sys_status.ai_weights[2]} | XGB={sys_status.ai_weights[3]}"
        st.write(f"**สเตตัสระบบ [{sys_status.mode_name}]:** 🌲 Trees = {sys_status.trees} | 🔄 BT = {sys_status.test_size} (Max Depth: {sys_status.depth})")
        st.write(f"**โครงสร้างข้อมูล:** Lags {sys_status.lags} | Rolling {sys_status.rolls} | เพิ่มลูกเล่นวงล้อ (Sin/Cos)")
        st.write(f"**การกรอง (FS):** ระบบคัดเฉพาะฟีเจอร์หัวกะทิ Top {sys_status.selected_feat_count} มาใช้ประมวลผล")
        st.write(f"**น้ำหนัก AI 4 สำนัก:** {weights_str}")
        

    st.write("### 📍 เจาะลึกเลขดับในแต่ละหลัก")
    for pos_th, res in results_output.items():
        with st.container():
            w_ai = int(res['w_ai'] * 100)
            w_st = int(res['w_stat'] * 100)
            w_dy = int(res['w_day'] * 100)
            st.markdown(f"#### {pos_th}")
            st.caption(f"น้ำหนักสุทธิ: AI {w_ai}% | Stat {w_st}% | Day {w_dy}% {res['bt_msg']}")
            
            dead_ai = get_dead_numbers(res['ai'], 7)
            dead_day = get_dead_numbers(res['day'], 7)
            dead_stat = get_dead_numbers(res['stat'], 7)
            dead_final = get_dead_numbers(res['final'], 7)

            st.write(f"- 🤖 **ดับ AI:** {format_dead_output(dead_ai)}")
            st.write(f"- 📅 **ดับกำลังวัน:** {format_dead_output(dead_day)}")
            st.write(f"- 📊 **ดับสถิติ:** {format_dead_output(dead_stat)}")
            st.markdown(f"- 🌟 **ดับสรุปรวม 7 ตัว:** **{format_dead_output(dead_final)}**")
            st.write("---")
