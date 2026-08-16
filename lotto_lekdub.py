# ==============================================================================
# 🛑 LOTTO AI PRO V4.2 (Upgraded Dead Number Logic - HIGH ACCURACY & CLEAR UI)
# CANDIDATE ELIMINATION - 7 DEAD
# ==============================================================================

import streamlit as st
import requests
import warnings
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. STREAMLIT SETUP & UI STYLING
# ==============================================================================

st.set_page_config(
    page_title="ระบบวิเคราะห์เลขดับ PRO V4.2",
    page_icon="🛑",
    layout="centered"
)

st.markdown("""
<style>
    .main-title { text-align:center; font-size:32px; font-weight:900; color: #B71C1C; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); margin-bottom: 5px; }
    .sub-title { text-align:center; color:#555; font-size:15px; margin-bottom: 25px; font-weight: 500; }
    
    /* กล่องการ์ดสำหรับเลขดับ */
    .dead-card { 
        background: linear-gradient(135deg, #ffffff, #fff5f5); 
        border-left: 6px solid #D32F2F; 
        padding: 20px; 
        border-radius: 12px; 
        margin-bottom: 20px; 
        box-shadow: 0 4px 10px rgba(211, 47, 47, 0.1); 
    }
    
    .position-title { font-size: 20px; font-weight: 800; color: #333; border-bottom: 2px solid #eee; padding-bottom: 8px; margin-bottom: 15px; }
    
    /* ตัวเลขดับ 7 ตัว */
    .dead-number-highlight { 
        font-size: 32px; 
        font-weight: 900; 
        color: #D32F2F; 
        letter-spacing: 4px; 
        text-align: center;
        margin: 10px 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* ป้ายกำกับ (Badges) */
    .badge { padding: 4px 12px; border-radius: 20px; font-size: 15px; font-weight: 700; border: 1px solid rgba(0,0,0,0.05); }
    .badge-ai { background: #E3F2FD; color: #1565C0; }
    .badge-stat { background: #E8F5E9; color: #2E7D32; }
    .badge-day { background: #FFF3E0; color: #E65100; }
    
    .info-row { margin: 8px 0; display: flex; align-items: center; font-size: 14px;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. DATA SOURCE
# ==============================================================================

LOTTO_URLS = {
    "หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "หวยธกส": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
}


# ==============================================================================
# 2. WEB SCRAPER
# ==============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(lotto_name):
    if lotto_name not in LOTTO_URLS: return None
    try:
        response = requests.get(
            LOTTO_URLS[lotto_name],
            headers={"User-Agent": "Mozilla/5.0 (Linux; Android 10)"},
            timeout=15
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        post_body = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content")) or soup
        
        matches = re.compile(r"\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d{2})").findall(post_body.get_text(separator="\n"))
        
        data = []
        for date_str, prize1, bot2 in matches:
            p1, p2 = str(prize1).zfill(3), str(bot2).zfill(2)
            data.append({
                "date": date_str, "draw_num": p1,
                "hundred": int(p1[0]), "ten": int(p1[1]), "unit": int(p1[2]),
                "bot_ten": int(p2[0]), "bot_unit": int(p2[1])
            })

        if len(data) < 30: return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ ดึงข้อมูลไม่สำเร็จ: {e}")
        return None


# ==============================================================================
# 3. ADAPTIVE CONFIG
# ==============================================================================

def get_adaptive_config(n):
    if n >= 700:
        return {"mode": "Mode 4 (700+ งวด)", "trees": 120, "test_size": 35, "early_stop": 15, "lags": [1, 2, 3, 5, 8, 13], "rolls": [3, 5, 10, 20], "rf": 1.0, "et": 1.0, "hgb": 1.0, "xgb": 1.0}
    elif n >= 400:
        return {"mode": "Mode 3 (400-699 งวด)", "trees": 100, "test_size": 30, "early_stop": 13, "lags": [1, 2, 3, 5, 8, 13], "rolls": [3, 5, 10, 20], "rf": 1.0, "et": 0.9, "hgb": 0.8, "xgb": 1.0}
    elif n >= 200:
        return {"mode": "Mode 2 (200-399 งวด)", "trees": 80, "test_size": 25, "early_stop": 10, "lags": [1, 2, 3, 5, 8], "rolls": [3, 5, 10, 20], "rf": 1.0, "et": 0.8, "hgb": 0.6, "xgb": 0.5}
    else:
        return {"mode": "Mode 1 (30-199 งวด)", "trees": 60, "test_size": 15, "early_stop": 8, "lags": [1, 2, 3, 5], "rolls": [3, 5, 10], "rf": 1.0, "et": 0.8, "hgb": 0.5, "xgb": 0.1}


# ==============================================================================
# 4. ENHANCED FEATURE ENGINEERING
# ==============================================================================

def build_features(df, target_col, lags, rolls):
    df_feat = df.copy()
    n = len(df_feat)

    prev = df_feat[target_col].shift(1)
    df_feat["prev_val"] = prev
    df_feat["mirror"] = ((prev + 5) % 10)
    df_feat["is_even"] = (prev % 2 == 0).astype(int)
    df_feat["is_high"] = (prev >= 5).astype(int)

    # Cyclic Time Features
    df_feat["weekday_sin"], df_feat["weekday_cos"] = np.sin(2 * np.pi * df_feat["date"].dt.weekday / 7.0), np.cos(2 * np.pi * df_feat["date"].dt.weekday / 7.0)
    df_feat["month_sin"], df_feat["month_cos"] = np.sin(2 * np.pi * df_feat["date"].dt.month / 12.0), np.cos(2 * np.pi * df_feat["date"].dt.month / 12.0)

    for lag in lags: df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

    # Enhanced Rolling (Added EWM to capture recent volatility)
    for w in rolls:
        shifted = df_feat[target_col].shift(1)
        df_feat[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        df_feat[f"rolling_std_{w}"] = shifted.rolling(w).std().fillna(0)
        df_feat[f"ewm_mean_{w}"] = shifted.ewm(span=w, adjust=False).mean()

    # SKIP calculation
    history = df_feat[target_col].values
    for d in range(10):
        skip_values = np.full(n, 100.0)
        last_seen = -1
        for i in range(n):
            if last_seen >= 0: skip_values[i] = (i - last_seen)
            if history[i] == d: last_seen = i
        df_feat[f"skip_{d}"] = skip_values

    return df_feat.fillna(-1)


# ==============================================================================
# 5. OPTIMIZED ELIMINATION SYSTEM
# ==============================================================================

class OptimizedEliminationSystemV4:
    def __init__(self, df, target_col, lotto_name):
        self.df, self.target_col, self.lotto_name, self.n = df.copy(), target_col, lotto_name, len(df)
        self.cfg = get_adaptive_config(self.n)
        self.trees, self.test_size, self.early_stop, self.lags, self.rolls = self.cfg["trees"], min(self.cfg["test_size"], max(0, self.n - 30)), self.cfg["early_stop"], self.cfg["lags"], self.cfg["rolls"]
        self.ai_weights = (self.cfg["rf"], self.cfg["et"], self.cfg["hgb"], self.cfg["xgb"] if XGBClassifier else 0)
        self.models = self.create_models()

    def create_models(self):
        models = {
            "rf": RandomForestClassifier(n_estimators=self.trees, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "et": ExtraTreesClassifier(n_estimators=self.trees, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "hgb": HistGradientBoostingClassifier(max_iter=80, max_depth=4, learning_rate=0.04, min_samples_leaf=4, l2_regularization=0.6, random_state=42)
        }
        if XGBClassifier:
            models["xgb"] = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, tree_method="hist", verbosity=0, random_state=42, n_jobs=-1)
        return models

    @staticmethod
    def convert_probs(model, probs):
        result = np.zeros(10)
        for idx, cls in enumerate(model.classes_):
            if 0 <= int(cls) <= 9: result[int(cls)] = probs[idx]
        return result / result.sum() if result.sum() > 0 else np.ones(10) / 10

    def train_ai(self, X_train, y_train, X_predict):
        ai_probs, total_weight = np.zeros(10), 0.0
        for idx, (name, base_model) in enumerate(self.models.items()):
            weight = self.ai_weights[idx]
            if weight <= 0: continue
            try:
                model = type(base_model)(**base_model.get_params())
                model.fit(X_train, y_train)
                ai_probs += (self.convert_probs(model, model.predict_proba(X_predict)[0]) * weight)
                total_weight += weight
            except: pass
        return (ai_probs / total_weight) if total_weight > 0 else np.ones(10) / 10

    def markov(self, df_hist):
        seq = df_hist[self.target_col].astype(int).values
        if len(seq) < 5: return np.ones(10) / 10
        last1 = seq[-1]
        p1 = np.bincount(seq[1:][seq[:-1] == last1], minlength=10).astype(float)
        p1 = p1 / p1.sum() if p1.sum() > 0 else np.ones(10) / 10
        return p1

    def freq_skip(self, df_hist):
        # Enhancements: Weigh recent history more aggressively
        series = df_hist[self.target_col].astype(int).values
        recent_series = series[-30:] if len(series) >= 30 else series
        
        result = np.zeros(10)
        for d in range(10):
            freq_all = np.sum(series == d) / max(len(series), 1)
            freq_recent = np.sum(recent_series == d) / max(len(recent_series), 1)
            
            positions = np.where(series == d)[0]
            skip = (len(series) - positions[-1] - 1) if len(positions) > 0 else 100
            
            norm_freq = min((freq_all * 0.4 + freq_recent * 0.6) * 10, 1.0)
            norm_skip = max(1.0 - skip / 30.0, 0.0)
            result[d] = (0.5 * norm_freq + 0.5 * norm_skip)
        return result / result.sum() if result.sum() > 0 else np.ones(10) / 10

    def day_probability(self, df_hist, target_dow):
        day_df = df_hist[df_hist["date"].dt.weekday == target_dow]
        if len(day_df) == 0: return np.ones(10) / 10
        probs = np.zeros(10)
        for k, v in day_df[self.target_col].value_counts(normalize=True).items(): probs[k] = v
        return probs / probs.sum() if probs.sum() > 0 else np.ones(10) / 10

    def run_backtest(self, X_all, y_all, df_all):
        if self.test_size <= 0 or len(X_all) <= self.test_size + 30: return {"ai": 0.5, "stat": 0.5, "day": 0.5, "steps": 0}
        ai_hits, stat_hits, day_hits, steps = 0, 0, 0, 0

        for i in range(len(X_all) - self.test_size, len(X_all)):
            X_train, y_train, X_test, actual, hist = X_all.iloc[:i], y_all.iloc[:i], X_all.iloc[[i]], int(y_all.iloc[i]), df_all.iloc[:i].copy()
            
            dead_ai = np.argsort(self.train_ai(X_train, y_train, X_test))[:7]
            if actual not in dead_ai: ai_hits += 1

            stat = 0.5 * self.markov(hist) + 0.5 * self.freq_skip(hist)
            if actual not in np.argsort(stat)[:7]: stat_hits += 1
            if actual not in np.argsort(self.day_probability(hist, df_all.iloc[i]["date"].weekday()))[:7]: day_hits += 1

            steps += 1
            if steps >= self.early_stop: break

        return {"ai": ai_hits/steps, "stat": stat_hits/steps, "day": day_hits/steps, "steps": steps} if steps > 0 else {"ai": 0.5, "stat": 0.5, "day": 0.5, "steps": 0}

    def analyze(self, target_date, target_dow):
        if self.n < 30: return None
        df_extended = pd.concat([self.df, pd.DataFrame([{"date": target_date, "draw_num": "000", "hundred": 0, "ten": 0, "unit": 0, "bot_ten": 0, "bot_unit": 0}])], ignore_index=True)
        df_feat = build_features(df_extended, self.target_col, self.lags, self.rolls)

        X_all = df_feat.iloc[:-1].drop(columns=["date", "draw_num", "hundred", "ten", "unit", "bot_ten", "bot_unit", self.target_col], errors='ignore')
        bt = self.run_backtest(X_all, self.df[self.target_col].astype(int), self.df)

        # Walk-Forward Weighting (Aggressive scaling for better accuracy)
        w_ai, w_stat, w_day = 0.50, 0.35, 0.15
        if bt["steps"] > 0:
            wa, ws, wd = w_ai * (max(0.10, bt["ai"])**3), w_stat * (max(0.10, bt["stat"])**3), w_day * (max(0.10, bt["day"])**3)
            tot = wa + ws + wd
            if tot > 0: w_ai, w_stat, w_day = wa/tot, ws/tot, wd/tot

        ai_probs = self.train_ai(X_all, self.df[self.target_col].astype(int), df_feat.iloc[[-1]][X_all.columns])
        stat_probs = (0.5 * self.markov(self.df) + 0.5 * self.freq_skip(self.df))
        day_probs = self.day_probability(self.df, target_dow)

        final_probs = (w_ai * ai_probs + w_stat * stat_probs + w_day * day_probs)
        final_probs /= final_probs.sum()

        bt_msg = f"📈 Walk-Forward {bt['steps']} งวด | อัตราการคัดเลขดับแม่นยำ: AI {bt['ai']:.0%} | สถิติ {bt['stat']:.0%} | วัน {bt['day']:.0%}"
        return {"ai": ai_probs, "stat": stat_probs, "day": day_probs, "final": final_probs, "w_ai": w_ai, "w_stat": w_stat, "w_day": w_day, "bt_msg": bt_msg}


# ==============================================================================
# 6. HELPERS FOR UI
# ==============================================================================

def get_dead_numbers(probs, k=7):
    return [(int(i), float(probs[i])) for i in np.argsort(probs)[:k]]

def format_html_dead(dead_list):
    return " &nbsp;•&nbsp; ".join(str(num) for num, prob in dead_list)


# ==============================================================================
# 7. MAIN APPLICATION RUN
# ==============================================================================

st.markdown('<div class="main-title">🛑 ระบบวิเคราะห์เลขดับ PRO V4.2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Candidate Elimination (7 ตัวดับ) • Tuned for High Accuracy</div>', unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)
with col1:
    target_lotto = st.selectbox("🎯 เลือกหวย:", list(LOTTO_URLS.keys()), index=0)
with col2:
    day_options = {"อัตโนมัติ (จากงวดล่าสุด)": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6}
    dow_input = day_options[st.selectbox("📅 ออกวัน:", list(day_options.keys()), index=0)]

if st.button("🛑 วิเคราะห์เลขดับ 7 ตัว", type="primary", use_container_width=True):
    with st.spinner("⏳ AI กำลังดึงข้อมูลและประมวลผลการคัดแยกความน่าจะเป็นต่ำสุด..."):
        df = fetch_data(target_lotto)

        if df is None or df.empty:
            st.error("❌ ไม่สามารถดึงข้อมูลได้ โปรดลองใหม่อีกครั้ง")
            st.stop()

        last_date = df["date"].iloc[-1]
        target_dow = dow_input if dow_input is not None else (last_date + timedelta(days=max(1, (last_date - df["date"].iloc[-2]).days) if len(df)>=2 else 7)).weekday()
        target_date = last_date + timedelta(days=(target_dow - last_date.weekday()) % 7 if dow_input is not None else max(1, (last_date - df["date"].iloc[-2]).days) if len(df)>=2 else 7)
        if target_date <= last_date: target_date += timedelta(days=7)

        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

        st.info(f"📅 **วิเคราะห์เป้าหมาย:** วัน{dow_names[target_dow]} ที่ {target_date.strftime('%d/%m/%Y')} (วิเคราะห์จากข้อมูลอ้างอิง {len(df)} งวด)")

        positions = {"💯 3 ตัวบน (ร้อย)": "hundred", "🔟 3 ตัวบน (สิบ)": "ten", "1️⃣ 3 ตัวบน (หน่วย)": "unit", "🔽 2 ตัวล่าง (สิบ)": "bot_ten", "⬇️ 2 ตัวล่าง (หน่วย)": "bot_unit"}
        store_final_probs = {}

        for position_name, col in positions.items():
            result = OptimizedEliminationSystemV4(df, col, target_lotto).analyze(target_date, target_dow)
            if not result: continue

            store_final_probs[col] = result["final"]
            dead_final = get_dead_numbers(result["final"], 7)
            
            # การ์ดแสดงผล
            st.markdown(f'''
            <div class="dead-card">
                <div class="position-title">{position_name}</div>
                <div style="text-align:center; color:#555; font-weight:bold;">🚫 สรุปดับ 7 ตัว</div>
                <div class="dead-number-highlight">{format_html_dead(dead_final)}</div>
                
                <div style="margin-top:20px; border-top: 1px dashed #ddd; padding-top:15px;">
                    <div class="info-row">🤖 <span style="width:100px; display:inline-block; margin-left:5px;"><b>AI คัดดับ:</b></span> <span class="badge badge-ai">{format_html_dead(get_dead_numbers(result["ai"], 7))}</span></div>
                    <div class="info-row">📊 <span style="width:100px; display:inline-block; margin-left:5px;"><b>สถิติคัดดับ:</b></span> <span class="badge badge-stat">{format_html_dead(get_dead_numbers(result["stat"], 7))}</span></div>
                    <div class="info-row">📅 <span style="width:100px; display:inline-block; margin-left:5px;"><b>วันคัดดับ:</b></span> <span class="badge badge-day">{format_html_dead(get_dead_numbers(result["day"], 7))}</span></div>
                </div>
                
                <div style="margin-top:10px; font-size:12px; color:#888;">
                    {result["bt_msg"]}<br>
                    ⚖️ น้ำหนักการตัดสินใจ: AI {result["w_ai"]:.0%} | สถิติ {result["w_stat"]:.0%} | กำลังวัน {result["w_day"]:.0%}
                </div>
            </div>
            ''', unsafe_allow_html=True)

        # สรุปภาพรวม
        st.subheader("🔥 สรุปภาพรวมเลขดับ (ดับบนรวม / ดับล่างรวม)")
        col_sum1, col_sum2 = st.columns(2)
        
        if all(x in store_final_probs for x in ["hundred", "ten", "unit"]):
            top_probs = (store_final_probs["hundred"] + store_final_probs["ten"] + store_final_probs["unit"])
            with col_sum1:
                st.markdown(f'''
                <div style="background:#fff3f3; padding:15px; border-radius:10px; border: 2px solid #ffcdd2; text-align:center;">
                    <div style="font-weight:bold; color:#d32f2f;">🚫 ดับบนรวม 7 ตัว</div>
                    <div style="font-size:24px; font-weight:900; color:#b71c1c; margin-top:10px; letter-spacing:2px;">{format_html_dead(get_dead_numbers(top_probs, 7))}</div>
                </div>
                ''', unsafe_allow_html=True)

        if all(x in store_final_probs for x in ["bot_ten", "bot_unit"]):
            bot_probs = (store_final_probs["bot_ten"] + store_final_probs["bot_unit"])
            with col_sum2:
                st.markdown(f'''
                <div style="background:#fff3f3; padding:15px; border-radius:10px; border: 2px solid #ffcdd2; text-align:center;">
                    <div style="font-weight:bold; color:#d32f2f;">🚫 ดับล่างรวม 7 ตัว</div>
                    <div style="font-size:24px; font-weight:900; color:#b71c1c; margin-top:10px; letter-spacing:2px;">{format_html_dead(get_dead_numbers(bot_probs, 7))}</div>
                </div>
                ''', unsafe_allow_html=True)
