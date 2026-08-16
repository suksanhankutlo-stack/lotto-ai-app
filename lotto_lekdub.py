# ==============================================================================
# 🛑 LOTTO AI PRO V7.0 NEURAL SINGULARITY (PREMIUM UI)
# CONSENSUS VARIANCE PENALTY • MTBO Z-SCORE • EXPONENTIAL WF
# CANDIDATE ELIMINATION TOP-7
# ENSEMBLE: ET + RF + HGB + LOGISTIC REGRESSION
# ==============================================================================

import streamlit as st
import requests
import warnings
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. STREAMLIT SETUP
# ==============================================================================

st.set_page_config(
    page_title="ระบบวิเคราะห์เลขดับ PRO V7.0 SINGULARITY",
    page_icon="🛑",
    layout="centered"
)

st.markdown("""
<style>
.main-title {
    text-align:center;
    font-size:32px;
    font-weight:900;
    background: -webkit-linear-gradient(45deg, #000000, #B71C1C, #4A148C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom:5px;
    letter-spacing: 1.5px;
}
.sub-title {
    text-align:center;
    color:#555;
    font-size:14px;
    margin-bottom:20px;
    font-weight: bold;
}
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
# 2. FETCH DATA
# ==============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(lotto_name):
    url = LOTTO_URLS.get(lotto_name)
    if not url: return None
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        post_body = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content")) or soup
        text = post_body.get_text(separator="\n")
        
        pattern = re.compile(r"\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d{2})")
        matches = pattern.findall(text)
        
        if len(matches) < 30: return None
        
        data = []
        for date_str, prize1, bottom2 in matches:
            top = str(prize1).zfill(3)
            bot = str(bottom2).zfill(2)
            data.append({
                "date": date_str, "draw_num": top,
                "hundred": int(top[0]), "ten": int(top[1]), "unit": int(top[2]),
                "bot_ten": int(bot[0]), "bot_unit": int(bot[1])
            })
            
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ ดึงข้อมูลไม่สำเร็จ: {e}")
        return None

# ==============================================================================
# 3. ADAPTIVE CONFIG
# ==============================================================================

def get_adaptive_config(n):
    if n >= 700:
        return {"mode": "SINGULARITY 700+", "trees": 150, "max_depth": 8, "bt_steps": 15, "min_train": 60, "lags": [1, 2, 3, 5, 8, 13], "rolls": [3, 5, 10, 20]}
    elif n >= 400:
        return {"mode": "SINGULARITY 400-699", "trees": 100, "max_depth": 7, "bt_steps": 12, "min_train": 50, "lags": [1, 2, 3, 5, 8], "rolls": [3, 5, 10, 15]}
    elif n >= 200:
        return {"mode": "NEURAL 200-399", "trees": 80, "max_depth": 6, "bt_steps": 10, "min_train": 40, "lags": [1, 2, 3, 5], "rolls": [3, 5, 10]}
    else:
        return {"mode": "NEURAL 30-199", "trees": 50, "max_depth": 5, "bt_steps": 7, "min_train": 30, "lags": [1, 2, 3], "rolls": [3, 5]}

# ==============================================================================
# 4. ADVANCED CAUSAL FEATURE ENGINEERING
# ==============================================================================

@st.cache_data(show_spinner=False)
def build_features_cached(df, target_col, lags, rolls):
    x = df.copy()
    target = pd.to_numeric(x[target_col], errors="coerce")
    
    prev = target.shift(1)
    x["prev_val"] = prev
    x["prev_even"] = (prev % 2 == 0).astype(np.float32)
    x["prev_high"] = (prev >= 5).astype(np.float32)
    x["mirror"] = (prev + 5) % 10
    
    primes = [2, 3, 5, 7]
    x["prev_prime"] = prev.isin(primes).astype(np.float32)
    x["prev_mod3"] = prev % 3
    
    dt = x["date"].dt
    weekday = dt.weekday.astype(float)
    x["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    
    for lag in lags: x[f"lag_{lag}"] = target.shift(lag)
    x["diff_1"] = (prev - target.shift(2)).abs()
    
    shifted = target.shift(1)
    for w in rolls:
        x[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        x[f"roll_std_{w}"] = shifted.rolling(w, min_periods=1).std().fillna(0)
        x[f"ema_{w}"] = shifted.ewm(span=w, adjust=False).mean()

    prev_arr = shifted.to_numpy()
    n = len(prev_arr)
    idx = np.arange(n)
    
    for d in range(10):
        hit = np.where(np.isfinite(prev_arr) & (prev_arr == d), idx, -1)
        last_seen = np.maximum.accumulate(hit)
        skip = np.where(last_seen >= 0, idx - last_seen, 60)
        x[f"skip_{d}"] = np.clip(skip, 0, 60)

    x = x.replace([np.inf, -np.inf], np.nan).fillna(-1)
    return x

# ==============================================================================
# 5. PROBABILITY NORMALIZER & SCALING
# ==============================================================================

def normalize_probs(p, temperature=1.0):
    p = np.asarray(p, dtype=float)
    p[~np.isfinite(p)] = 0
    p = np.maximum(p, 1e-6)
    p = np.power(p, 1 / temperature)
    total = p.sum()
    if total <= 0: return np.ones(10) / 10
    return p / total

def model_probs(model, X):
    try:
        raw = model.predict_proba(X)[0]
        result = np.zeros(10)
        for i, cls in enumerate(model.classes_):
            c = int(cls)
            if 0 <= c <= 9: result[c] = raw[i]
        return normalize_probs(result)
    except Exception:
        return np.ones(10) / 10

# ==============================================================================
# 6. SINGULARITY STATISTICAL SYSTEM
# ==============================================================================

class SingularityStatSystem:
    
    @staticmethod
    def markov_blend(seq):
        seq = np.asarray(seq, dtype=int)
        if len(seq) < 15: return np.ones(10) / 10
        last = int(seq[-1])
        mask = seq[:-1] == last
        next_values = seq[1:][mask]
        counts = np.bincount(next_values, minlength=10).astype(float)
        counts += 0.8 
        return normalize_probs(counts)

    @staticmethod
    def mtbo_skip(seq):
        seq = np.asarray(seq, dtype=int)
        n = len(seq)
        if n == 0: return np.ones(10) / 10
        
        result = np.zeros(10)
        for d in range(10):
            pos = np.where(seq == d)[0]
            if len(pos) > 1:
                gaps = np.diff(pos)
                avg_gap = np.mean(gaps)
                std_gap = np.std(gaps) + 0.1
            else:
                avg_gap, std_gap = 10.0, 5.0
                
            current_gap = n - pos[-1] - 1 if len(pos) else 60
            z = (current_gap - avg_gap) / std_gap
            freq = np.mean(seq == d)
            prob_z = 1 / (1 + np.exp(-z))
            
            result[d] = (0.7 * prob_z) + (0.3 * freq)
        return normalize_probs(result)

    @staticmethod
    def day_probability(df, target_col, target_dow):
        mask = (df["date"].dt.weekday == target_dow)
        values = df.loc[mask, target_col].astype(int).to_numpy()
        if len(values) < 5: return np.ones(10) / 10
        counts = np.bincount(values, minlength=10).astype(float)
        counts += 1.0 
        return normalize_probs(counts)

# ==============================================================================
# 7. NEURAL SINGULARITY AI ENGINE
# ==============================================================================

class SingularityAI:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.n = len(df)
        self.cfg = get_adaptive_config(self.n)
        
        self.models = {
            "LR": make_pipeline(
                StandardScaler(), 
                LogisticRegression(max_iter=300, class_weight='balanced', C=0.5, random_state=42)
            ),
            "ET": ExtraTreesClassifier(n_estimators=self.cfg["trees"], max_depth=self.cfg["max_depth"], min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=43, n_jobs=-1),
            "RF": RandomForestClassifier(n_estimators=self.cfg["trees"], max_depth=self.cfg["max_depth"], min_samples_leaf=2, max_features="log2", class_weight="balanced", random_state=44, n_jobs=-1),
            "HGB": HistGradientBoostingClassifier(max_iter=70, max_depth=min(6, self.cfg["max_depth"]), learning_rate=0.04, min_samples_leaf=3, l2_regularization=1.0, random_state=45)
        }

    def train_predict(self, X_train, y_train, X_predict, weights=None):
        if weights is None: weights = {"LR": 0.20, "ET": 0.30, "RF": 0.30, "HGB": 0.20}
        result = np.zeros(10)
        total = 0.0
        for name, base in self.models.items():
            w = float(weights.get(name, 0))
            if w <= 0: continue
            try:
                model_clone = clone(base)
                model_clone.fit(X_train, y_train)
                p = model_probs(model_clone, X_predict)
                result += p * w
                total += w
            except: continue
        if total <= 0: return np.ones(10) / 10
        return normalize_probs(result / total)

    def walk_forward(self, X, y, df):
        n = len(X)
        min_train, steps = self.cfg["min_train"], self.cfg["bt_steps"]
        if n <= min_train + 2: return {"ai": 0.5, "stat": 0.5, "day": 0.5, "steps": 0}
            
        start = max(min_train, n - steps)
        indices = np.arange(start, n)
        proxy = ExtraTreesClassifier(n_estimators=30, max_depth=5, min_samples_leaf=3, random_state=99, n_jobs=-1)
        ai_hits, stat_hits, day_hits, count = 0, 0, 0, 0
        values = y.to_numpy(dtype=int)
        
        for i in indices:
            X_train, y_train = X.iloc[:i], y.iloc[:i]
            actual = int(y.iloc[i])
            try:
                proxy.fit(X_train, y_train)
                p_ai = model_probs(proxy, X.iloc[[i]])
                if actual in np.argsort(p_ai)[:7]: ai_hits += 1
            except: pass
            
            hist = values[:i]
            p_stat = normalize_probs(0.4 * SingularityStatSystem.markov_blend(hist) + 0.6 * SingularityStatSystem.mtbo_skip(hist))
            if actual in np.argsort(p_stat)[:7]: stat_hits += 1
            
            target_day = int(df.iloc[i]["date"].weekday())
            p_day = SingularityStatSystem.day_probability(df.iloc[:i], self.target_col, target_day)
            if actual in np.argsort(p_day)[:7]: day_hits += 1
            count += 1
            
        if count == 0: return {"ai": 0.5, "stat": 0.5, "day": 0.5, "steps": 0}
        return {"ai": ai_hits/count, "stat": stat_hits/count, "day": day_hits/count, "steps": count}

    def analyze(self, target_date, target_dow):
        if self.n < 30: return None
        
        future = {"date": target_date, "draw_num": "000", "hundred": np.nan, "ten": np.nan, "unit": np.nan, "bot_ten": np.nan, "bot_unit": np.nan}
        extended = pd.concat([self.df, pd.DataFrame([future])], ignore_index=True)
        features = build_features_cached(extended, self.target_col, self.cfg["lags"], self.cfg["rolls"])
        drop_cols = ["date", "draw_num", "hundred", "ten", "unit", "bot_ten", "bot_unit", self.target_col]
        
        X_all = features.iloc[:-1].drop(columns=drop_cols, errors="ignore")
        X_predict = features.iloc[[-1]][X_all.columns]
        y_all = self.df[self.target_col].astype(int)
        
        bt = self.walk_forward(X_all, y_all, self.df)
        
        base_ai, base_stat, base_day = 0.50, 0.35, 0.15
        if bt["steps"] > 0:
            ai_score = np.exp(5 * (bt["ai"] - 0.5))
            stat_score = np.exp(5 * (bt["stat"] - 0.5))
            day_score = np.exp(5 * (bt["day"] - 0.5))
            total = (base_ai * ai_score) + (base_stat * stat_score) + (base_day * day_score)
            w_ai = (base_ai * ai_score) / total
            w_stat = (base_stat * stat_score) / total
            w_day = (base_day * day_score) / total
        else:
            w_ai, w_stat, w_day = base_ai, base_stat, base_day
            
        ai_probs = self.train_predict(X_all, y_all, X_predict)
        seq = y_all.to_numpy(dtype=int)
        p_stat = normalize_probs((0.4 * SingularityStatSystem.markov_blend(seq)) + (0.6 * SingularityStatSystem.mtbo_skip(seq)))
        p_day = SingularityStatSystem.day_probability(self.df, self.target_col, target_dow)
        
        # Consensus Variance Penalty
        mean_probs = (w_ai * ai_probs) + (w_stat * p_stat) + (w_day * p_day)
        stacked_probs = np.vstack([ai_probs, p_stat, p_day])
        std_probs = np.std(stacked_probs, axis=0)
        final_score = mean_probs + (1.5 * std_probs)
        final = normalize_probs(final_score, temperature=1.0)
        
        return {
            "ai": ai_probs, "stat": p_stat, "day": p_day, "final": final,
            "w_ai": w_ai, "w_stat": w_stat, "w_day": w_day, 
            "bt_ai": bt["ai"], "bt_stat": bt["stat"], "bt_day": bt["day"], "bt_steps": bt["steps"],
            "std_max": np.max(std_probs)
        }

# ==============================================================================
# 8. UI HELPERS
# ==============================================================================

def get_dead_numbers(probs, k=7):
    idx = np.argsort(probs)[:k]
    return [(int(i), float(probs[i])) for i in idx]

def format_dead(dead_list):
    return " • ".join(str(num) for num, _ in dead_list)

def target_date_from_last(df, dow_input):
    last_date = df["date"].iloc[-1]
    if dow_input is None:
        gap = (df["date"].iloc[-1] - df["date"].iloc[-2]).days if len(df) >= 2 else 7
        gap = max(1, min(gap, 31))
        target_date = last_date + timedelta(days=gap)
        target_dow = target_date.weekday()
    else:
        target_dow = dow_input
        delta = (target_dow - last_date.weekday()) % 7
        if delta == 0: delta = 7
        target_date = last_date + timedelta(days=delta)
    return target_date, target_dow

# ==============================================================================
# 9. MAIN UI
# ==============================================================================

st.markdown('<div class="main-title">🛑 LOTTO AI PRO V7.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">NEURAL SINGULARITY • CONSENSUS VARIANCE PENALTY</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    target_lotto = st.selectbox("🎯 เลือกหวย", list(LOTTO_URLS.keys()), index=0)
with col2:
    day_options = {
        "อัตโนมัติ (จากงวดล่าสุด)": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, 
        "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6
    }
    day_label = st.selectbox("📅 ออกวัน", list(day_options.keys()), index=0)
    dow_input = day_options[day_label]

if st.button("🛑 วิเคราะห์เลขดับ 7 ตัว ⚡ SINGULARITY RUN", type="primary", use_container_width=True):
    with st.spinner("⚡ กำลังคำนวณ Variance Penalty และ MTBO Z-Score..."):
        df = fetch_data(target_lotto)
        if df is None or df.empty:
            st.error("❌ ไม่สามารถดึงข้อมูลได้")
            st.stop()

        target_date, target_dow = target_date_from_last(df, dow_input)
        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
        
        st.info(f"📅 **งวดเป้าหมาย:** วัน{dow_names[target_dow]} {target_date.strftime('%d/%m/%Y')} | อ้างอิง {len(df)} งวด")
        
        cfg = get_adaptive_config(len(df))
        st.caption(f"⚙️ {cfg['mode']} | Linear+Trees Ensemble | Consensus Penalty Applied")

        positions = {
            "💯 3 ตัวบน (ร้อย)": "hundred", "🔟 3 ตัวบน (สิบ)": "ten", "1️⃣ 3 ตัวบน (หน่วย)": "unit",
            "🔽 2 ตัวล่าง (สิบ)": "bot_ten", "⬇️ 2 ตัวล่าง (หน่วย)": "bot_unit"
        }

        store_final_probs = {}
        progress = st.progress(0, text="Init Singularity Engine...")
        
        for pos_idx, (position_name, col) in enumerate(positions.items(), start=1):
            progress.progress(pos_idx / len(positions), text=f"⚡ วิเคราะห์ {position_name}")
            
            system = SingularityAI(df, col)
            result = system.analyze(target_date, target_dow)
            if result is None: continue
            
            store_final_probs[col] = result["final"]
            
            dead_final = get_dead_numbers(result["final"], 7)
            dead_ai = get_dead_numbers(result["ai"], 7)
            dead_stat = get_dead_numbers(result["stat"], 7)
            dead_day = get_dead_numbers(result["day"], 7)
            
            # -------------------------------------------------------------
            # NEW PREMIUM UI INJECTION
            # -------------------------------------------------------------
            st.markdown(
                f"""
                <div style="background: #ffffff; border-radius: 12px; border: 1px solid #e0e0e0; box-shadow: 0 8px 20px rgba(0,0,0,0.06); padding: 20px; margin-bottom: 20px;">
                    
                    <!-- Header: ตำแหน่ง / ชื่อ -->
                    <div style="font-size: 20px; font-weight: 900; color: #222; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; margin-bottom: 15px;">
                        {position_name}
                    </div>
                    
                    <!-- ส่วนที่ 1: เลขดับเอกฉันท์ -->
                    <div style="background: linear-gradient(135deg, #fff5f5, #ffebee); border: 2px solid #ffcdd2; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px;">
                        <div style="color: #D32F2F; font-weight: 800; font-size: 15px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">
                            🚫 ดับเอกฉันท์ 7 ตัว (Consensus Top-7)
                        </div>
                        <div style="font-size: 36px; font-weight: 900; color: #B71C1C; letter-spacing: 8px; text-shadow: 1px 1px 0px rgba(255,255,255,0.8);">
                            {format_dead(dead_final)}
                        </div>
                    </div>

                    <!-- ส่วนที่ 2: ที่มาของเลข -->
                    <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;">
                        
                        <div style="display: flex; justify-content: space-between; align-items: center; background: #f8f9fa; padding: 12px 15px; border-radius: 8px; border-left: 4px solid #1976D2;">
                            <span style="font-size: 14px; color: #333;">🤖 <b>AI (LR+ET+RF+HGB)</b></span>
                            <span style="background: #E3F2FD; color: #1565C0; padding: 6px 15px; border-radius: 20px; font-weight: 800; font-size: 15px; letter-spacing: 3px;">
                                {format_dead(dead_ai)}
                            </span>
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; align-items: center; background: #f8f9fa; padding: 12px 15px; border-radius: 8px; border-left: 4px solid #388E3C;">
                            <span style="font-size: 14px; color: #333;">📊 <b>สถิติ (MTBO+Markov)</b></span>
                            <span style="background: #E8F5E9; color: #2E7D32; padding: 6px 15px; border-radius: 20px; font-weight: 800; font-size: 15px; letter-spacing: 3px;">
                                {format_dead(dead_stat)}
                            </span>
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; align-items: center; background: #f8f9fa; padding: 12px 15px; border-radius: 8px; border-left: 4px solid #F57C00;">
                            <span style="font-size: 14px; color: #333;">📅 <b>วัน</b></span>
                            <span style="background: #FFF3E0; color: #E65100; padding: 6px 15px; border-radius: 20px; font-weight: 800; font-size: 15px; letter-spacing: 3px;">
                                {format_dead(dead_day)}
                            </span>
                        </div>

                    </div>

                    <!-- ส่วนที่ 3: Analytics Dashboard -->
                    <div style="background: #263238; border-radius: 10px; padding: 15px; color: #CFD8DC; font-family: sans-serif;">
                        <div style="text-align: center; color: #ffffff; font-weight: bold; font-size: 14px; margin-bottom: 12px; letter-spacing: 0.5px;">
                            ⚡ Walk-Forward Analysis (WF {result['bt_steps']} งวด)
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; text-align: center; border-bottom: 1px solid #37474F; padding-bottom: 12px; margin-bottom: 12px;">
                            <div style="flex: 1;">
                                <div style="color: #64B5F6; font-size: 12px; margin-bottom: 4px;">🤖 ความแม่น AI</div>
                                <div style="font-size: 18px; color: #ffffff; font-weight: 900;">{(result['bt_ai']*100):.0f}%</div>
                                <div style="font-size: 11px; color: #90A4AE; margin-top: 2px;">น้ำหนัก: {(result['w_ai']*100):.0f}%</div>
                            </div>
                            <div style="flex: 1; border-left: 1px solid #37474F; border-right: 1px solid #37474F;">
                                <div style="color: #81C784; font-size: 12px; margin-bottom: 4px;">📊 ความแม่นสถิติ</div>
                                <div style="font-size: 18px; color: #ffffff; font-weight: 900;">{(result['bt_stat']*100):.0f}%</div>
                                <div style="font-size: 11px; color: #90A4AE; margin-top: 2px;">น้ำหนัก: {(result['w_stat']*100):.0f}%</div>
                            </div>
                            <div style="flex: 1;">
                                <div style="color: #FFB74D; font-size: 12px; margin-bottom: 4px;">📅 ความแม่นวัน</div>
                                <div style="font-size: 18px; color: #ffffff; font-weight: 900;">{(result['bt_day']*100):.0f}%</div>
                                <div style="font-size: 11px; color: #90A4AE; margin-top: 2px;">น้ำหนัก: {(result['w_day']*100):.0f}%</div>
                            </div>
                        </div>
                        
                        <div style="text-align: center; color: #FF8A65; font-size: 13px; font-weight: 700;">
                            ⚠️ Max Disagreement (Penalty): +{(result['std_max']*100):.1f}
                        </div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )
            
        progress.empty()

        # ==============================================================================
        # SUMMARY
        # ==============================================================================
        st.subheader("🔥 สรุปเลขดับเอกฉันท์ (Top/Bottom)")
        col_sum1, col_sum2 = st.columns(2)

        if all(x in store_final_probs for x in ["hundred", "ten", "unit"]):
            top_probs = normalize_probs((store_final_probs["hundred"] + store_final_probs["ten"] + store_final_probs["unit"]) / 3.0)
            dead_top = get_dead_numbers(top_probs, 7)
            with col_sum1:
                st.markdown(
                    f"""
                    <div style="background:#fff5f5; padding:20px; border-radius:12px; border:2px solid #ffcdd2; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                        <div style="font-weight:900; color:#d32f2f; font-size: 18px;">🚫 ดับบนรวม 7 ตัว</div>
                        <div style="font-size:28px; font-weight:900; color:#B71C1C; margin-top:15px; letter-spacing:4px;">
                            {format_dead(dead_top)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

        if all(x in store_final_probs for x in ["bot_ten", "bot_unit"]):
            bot_probs = normalize_probs((store_final_probs["bot_ten"] + store_final_probs["bot_unit"]) / 2.0)
            dead_bot = get_dead_numbers(bot_probs, 7)
            with col_sum2:
                st.markdown(
                    f"""
                    <div style="background:#fff5f5; padding:20px; border-radius:12px; border:2px solid #ffcdd2; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                        <div style="font-weight:900; color:#d32f2f; font-size: 18px;">🚫 ดับล่างรวม 7 ตัว</div>
                        <div style="font-size:28px; font-weight:900; color:#B71C1C; margin-top:15px; letter-spacing:4px;">
                            {format_dead(dead_bot)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )
                
        st.divider()
        st.caption("🛡️ V7.0 SINGULARITY CORE: Consensus Variance Penalty • Exponential Walk-Forward Weighting • MTBO Z-Score Stats • Logistic Regression Included")
