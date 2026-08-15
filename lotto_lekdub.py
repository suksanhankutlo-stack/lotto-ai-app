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

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# 🛑 LOTTO AI PRO V4.2 (Upgraded Dead Number Logic)
# CANDIDATE ELIMINATION - 7 DEAD
#
# SEQUENTIAL / NO MEMORY
# ------------------------------------------------------------------------------
# ✅ RF (Tuned for anti-overfitting)
# ✅ ExtraTrees (Tuned for anti-overfitting)
# ✅ HistGradientBoosting (Tuned for anti-overfitting)
# ✅ XGBoost (Tuned for anti-overfitting)
# ✅ Walk-Forward Backtest (Evaluated on Dead Number Survival)
# ✅ Cyclic Time Features (Sin/Cos)
# ✅ Frequency + Skip
# ✅ Markov
# ✅ Day-of-Week
# ✅ Adaptive Lags / Rolling
# ✅ 7 Dead Numbers
# ==============================================================================


# ==============================================================================
# 0. STREAMLIT SETUP
# ==============================================================================

st.set_page_config(
    page_title="ระบบวิเคราะห์เลขดับ PRO V4.2",
    page_icon="🛑",
    layout="centered"
)


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

def fetch_data(lotto_name):
    if lotto_name not in LOTTO_URLS:
        return None

    url = LOTTO_URLS[lotto_name]

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent":
                    "Mozilla/5.0 (Linux; Android 10) "
                    "AppleWebKit/537.36 Chrome/120 Mobile Safari/537.36"
            },
            timeout=15
        )
        response.raise_for_status()

        if not response.content:
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        post_body = soup.find(
            "div",
            class_=re.compile(r"post-body|entry-content|post-content|content")
        )

        if post_body is None:
            post_body = soup

        text_content = post_body.get_text(separator="\n")

        pattern = re.compile(r"\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d{2})")
        matches = pattern.findall(text_content)

        data = []
        for date_str, prize1, bot2 in matches:
            p1 = str(prize1).zfill(3)
            p2 = str(bot2).zfill(2)

            data.append({
                "date": date_str,
                "draw_num": p1,
                "hundred": int(p1[0]),
                "ten": int(p1[1]),
                "unit": int(p1[2]),
                "bot_ten": int(p2[0]),
                "bot_unit": int(p2[1])
            })

        if len(data) < 30:
            return None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except Exception as e:
        st.error(f"❌ ดึงข้อมูลไม่สำเร็จ: {e}")
        return None


# ==============================================================================
# 3. ADAPTIVE CONFIG
# ==============================================================================

def get_adaptive_config(n):
    if n >= 700:
        return {
            "mode": "Mode 4 (700+ งวด)",
            "trees": 100,
            "test_size": 30,
            "early_stop": 15,
            "lags": [1, 2, 3, 5, 8, 13],
            "rolls": [3, 5, 10, 20],
            "rf": 1.0,
            "et": 1.0,
            "hgb": 1.0,
            "xgb": 1.0
        }
    elif n >= 400:
        return {
            "mode": "Mode 3 (400-699 งวด)",
            "trees": 100,
            "test_size": 25,
            "early_stop": 13,
            "lags": [1, 2, 3, 5, 8, 13],
            "rolls": [3, 5, 10, 20],
            "rf": 1.0,
            "et": 0.9,
            "hgb": 0.8,
            "xgb": 1.0
        }
    elif n >= 200:
        return {
            "mode": "Mode 2 (200-399 งวด)",
            "trees": 80,
            "test_size": 20,
            "early_stop": 10,
            "lags": [1, 2, 3, 5, 8],
            "rolls": [3, 5, 10, 20],
            "rf": 1.0,
            "et": 0.8,
            "hgb": 0.6,
            "xgb": 0.5
        }
    else:
        return {
            "mode": "Mode 1 (30-199 งวด)",
            "trees": 60,
            "test_size": 15,
            "early_stop": 8,
            "lags": [1, 2, 3, 5],
            "rolls": [3, 5, 10],
            "rf": 1.0,
            "et": 0.8,
            "hgb": 0.5,
            "xgb": 0.1
        }


# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================

def build_features(df, target_col, lags, rolls):
    df_feat = df.copy()
    n = len(df_feat)

    # ------------------------------------------------------------------
    # Previous value
    # ------------------------------------------------------------------
    df_feat["prev_val"] = df_feat[target_col].shift(1)
    prev = df_feat["prev_val"]

    # ------------------------------------------------------------------
    # Basic features
    # ------------------------------------------------------------------
    df_feat["mirror"] = ((prev + 5) % 10)
    df_feat["is_even"] = (prev % 2 == 0).astype(int)
    df_feat["is_high"] = (prev >= 5).astype(int)
    df_feat["mod3"] = (prev % 3)

    df_feat["weekday"] = df_feat["date"].dt.weekday
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["day"] = df_feat["date"].dt.day

    # ------------------------------------------------------------------
    # Cyclic Time Features (ป้องกัน AI เข้าใจเวลาผิดพลาด)
    # ------------------------------------------------------------------
    df_feat["weekday_sin"] = np.sin(2 * np.pi * df_feat["weekday"] / 7.0)
    df_feat["weekday_cos"] = np.cos(2 * np.pi * df_feat["weekday"] / 7.0)
    
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)
    
    df_feat["day_sin"] = np.sin(2 * np.pi * df_feat["day"] / 31.0)
    df_feat["day_cos"] = np.cos(2 * np.pi * df_feat["day"] / 31.0)

    # ------------------------------------------------------------------
    # LAGS
    # ------------------------------------------------------------------
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

    # ------------------------------------------------------------------
    # Repeat
    # ------------------------------------------------------------------
    if "lag_1" in df_feat.columns:
        df_feat["repeat_2"] = (df_feat["lag_1"] == df_feat.get("lag_2", -999)).astype(int)

    if ("lag_1" in df_feat.columns and "lag_2" in df_feat.columns and "lag_3" in df_feat.columns):
        df_feat["repeat_3"] = ((df_feat["lag_1"] == df_feat["lag_2"]) & 
                               (df_feat["lag_2"] == df_feat["lag_3"])).astype(int)

    # ------------------------------------------------------------------
    # Rolling
    # ------------------------------------------------------------------
    for w in rolls:
        shifted = df_feat[target_col].shift(1)
        df_feat[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        df_feat[f"rolling_std_{w}"] = shifted.rolling(w).std()

    # ------------------------------------------------------------------
    # HOT / COLD
    # ------------------------------------------------------------------
    windows = list(rolls)
    if n >= 500 and 50 not in windows:
        windows.append(50)

    history = df_feat[target_col].values

    for w in windows:
        for d in range(10):
            hot_values = np.zeros(n)
            cold_values = np.zeros(n)

            for i in range(n):
                start = max(0, i - w)
                window = history[start:i]
                count = np.sum(window == d)
                hot_values[i] = count
                if w < 50:
                    cold_values[i] = (len(window) - count)

            df_feat[f"hot{w}_{d}"] = hot_values
            if w < 50:
                df_feat[f"cold{w}_{d}"] = cold_values

    # ------------------------------------------------------------------
    # SKIP
    # ------------------------------------------------------------------
    for d in range(10):
        skip_values = np.full(n, 100.0)
        last_seen = -1

        for i in range(n):
            if last_seen >= 0:
                skip_values[i] = (i - last_seen)
            if history[i] == d:
                last_seen = i

        df_feat[f"skip_{d}"] = skip_values

    return df_feat.fillna(-1)


# ==============================================================================
# 5. SYSTEM
# ==============================================================================

class OptimizedEliminationSystemV4:
    def __init__(self, df, target_col, lotto_name):
        self.df = df.copy()
        self.target_col = target_col
        self.lotto_name = lotto_name
        self.n = len(df)
        self.cfg = get_adaptive_config(self.n)
        self.mode_name = self.cfg["mode"]
        self.trees = self.cfg["trees"]
        self.test_size = min(self.cfg["test_size"], max(0, self.n - 30))
        self.early_stop = self.cfg["early_stop"]
        self.lags = self.cfg["lags"]
        self.rolls = self.cfg["rolls"]
        self.ai_weights = (self.cfg["rf"], self.cfg["et"], self.cfg["hgb"], self.cfg["xgb"])
        self.models = self.create_models()

    # ------------------------------------------------------------------
    # Models (Tuned for anti-overfitting & conservative dead-number logic)
    # ------------------------------------------------------------------
    def create_models(self):
        return {
            "rf": RandomForestClassifier(
                n_estimators=self.trees,
                max_depth=4,              
                min_samples_leaf=4,       
                random_state=42,
                n_jobs=1
            ),
            "et": ExtraTreesClassifier(
                n_estimators=self.trees,
                max_depth=4,              
                min_samples_leaf=3,       
                random_state=42,
                n_jobs=1
            ),
            "hgb": HistGradientBoostingClassifier(
                max_iter=50,
                max_depth=3,              
                learning_rate=0.05,       
                min_samples_leaf=5,       
                l2_regularization=0.5,    
                random_state=42
            ),
            "xgb": XGBClassifier(
                n_estimators=50,
                max_depth=2,              
                learning_rate=0.05,       
                subsample=0.8,            
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="mlogloss",
                verbosity=0,
                random_state=42,
                n_jobs=1
            )
        }

    # ------------------------------------------------------------------
    # Probability conversion
    # ------------------------------------------------------------------
    @staticmethod
    def convert_probs(model, probs):
        result = np.zeros(10)
        for idx, cls in enumerate(model.classes_):
            try:
                digit = int(cls)
                if 0 <= digit <= 9:
                    result[digit] = probs[idx]
            except:
                pass

        total = result.sum()
        if total <= 0:
            return np.ones(10) / 10
        return result / total

    # ------------------------------------------------------------------
    # AI probability
    # ------------------------------------------------------------------
    def train_ai(self, X_train, y_train, X_predict):
        ai_probs = np.zeros(10)
        weights = self.ai_weights
        total_weight = 0.0

        for idx, (name, base_model) in enumerate(self.models.items()):
            weight = weights[idx]
            if weight <= 0:
                continue

            model = type(base_model)(**base_model.get_params())
            try:
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_predict)[0]
                model_probs = self.convert_probs(model, probs)
                ai_probs += (model_probs * weight)
                total_weight += weight
            except Exception:
                continue

        if total_weight <= 0:
            return np.ones(10) / 10

        ai_probs /= total_weight
        total = ai_probs.sum()

        if total <= 0:
            return np.ones(10) / 10
        return ai_probs / total

    # ------------------------------------------------------------------
    # Markov
    # ------------------------------------------------------------------
    def markov(self, df_hist):
        seq = df_hist[self.target_col].astype(int).values
        n = len(seq)
        if n < 5:
            return np.ones(10) / 10
        
        last1 = seq[-1]
        
        p1 = np.zeros(10)
        total1 = 0
        for i in range(0, n - 1):
            if seq[i] == last1:
                p1[seq[i + 1]] += 1
                total1 += 1
        
        if total1 > 0:
            p1 /= total1
        else:
            p1[:] = 0.1

        if n < 200:
            return p1

        last2 = seq[-2]
        p2 = np.zeros(10)
        total2 = 0
        for i in range(1, n - 1):
            if seq[i - 1] == last2 and seq[i] == last1:
                p2[seq[i + 1]] += 1
                total2 += 1
        
        if total2 > 0:
            p2 /= total2
        else:
            p2 = p1.copy()

        if n < 500:
            return (0.6 * p2 + 0.4 * p1)

        last3 = seq[-3]
        p3 = np.zeros(10)
        total3 = 0
        for i in range(2, n - 1):
            if (seq[i - 2] == last3 and seq[i - 1] == last2 and seq[i] == last1):
                p3[seq[i + 1]] += 1
                total3 += 1
        
        if total3 > 0:
            p3 /= total3
        else:
            p3 = p2.copy()

        return (0.5 * p3 + 0.3 * p2 + 0.2 * p1)

    # ------------------------------------------------------------------
    # Frequency + Skip
    # ------------------------------------------------------------------
    def freq_skip(self, df_hist):
        result = np.zeros(10)
        series = df_hist[self.target_col].astype(int).values
        n = len(series)

        for d in range(10):
            count = np.sum(series == d)
            freq = (count / max(n, 1))
            positions = np.where(series == d)[0]
            if len(positions) > 0:
                skip = (n - positions[-1] - 1)
            else:
                skip = 100

            norm_freq = min(freq * 10, 1.0)
            norm_skip = max(1.0 - skip / 30.0, 0.0)
            result[d] = (0.5 * norm_freq + 0.5 * norm_skip)

        total = result.sum()
        if total <= 0:
            return np.ones(10) / 10
        return result / total

    # ------------------------------------------------------------------
    # Day of week
    # ------------------------------------------------------------------
    def day_probability(self, df_hist, target_dow):
        day_df = df_hist[df_hist["date"].dt.weekday == target_dow]
        if len(day_df) == 0:
            return np.ones(10) / 10

        counts = day_df[self.target_col].value_counts(normalize=True)
        probs = np.zeros(10)
        for d in range(10):
            probs[d] = counts.get(d, 0.0)

        total = probs.sum()
        if total <= 0:
            return np.ones(10) / 10
        return probs / total

    # ------------------------------------------------------------------
    # WALK FORWARD BACKTEST (Evaluated on "Survival from Dead Numbers")
    # ------------------------------------------------------------------
    def run_backtest(self, X_all, y_all, df_all):
        test_size = self.test_size
        if (test_size <= 0 or len(X_all) <= test_size + 30):
            return {"ai": 0.5, "stat": 0.5, "day": 0.5, "steps": 0}

        start = (len(X_all) - test_size)
        ai_hits = 0
        stat_hits = 0
        day_hits = 0
        steps = 0

        for i in range(start, len(X_all)):
            X_train = X_all.iloc[:i]
            y_train = y_all.iloc[:i]
            X_test = X_all.iloc[[i]]
            actual = int(y_all.iloc[i])
            hist = df_all.iloc[:i].copy()

            if len(hist) < 30:
                continue

            # --------------------------------------------------------------
            # ประเมิน: ให้คะแนนถ้าระบุ "เลขดับ" ถูกต้อง
            # ระบุถูกต้อง = เลขจริง (actual) ไม่อยู่ใน 7 ตัวล่าง (dead numbers)
            # --------------------------------------------------------------
            
            # AI
            ai = self.train_ai(X_train, y_train, X_test)
            dead_ai = np.argsort(ai)[:7] # 7 ตัวที่โอกาสออกน้อยสุด
            if actual not in dead_ai:
                ai_hits += 1

            # STAT
            mk = self.markov(hist)
            fs = self.freq_skip(hist)
            stat = (0.5 * mk + 0.5 * fs)
            stat /= (stat.sum() + 1e-12)
            dead_stat = np.argsort(stat)[:7]
            if actual not in dead_stat:
                stat_hits += 1

            # DAY
            target_dow = df_all.iloc[i]["date"].weekday()
            day = self.day_probability(hist, target_dow)
            dead_day = np.argsort(day)[:7]
            if actual not in dead_day:
                day_hits += 1

            steps += 1
            if steps >= self.early_stop:
                break

        if steps <= 0:
            return {"ai": 0.5, "stat": 0.5, "day": 0.5, "steps": 0}

        return {
            "ai": ai_hits / steps,
            "stat": stat_hits / steps,
            "day": day_hits / steps,
            "steps": steps
        }

    # ------------------------------------------------------------------
    # MAIN ANALYSIS
    # ------------------------------------------------------------------
    def analyze(self, target_date, target_dow):
        if self.n < 30:
            return None

        dummy = {
            "date": target_date,
            "draw_num": "000",
            "hundred": 0,
            "ten": 0,
            "unit": 0,
            "bot_ten": 0,
            "bot_unit": 0
        }
        df_extended = pd.concat([self.df, pd.DataFrame([dummy])], ignore_index=True)
        df_feat = build_features(df_extended, self.target_col, self.lags, self.rolls)

        X_all = df_feat.iloc[:-1].copy()
        X_next = df_feat.iloc[[-1]].copy()
        y_all = self.df[self.target_col].astype(int)

        exclude = ["date", "draw_num", "hundred", "ten", "unit", "bot_ten", "bot_unit"]
        feature_cols = [c for c in X_all.columns if c not in exclude and c != self.target_col]

        X_train = X_all[feature_cols]
        X_predict = X_next[feature_cols]

        bt = self.run_backtest(X_train, y_all, self.df)

        if self.n < 200:
            w_ai, w_stat, w_day = 0.30, 0.50, 0.20
        elif self.n < 500:
            w_ai, w_stat, w_day = 0.40, 0.40, 0.20
        else:
            w_ai, w_stat, w_day = 0.50, 0.35, 0.15

        if bt["steps"] > 0:
            ai_score = max(0.10, bt["ai"]) ** 2
            stat_score = max(0.10, bt["stat"]) ** 2
            day_score = max(0.10, bt["day"]) ** 2

            wa = (w_ai * ai_score)
            ws = (w_stat * stat_score)
            wd = (w_day * day_score)
            total = (wa + ws + wd)

            if total > 0:
                w_ai = wa / total
                w_stat = ws / total
                w_day = wd / total

        ai_probs = self.train_ai(X_train, y_all, X_predict)

        markov_probs = self.markov(self.df)
        freq_probs = self.freq_skip(self.df)
        stat_probs = (0.5 * markov_probs + 0.5 * freq_probs)
        stat_probs /= (stat_probs.sum() + 1e-12)

        day_probs = self.day_probability(self.df, target_dow)

        final_probs = (w_ai * ai_probs + w_stat * stat_probs + w_day * day_probs)
        final_probs /= (final_probs.sum() + 1e-12)

        bt_msg = (
            f"BT-WalkForward {bt['steps']} งวด | "
            f"ความแม่นยำในการคัดเลขดับ: AI {bt['ai']*100:.1f}% | "
            f"Stat {bt['stat']*100:.1f}% | Day {bt['day']*100:.1f}%"
        )

        return {
            "ai": ai_probs,
            "stat": stat_probs,
            "day": day_probs,
            "final": final_probs,
            "w_ai": w_ai,
            "w_stat": w_stat,
            "w_day": w_day,
            "bt_msg": bt_msg
        }


# ==============================================================================
# 6. DEAD NUMBER FUNCTIONS
# ==============================================================================

def get_dead_numbers(probs, k=7):
    idx = np.argsort(probs)[:k]
    return [(int(i), float(probs[i])) for i in idx]

def format_dead_output(dead_list):
    return " - ".join(str(num) for num, prob in dead_list)


# ==============================================================================
# 7. UI
# ==============================================================================

st.title("🛑 ระบบวิเคราะห์เลขดับ PRO V4.2")
st.markdown("**Candidate Elimination - 7 ดับ (Tuned for Dead Numbers)**")
st.caption("Sequential / No Memory / Adjusted Walk-Forward / Cyclic Features")
st.divider()

col1, col2 = st.columns(2)

with col1:
    target_lotto = st.selectbox(
        "🎯 เลือกหวย:",
        list(LOTTO_URLS.keys()),
        index=0, key="dub_1"
    )

with col2:
    day_options = {
        "อัตโนมัติ (คำนวณจากงวดล่าสุด)": None,
        "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3,
        "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6
    }
    selected_day = st.selectbox(
        "📅 ออกวัน:",
        list(day_options.keys()),
        index=0, key="dub_2"
    )
    dow_input = day_options[selected_day]


# ==============================================================================
# ANALYZE BUTTON
# ==============================================================================

if st.button("🛑 วิเคราะห์เลขดับ PRO V4.2", type="primary", use_container_width=True):
    with st.spinner("⏳ กำลังดึงข้อมูล + สร้างโมเดลใหม่ + Walk-Forward..."):
        df = fetch_data(target_lotto)

        if df is None or df.empty:
            st.error("❌ ไม่สามารถดึงข้อมูลได้")
            st.stop()

        last_date = df["date"].iloc[-1]

        if dow_input is not None:
            days_ahead = (dow_input - last_date.weekday())
            if days_ahead <= 0:
                days_ahead += 7
            target_date = (last_date + timedelta(days=days_ahead))
            target_dow = dow_input
        else:
            if len(df) >= 2:
                gap = (df["date"].iloc[-1] - df["date"].iloc[-2]).days
                if gap <= 0: gap = 7
            else:
                gap = 7
            target_date = (last_date + timedelta(days=gap))
            target_dow = target_date.weekday()

        cfg = get_adaptive_config(len(df))

        st.info(f"""
**⚙️ ระบบ [{cfg['mode']}]**
- 📊 ข้อมูลย้อนหลัง: **{len(df)} งวด**
- 🔄 Backtest: ประเมินจาก **การรอดพ้นเลขดับ 7 ตัว**
- 🧭 Features: เพิ่มระบบ **วงจรเวลา (Cyclic)**

### 🔐 โหมดความจำ
**NO MEMORY / NO MODEL CACHE**
เริ่มคำนวณใหม่แบบสดใหม่ 100% ทุกครั้ง
""")

        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

        st.markdown(f"""
### 🔮 ผลการวิเคราะห์เลขดับ
**งวดเป้าหมาย:** วัน{dow_names[target_dow]} ที่ {target_date.strftime('%d/%m/%Y')}
**ข้อมูลอ้างอิง:** {len(df)} งวด
""")
        st.divider()

        positions = {
            "💯 3 ตัวบน (ร้อย)": "hundred",
            "🔟 3 ตัวบน (สิบ)": "ten",
            "1️⃣ 3 ตัวบน (หน่วย)": "unit",
            "🔽 2 ตัวล่าง (สิบ)": "bot_ten",
            "⬇️ 2 ตัวล่าง (หน่วย)": "bot_unit"
        }

        store_final_probs = {}

        for position_name, col in positions.items():
            system = OptimizedEliminationSystemV4(df, col, target_lotto)
            result = system.analyze(target_date, target_dow)

            if result is None:
                st.warning(f"⚠️ ข้อมูลไม่เพียงพอ: {position_name}")
                continue

            store_final_probs[col] = result["final"]

            dead_ai = get_dead_numbers(result["ai"], 7)
            dead_stat = get_dead_numbers(result["stat"], 7)
            dead_day = get_dead_numbers(result["day"], 7)
            dead_final = get_dead_numbers(result["final"], 7)

            w_ai = int(result["w_ai"] * 100)
            w_stat = int(result["w_stat"] * 100)
            w_day = int(result["w_day"] * 100)

            with st.expander(f"📌 {position_name} (AI {w_ai}% | Stat {w_stat}% | Day {w_day}%)", expanded=True):
                st.caption(result["bt_msg"])
                st.markdown(f"- 🤖 **ดับ AI:** `{format_dead_output(dead_ai)}`")
                st.markdown(f"- 📊 **ดับสถิติ:** `{format_dead_output(dead_stat)}`")
                st.markdown(f"- 📅 **ดับกำลังวัน:** `{format_dead_output(dead_day)}`")
                st.success(f"🌟 **ดับสรุปรวม 7 ตัว:** `{format_dead_output(dead_final)}`")

        st.divider()
        st.subheader("🔥 สรุปภาพรวมเลขดับ")

        if all(x in store_final_probs for x in ["hundred", "ten", "unit"]):
            top_probs = (store_final_probs["hundred"] + store_final_probs["ten"] + store_final_probs["unit"]) / 3.0
            top_probs /= (top_probs.sum() + 1e-12)
            st.markdown(f"""
🚫 **ดับบนรวม (ร้อย-สิบ-หน่วย):**
`{format_dead_output(get_dead_numbers(top_probs, 7))}`
""")

        if all(x in store_final_probs for x in ["bot_ten", "bot_unit"]):
            bot_probs = (store_final_probs["bot_ten"] + store_final_probs["bot_unit"]) / 2.0
            bot_probs /= (bot_probs.sum() + 1e-12)
            st.markdown(f"""
🚫 **ดับล่างรวม (สิบ-หน่วย):**
`{format_dead_output(get_dead_numbers(bot_probs, 7))}`
""")

        st.divider()
        st.success("""
🔐 **PRO V4.2 Upgraded Dead Number Logic**
ระบบได้เปลี่ยนตรรกะใหม่เพื่อเน้นค้นหา 'ความน่าจะเป็นที่ต่ำที่สุด' 
พร้อมเพิ่มฟีเจอร์เวลาแบบวงกลม เพื่อให้ AI ตรวจจับรอบของการออกหวยได้แม่นยำยิ่งขึ้น
""")
