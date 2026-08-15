# ============================================================
# 🚀 LOTTO AI ULTIMATE ENSEMBLE (TUNED VERSION)
# SEQUENTIAL DRAW-TO-DRAW EDITION + TIME DECAY
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import copy

# Machine Learning
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 0. STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="ระบบวิเคราะห์เลขเด่น Ultimate Ensemble",
    page_icon="🚀",
    layout="centered"
)

# ============================================================
# 1. LOTTERY SOURCES
# ============================================================
LOTTERY_SOURCES = {
    "1. หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "2. หวยธกส.": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "3. หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "4. หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "5. หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "6. หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "7. หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "8. หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "9. หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "10. หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
}

# ============================================================
# 2. FETCH DATA
# ============================================================
def fetch_and_clean_data(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Mobile Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content"))
        if main_content is None:
            main_content = soup
        text_lines = main_content.get_text(separator="\n").split("\n")
        
        extracted = []
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
        num_pattern = re.compile(r"\b(\d{3})\b.*?\b(\d{2})\b|\b(\d{5,6})\b.*?\b(\d{2})\b")
        current_date = datetime.now()

        for line in text_lines:
            line = line.strip()
            if not line: continue
            
            date_match = date_pattern.search(line)
            if date_match:
                try:
                    current_date = pd.to_datetime(date_match.group(1), errors="coerce")
                except Exception: pass

            num_match = num_pattern.search(line)
            if not num_match: continue

            if num_match.group(1) and num_match.group(2):
                res3d, res2d = num_match.group(1), num_match.group(2)
            elif num_match.group(3) and num_match.group(4):
                res3d, res2d = num_match.group(3)[-3:], num_match.group(4)
            else: continue

            extracted.append({
                "Date": current_date,
                "Result_3D": str(res3d).zfill(3),
                "Result_2D": str(res2d).zfill(2)
            })

        if len(extracted) < 10: raise ValueError("ข้อมูลที่ดึงมาไม่เพียงพอ")

        df = pd.DataFrame(extracted)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Result_3D", "Result_2D"]).drop_duplicates(subset=["Date", "Result_3D", "Result_2D"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูลได้: {e}")
        return pd.DataFrame()

# ============================================================
# 3. FEATURE ENGINEERING (TUNED)
# ============================================================
def build_features(df, lags, rolls):
    df_feat = df.copy()

    # Digit extraction
    df_feat["H"] = df_feat["Result_3D"].astype(str).str[0].astype(int)
    df_feat["T"] = df_feat["Result_3D"].astype(str).str[1].astype(int)
    df_feat["O"] = df_feat["Result_3D"].astype(str).str[2].astype(int)
    df_feat["T2"] = df_feat["Result_2D"].astype(str).str[0].astype(int)
    df_feat["O2"] = df_feat["Result_2D"].astype(str).str[1].astype(int)

    # Calendar
    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["Day"] = df_feat["Date"].dt.day
    df_feat["Gap"] = df_feat["Date"].diff().dt.days.fillna(7).clip(lower=0).astype(int)
    df_feat["DOW_SIN"] = np.sin(2 * np.pi * df_feat["DayOfWeek"] / 7)
    df_feat["DOW_COS"] = np.cos(2 * np.pi * df_feat["DayOfWeek"] / 7)
    df_feat["MONTH_SIN"] = np.sin(2 * np.pi * df_feat["Month"] / 12)
    df_feat["MONTH_COS"] = np.cos(2 * np.pi * df_feat["Month"] / 12)

    # Cross-Digit Features (Distance)
    df_feat["Dist_HT"] = (df_feat["H"].shift(1) - df_feat["T"].shift(1)).abs().fillna(0)
    df_feat["Dist_TO"] = (df_feat["T"].shift(1) - df_feat["O"].shift(1)).abs().fillna(0)

    positions = ["H", "T", "O", "T2", "O2"]
    prime_digits = [2, 3, 5, 7]

    for pos in positions:
        prev = df_feat[pos].shift(1)
        
        # Advanced States
        df_feat[f"OddEven_{pos}"] = (prev % 2).fillna(0)
        df_feat[f"HighLow_{pos}"] = (prev >= 5).fillna(0).astype(int)
        df_feat[f"IsPrime_{pos}"] = prev.isin(prime_digits).astype(int) # NEW
        df_feat[f"Mirror_{pos}"] = ((prev + 5) % 10).fillna(0)

        # Lags & Rolling
        for lag in lags: df_feat[f"Lag_{lag}_{pos}"] = df_feat[pos].shift(lag)
        for w in rolls:
            shifted = df_feat[pos].shift(1)
            df_feat[f"Roll_{w}_Mean_{pos}"] = shifted.rolling(w).mean()
            df_feat[f"Roll_{w}_Std_{pos}"] = shifted.rolling(w).std()

        # Repeat Flag
        if f"Lag_1_{pos}" in df_feat.columns and f"Lag_2_{pos}" in df_feat.columns:
            df_feat[f"Repeat_{pos}"] = (df_feat[f"Lag_1_{pos}"] == df_feat[f"Lag_2_{pos}"]).astype(int)
        else:
            df_feat[f"Repeat_{pos}"] = 0

        # Hot/Cold 20
        shifted = df_feat[pos].shift(1)
        for d in range(10): df_feat[f"Hot20_{pos}_{d}"] = shifted.eq(d).rolling(20).sum()

        # Skip logic
        skips = np.zeros(len(df_feat))
        last_seen = {}
        values = df_feat[pos].values
        for i in range(len(values)):
            if i == 0: skips[i] = 100
            else:
                prev_value = values[i - 1]
                skips[i] = (i - last_seen[prev_value]) if prev_value in last_seen else i
            last_seen[values[i]] = i
        df_feat[f"Skip_{pos}"] = skips

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(-1)
    return df_feat

# ============================================================
# 4. POSITIONAL EQUATION
# ============================================================
class PositionalEquation:
    def analyze(self, df):
        latest = df.iloc[-1]
        H, T, O = latest["H"], latest["T"], latest["O"]
        probs = np.zeros(10)
        # เพิ่มสูตรคำนวณกำลังเลข
        values = [
            (H + T) % 10, (T + O) % 10, abs(H - O) % 10, (H * T) % 10,
            (H + T + O) % 10, (H * 2 + O) % 10
        ]
        for v in values: probs[int(v)] += 1
        probs += 0.1
        return probs / probs.sum()

# ============================================================
# 5. FREQUENCY
# ============================================================
class FrequencyEngine:
    def analyze(self, df, pos):
        series = df[pos].dropna()
        probs = np.zeros(10)
        if len(series) == 0: return np.ones(10) / 10
        
        freq_all = series.value_counts(normalize=True).to_dict()
        freq_15 = series.tail(15).value_counts(normalize=True).to_dict() # ขยายช่วงใกล้

        for i in range(10):
            idxs = np.where(series.values == i)[0]
            skip = (len(series) - 1 - idxs[-1]) if len(idxs) > 0 else len(series)
            
            # ลดน้ำหนักความถี่รวม เพิ่มน้ำหนักระยะใกล้ และโอกาสออกเมื่อทิ้งช่วงนาน
            probs[i] = (freq_all.get(i, 0) * 0.3) + (freq_15.get(i, 0) * 0.5) + ((1.0 / (skip + 1)) * 0.2)
        
        probs += 0.01
        return probs / probs.sum()

# ============================================================
# 6. CONDITIONAL / CALENDAR
# ============================================================
class ConditionalSystem:
    def analyze(self, df, pos, next_date):
        probs = np.zeros(10)
        subset = df[df["DayOfWeek"] == next_date.dayofweek]
        if len(subset) < 3: subset = df
        freq = subset[pos].value_counts(normalize=True).to_dict()
        for i in range(10): probs[i] = freq.get(i, 0)
        probs += 0.01
        return probs / probs.sum()

# ============================================================
# 7. STATE TRANSITION
# ============================================================
class StateTransitionSystem:
    def analyze(self, df, pos):
        probs = np.zeros(10)
        if len(df) < 3: return np.ones(10) / 10
        last_value = df[pos].iloc[-1]
        subset = df[df[f"Lag_1_{pos}"] == last_value]
        if len(subset) > 0:
            freq = subset[pos].value_counts(normalize=True).to_dict()
            for i in range(10): probs[i] = freq.get(i, 0)
        probs += 0.01
        return probs / probs.sum()

# ============================================================
# 8. PATTERN BACKTEST
# ============================================================
class PatternBacktestSystem:
    def analyze(self, df, pos):
        probs = np.zeros(10)
        if len(df) < 4: return np.ones(10) / 10
        l1, l2 = df[pos].iloc[-1], df[pos].iloc[-2]
        subset = df[(df[f"Lag_1_{pos}"] == l1) & (df[f"Lag_2_{pos}"] == l2)]
        if len(subset) == 0: subset = df[df[f"Lag_1_{pos}"] == l1]
        if len(subset) > 0:
            freq = subset[pos].value_counts(normalize=True).to_dict()
            for i in range(10): probs[i] = freq.get(i, 0)
        probs += 0.01
        return probs / probs.sum()

# ============================================================
# 9. AI SYSTEM (TUNED PARAMS)
# ============================================================
class AISystem:
    def __init__(self, trees, rf_w, et_w, hgb_w, xgb_w):
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=trees, max_depth=6, min_samples_leaf=2, class_weight='balanced', n_jobs=1, random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=trees, max_depth=6, min_samples_leaf=2, class_weight='balanced', n_jobs=1, random_state=42)),
            ("hgb", HistGradientBoostingClassifier(max_iter=50, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.1, random_state=42))
        ]
        weights = [rf_w, et_w, hgb_w]
        
        if xgb_w > 0:
            estimators.append(
                ("xgb", XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss", verbosity=0, random_state=42, n_jobs=1))
            )
            weights.append(xgb_w)

        self.voting = VotingClassifier(estimators=estimators, voting="soft", weights=weights)

    def analyze(self, X_train, y_train, X_next):
        model = copy.deepcopy(self.voting)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_next)[0]
        result = np.zeros(10)
        for c, p in zip(model.classes_, probs): result[int(c)] = p
        total = result.sum()
        if total <= 0: return np.ones(10) / 10
        return result / total

# ============================================================
# 10. ENSEMBLE ENGINE (WITH TIME DECAY BACKTEST)
# ============================================================
class EnsembleEngine:
    def __init__(self, df_raw, lottery_name, target_dow=None):
        self.df_raw = df_raw.copy()
        self.target_dow = target_dow
        self.lottery_name = lottery_name
        n = len(df_raw)

        if n >= 700:
            self.mode_name = "Mode 4 (700+ งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop = 100, 30, 13
            self.lags, self.rolls = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20]
            self.ai_weights = (1.0, 1.0, 1.0, 1.0)
        elif n >= 400:
            self.mode_name = "Mode 3 (400-699 งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop = 100, 25, 13
            self.lags, self.rolls = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20]
            self.ai_weights = (1.0, 0.9, 0.8, 1.0)
        elif n >= 200:
            self.mode_name = "Mode 2 (200-399 งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop = 80, 20, 10
            self.lags, self.rolls = [1, 2, 3, 5, 8], [3, 5, 10, 20]
            self.ai_weights = (1.0, 0.8, 0.6, 0.5)
        else:
            self.mode_name = "Mode 1 (100-199 งวด) - Super Fast"
            self.trees, self.test_size, self.early_stop = 60, 15, 8
            self.lags, self.rolls = [1, 2, 3, 5], [3, 5, 10]
            self.ai_weights = (1.0, 0.8, 0.5, 0.1)

        if n < 100: self.test_size = min(5, max(0, n - 30))

        # Build Feature List
        self.features = ["DayOfWeek", "Month", "Day", "Gap", "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS", "Dist_HT", "Dist_TO"]
        for pos in ["H", "T", "O", "T2", "O2"]:
            self.features.extend([f"OddEven_{pos}", f"HighLow_{pos}", f"IsPrime_{pos}", f"Mirror_{pos}", f"Skip_{pos}", f"Repeat_{pos}"])
            for lag in self.lags: self.features.append(f"Lag_{lag}_{pos}")
            for w in self.rolls: self.features.extend([f"Roll_{w}_Mean_{pos}", f"Roll_{w}_Std_{pos}"])
            for d in range(10): self.features.append(f"Hot20_{pos}_{d}")

        self.pos_sys, self.freq_sys, self.cond_sys = PositionalEquation(), FrequencyEngine(), ConditionalSystem()
        self.st_sys, self.ptn_sys = StateTransitionSystem(), PatternBacktestSystem()
        self.ai_sys = AISystem(self.trees, *self.ai_weights)
        self.base_weights = {"AI": 0.35, "Freq": 0.20, "ST": 0.15, "Cal": 0.10, "BT": 0.10, "Eq": 0.10}

    def _process_single_position(self, pos, df_hist, X_all, next_x, next_date):
        bt_size = self.test_size
        
        if len(df_hist) < bt_size + 30 or bt_size <= 0:
            norm_weights = self.base_weights.copy()
            bt_msg = "(ข้อมูลน้อย ข้าม Backtest)"
        else:
            ai_hits, fq_hits, cal_hits, st_hits, ptn_hits = 0, 0, 0, 0, 0
            steps_run = 0
            total_decay = 0

            # Walk Forward with TIME DECAY
            for i in range(bt_size):
                curr_train_len = len(X_all) - bt_size + i
                if curr_train_len < 30: continue
                
                # Decay Weight ยิ่งใกล้ปัจจุบันค่า i จะสูง -> ให้น้ำหนักช่วงใกล้มากกว่า
                decay_weight = 1.05 ** i
                total_decay += decay_weight

                X_train_step = X_all.iloc[:curr_train_len]
                y_train_step = df_hist[pos].iloc[:curr_train_len]
                X_test_step = X_all.iloc[[curr_train_len]]
                actual_val = df_hist[pos].iloc[curr_train_len]

                # Proxy AI (Stricter Hit: Top 4 instead of 5)
                proxy_model = ExtraTreesClassifier(n_estimators=10, max_depth=4, min_samples_leaf=2, random_state=42)
                proxy_model.fit(X_train_step, y_train_step)
                probs = proxy_model.predict_proba(X_test_step)[0]
                ai_res = np.zeros(10)
                for idx, c in enumerate(proxy_model.classes_): ai_res[int(c)] = probs[idx]
                if actual_val in np.argsort(ai_res)[::-1][:4]: ai_hits += decay_weight

                curr_df = df_hist.iloc[:curr_train_len]
                target_date = df_hist["Date"].iloc[curr_train_len]

                if actual_val in np.argsort(self.freq_sys.analyze(curr_df, pos))[::-1][:4]: fq_hits += decay_weight
                if actual_val in np.argsort(self.cond_sys.analyze(curr_df, pos, target_date))[::-1][:4]: cal_hits += decay_weight
                if actual_val in np.argsort(self.st_sys.analyze(curr_df, pos))[::-1][:4]: st_hits += decay_weight
                if actual_val in np.argsort(self.ptn_sys.analyze(curr_df, pos))[::-1][:4]: ptn_hits += decay_weight

                steps_run += 1
                if steps_run >= self.early_stop: break

            if steps_run <= 0 or total_decay == 0:
                norm_weights = self.base_weights.copy()
                bt_msg = "(Backtest ไม่สามารถทำงานได้)"
            else:
                w_ai = self.base_weights["AI"] * max(0.1, ai_hits / total_decay) ** 2
                w_fq = self.base_weights["Freq"] * max(0.1, fq_hits / total_decay) ** 2
                w_cal = self.base_weights["Cal"] * max(0.1, cal_hits / total_decay) ** 2
                w_st = self.base_weights["ST"] * max(0.1, st_hits / total_decay) ** 2
                w_bt = self.base_weights["BT"] * max(0.1, ptn_hits / total_decay) ** 2
                w_eq = self.base_weights["Eq"] * 0.1
                
                total = w_ai + w_fq + w_cal + w_st + w_bt + w_eq
                norm_weights = {"AI": w_ai / total, "Freq": w_fq / total, "Cal": w_cal / total, "ST": w_st / total, "BT": w_bt / total, "Eq": w_eq / total}
                
                bt_msg = (f"(Backtest {steps_run} งวด (Time-Decay Weighted): "
                          f"AI {int(ai_hits/total_decay*100)}% | Freq {int(fq_hits/total_decay*100)}% | "
                          f"ST {int(st_hits/total_decay*100)}% | Pattern {int(ptn_hits/total_decay*100)}%)")

        # CURRENT PREDICTION
        p_ai = self.ai_sys.analyze(X_all, df_hist[pos], next_x)
        p_fq = self.freq_sys.analyze(df_hist, pos)
        p_cal = self.cond_sys.analyze(df_hist, pos, next_date)
        p_st = self.st_sys.analyze(df_hist, pos)
        p_bt = self.ptn_sys.analyze(df_hist, pos)
        p_eq = self.pos_sys.analyze(df_hist)

        W = norm_weights
        final_score = (W["AI"] * p_ai + W["Freq"] * p_fq + W["Cal"] * p_cal + W["ST"] * p_st + W["BT"] * p_bt + W["Eq"] * p_eq)
        
        total = final_score.sum()
        final_score = (np.ones(10)/10) if total <= 0 else (final_score / total)

        def get_top5(probs):
            return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]

        return pos, {"AI": get_top5(p_ai), "Calendar": get_top5(p_cal), "Frequency": get_top5(p_fq), "Final": get_top5(final_score), "Probs_For_Graph": final_score, "Weights": norm_weights, "BT_Msg": bt_msg}

    def predict_all(self):
        last_date = self.df_raw["Date"].iloc[-1]
        
        if self.target_dow is not None:
            days_ahead = self.target_dow - last_date.dayofweek
            if days_ahead <= 0: days_ahead += 7
            next_date = last_date + timedelta(days=days_ahead)
        else:
            gap_days = 7 if len(self.df_raw) <= 1 else max(1, (self.df_raw["Date"].iloc[-1] - self.df_raw["Date"].iloc[-2]).days)
            next_date = last_date + timedelta(days=gap_days)

        dummy = pd.DataFrame([{"Date": next_date, "Result_3D": "000", "Result_2D": "00"}])
        df_ext = pd.concat([self.df_raw, dummy], ignore_index=True)
        df_ext = build_features(df_ext, self.lags, self.rolls)

        df_hist = df_ext.iloc[:-1].copy()
        X_all = df_hist[self.features].copy()
        next_x = df_ext.iloc[[-1]][self.features].copy()

        results = []
        for pos in ["H", "T", "O", "T2", "O2"]:
            results.append(self._process_single_position(pos, df_hist, X_all, next_x, next_date))

        return {pos: data for pos, data in results}, next_date

# ============================================================
# 11. STREAMLIT UI
# ============================================================
st.title("🚀 ระบบวิเคราะห์เลขเด่น Ultimate Ensemble (Tuned)")
st.markdown("**Sequential Draw-to-Draw Edition (Time-Decay Weighted)**")
st.caption("🔄 ปรับปรุงใหม่: เพิ่ม Features จำแนกจำนวนเฉพาะ/ระยะห่าง และ Backtest ถ่วงน้ำหนักใกล้งวดล่าสุด")
st.divider()

col1, col2 = st.columns(2)
with col1: selected_lotto = st.selectbox("🎯 เลือกหวย:", list(LOTTERY_SOURCES.keys()), index=0, key="den_1")
with col2:
    day_options = {"อัตโนมัติ (คำนวณจากงวดล่าสุด)": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6}
    selected_day_label = st.selectbox("📅 ออกวัน:", list(day_options.keys()), index=0, key="den_2")
    target_dow = day_options[selected_day_label]

if st.button("🚀 วิเคราะห์เลขเด่น V.Max Sequential", type="primary", use_container_width=True):
    with st.spinner("⏳ กำลังดึงข้อมูลและคำนวณใหม่ทั้งหมด... (AI กำลังทำ Time-Decay Backtest)"):
        url = LOTTERY_SOURCES[selected_lotto]
        df_raw = fetch_and_clean_data(url)
        if df_raw.empty: st.stop()

        engine = EnsembleEngine(df_raw, selected_lotto, target_dow=target_dow)

        st.info(f"""
**⚙️ สเตตัสระบบ [{engine.mode_name}]**
- 📚 ข้อมูลทั้งหมด: **{len(df_raw)} งวด**
- ✨ **New:** Advanced Features (IsPrime, Digit Distance)
- ⚖️ **New:** Time-Decay Backtest (คัดกรองโมเดลแม่นยำช่วง 10 งวดหลังได้ดีที่สุด)
""")

        preds, next_date = engine.predict_all()
        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
        labels = {"H": "หลักร้อย (บน)", "T": "หลักสิบ (บน)", "O": "หลักหน่วย (บน)", "T2": "หลักสิบ (ล่าง)", "O2": "หลักหน่วย (ล่าง)"}

        st.markdown(f"### 🔮 ผลการวิเคราะห์ ประจำวัน{dow_names[next_date.dayofweek]} ที่ {next_date.strftime('%d-%m-%Y')}")
        st.divider()

        for pos in ["H", "T", "O", "T2", "O2"]:
            nums_ai = ", ".join([str(num) for num, prob in preds[pos]["AI"]])
            nums_day = ", ".join([str(num) for num, prob in preds[pos]["Calendar"]])
            nums_stat = ", ".join([str(num) for num, prob in preds[pos]["Frequency"]])
            nums_final = ", ".join([str(num) for num, prob in preds[pos]["Final"]])

            with st.expander(f"📍 ตำแหน่ง: {labels[pos]}", expanded=True):
                st.caption(preds[pos]["BT_Msg"])
                st.markdown(f"- 🤖 **เลขเด่น AI** : `{nums_ai}`")
                st.markdown(f"- 📅 **เลขเด่น กำลังวัน** : `{nums_day}`")
                st.markdown(f"- 📊 **เลขเด่น สถิติ** : `{nums_stat}`")
                st.success(f"🌟 **เด่นสรุปรวม 5 ตัว**: `{nums_final}`")
                W = preds[pos]["Weights"]
                st.caption(f"⚖️ น้ำหนัก Ensemble: AI={W['AI']:.1%} | Freq={W['Freq']:.1%} | Cal={W['Cal']:.1%} | ST={W['ST']:.1%} | BT={W['BT']:.1%} | Eq={W['Eq']:.1%}")

        probs_top = (preds["H"]["Probs_For_Graph"] + preds["T"]["Probs_For_Graph"] + preds["O"]["Probs_For_Graph"]) / 3
        probs_bot = (preds["T2"]["Probs_For_Graph"] + preds["O2"]["Probs_For_Graph"]) / 2

        def get_top5(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]
        
        st.divider()
        st.subheader("🔥 สรุปฟันธง เลขเด่นมาแรง")
        st.markdown("🚀 **เด่นบนรวม (ร้อย-สิบ-หน่วย)** : `" + " , ".join([str(x[0]) for x in get_top5(probs_top)]) + "`")
        st.markdown("⬇️ **เด่นล่างรวม (สิบ-หน่วย)** : `" + " , ".join([str(x[0]) for x in get_top5(probs_bot)]) + "`")

        st.divider()
        st.subheader("📊 กราฟความน่าจะเป็นแต่ละหลัก")
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for idx, pos in enumerate(["H", "T", "O", "T2", "O2"]):
            ax = axes[idx]
            top_5_items = preds[pos]["Final"]
            numbers = [str(x[0]) for x in top_5_items]
            probabilities = [x[1] * 100 for x in top_5_items]
            ax.bar(numbers, probabilities)
            ax.set_title(labels[pos])
            ax.set_ylabel("โอกาส (%)")
            ax.set_ylim(0, max(probabilities) * 1.25 if probabilities else 1)

        fig.delaxes(axes[5])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.caption("⚠️ ระบบเป็นการวิเคราะห์เชิงสถิติและ Machine Learning ไม่สามารถรับประกันผลรางวัลได้")
