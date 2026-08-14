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
# 0. STREAMLIT CONFIG (ต้องอยู่บนสุด)
# ============================================================
st.set_page_config(
    page_title="AI วิเคราะห์หวย ครบวงจร",
    page_icon="🎯",
    layout="centered"
)

# ============================================================
# 1. LOTTERY SOURCES (ใช้ร่วมกัน)
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
# ================= CORE LOGIC: เลขเด่น ======================
# ============================================================
def fetch_and_clean_data_den(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
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
                raw_date = date_match.group(1)
                try: current_date = pd.to_datetime(raw_date, errors="coerce")
                except Exception: pass
            
            num_match = num_pattern.search(line)
            if not num_match: continue
            if num_match.group(1) and num_match.group(2):
                res3d = num_match.group(1)
                res2d = num_match.group(2)
            elif num_match.group(3) and num_match.group(4):
                res3d = num_match.group(3)[-3:]
                res2d = num_match.group(4)
            else: continue

            extracted.append({"Date": current_date, "Result_3D": str(res3d).zfill(3), "Result_2D": str(res2d).zfill(2)})

        if len(extracted) < 10: raise ValueError("ข้อมูลที่ดึงมาไม่เพียงพอ")
        df = pd.DataFrame(extracted)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Result_3D"] = df["Result_3D"].astype(str).str.extract(r"(\d{3})")[0]
        df["Result_2D"] = df["Result_2D"].astype(str).str.extract(r"(\d{2})")[0]
        df = df.dropna(subset=["Result_3D", "Result_2D"])
        df = df.drop_duplicates(subset=["Date", "Result_3D", "Result_2D"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูลได้: {e}")
        return pd.DataFrame()

def build_features_den(df, lags, rolls):
    df_feat = df.copy()
    df_feat["H"] = df_feat["Result_3D"].astype(str).str[0].astype(int)
    df_feat["T"] = df_feat["Result_3D"].astype(str).str[1].astype(int)
    df_feat["O"] = df_feat["Result_3D"].astype(str).str[2].astype(int)
    df_feat["T2"] = df_feat["Result_2D"].astype(str).str[0].astype(int)
    df_feat["O2"] = df_feat["Result_2D"].astype(str).str[1].astype(int)
    
    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["Day"] = df_feat["Date"].dt.day
    df_feat["WeekOfYear"] = df_feat["Date"].dt.isocalendar().week.astype(int)
    df_feat["DayOfYear"] = df_feat["Date"].dt.dayofyear
    df_feat["DrawIndex"] = np.arange(len(df_feat))
    df_feat["Gap"] = df_feat["Date"].diff().dt.days.fillna(7).clip(lower=0).astype(int)
    df_feat["DOW_SIN"] = np.sin(2 * np.pi * df_feat["DayOfWeek"] / 7)
    df_feat["DOW_COS"] = np.cos(2 * np.pi * df_feat["DayOfWeek"] / 7)
    df_feat["MONTH_SIN"] = np.sin(2 * np.pi * df_feat["Month"] / 12)
    df_feat["MONTH_COS"] = np.cos(2 * np.pi * df_feat["Month"] / 12)
    df_feat["DigitSum_3D"] = (df_feat["H"].shift(1) + df_feat["T"].shift(1) + df_feat["O"].shift(1)).fillna(0) % 10

    positions = ["H", "T", "O", "T2", "O2"]
    for pos in positions:
        prev = df_feat[pos].shift(1)
        df_feat[f"OddEven_{pos}"] = (prev % 2).fillna(0)
        df_feat[f"HighLow_{pos}"] = (prev >= 5).fillna(0).astype(int)
        df_feat[f"Mirror_{pos}"] = ((prev + 5) % 10).fillna(0)
        df_feat[f"Mod3_{pos}"] = (prev % 3).fillna(0)

        for lag in lags: df_feat[f"Lag_{lag}_{pos}"] = df_feat[pos].shift(lag)
        for w in rolls:
            shifted = df_feat[pos].shift(1)
            df_feat[f"Roll_{w}_Mean_{pos}"] = shifted.rolling(w).mean()
            df_feat[f"Roll_{w}_Std_{pos}"] = shifted.rolling(w).std()

        if f"Lag_1_{pos}" in df_feat.columns and f"Lag_2_{pos}" in df_feat.columns:
            df_feat[f"Repeat_{pos}"] = (df_feat[f"Lag_1_{pos}"] == df_feat[f"Lag_2_{pos}"]).astype(int)
        else:
            df_feat[f"Repeat_{pos}"] = 0

        shifted = df_feat[pos].shift(1)
        for d in range(10): df_feat[f"Hot20_{pos}_{d}"] = shifted.eq(d).rolling(20).sum()

        skips = np.zeros(len(df_feat))
        last_seen = {}
        values = df_feat[pos].values
        for i in range(len(values)):
            if i == 0: skips[i] = 100
            else:
                prev_value = values[i - 1]
                if prev_value in last_seen: skips[i] = i - last_seen[prev_value]
                else: skips[i] = i
            last_seen[values[i]] = i
        df_feat[f"Skip_{pos}"] = skips

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    return df_feat.fillna(-1)

class PositionalEquation:
    def analyze(self, df):
        latest = df.iloc[-1]
        H, T, O = latest["H"], latest["T"], latest["O"]
        probs = np.zeros(10)
        values = [(H + T) % 10, (T + O) % 10, abs(H - O) % 10, (H * T) % 10]
        for v in values: probs[int(v)] += 1
        probs += 0.1
        return probs / probs.sum()

class FrequencyEngine:
    def analyze(self, df, pos):
        series = df[pos].dropna()
        probs = np.zeros(10)
        if len(series) == 0: return np.ones(10) / 10
        freq_all = series.value_counts(normalize=True).to_dict()
        freq_10 = series.tail(10).value_counts(normalize=True).to_dict()
        for i in range(10):
            idxs = np.where(series.values == i)[0]
            skip = (len(series) - 1 - idxs[-1]) if len(idxs) > 0 else len(series)
            probs[i] = freq_all.get(i, 0)*0.4 + freq_10.get(i, 0)*0.4 + (1.0 / (skip + 1))*0.2
        probs += 0.01
        return probs / probs.sum()

class ConditionalSystem:
    def analyze(self, df, pos, next_date):
        probs = np.zeros(10)
        subset = df[df["DayOfWeek"] == next_date.dayofweek]
        if len(subset) < 3: subset = df
        freq = subset[pos].value_counts(normalize=True).to_dict()
        for i in range(10): probs[i] = freq.get(i, 0)
        probs += 0.01
        return probs / probs.sum()

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

class AISystem:
    def __init__(self, trees, rf_w, et_w, hgb_w, xgb_w):
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=trees, max_depth=5, min_samples_leaf=2, n_jobs=1, random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=trees, max_depth=5, min_samples_leaf=2, n_jobs=1, random_state=42)),
            ("hgb", HistGradientBoostingClassifier(max_iter=50, learning_rate=0.05, max_leaf_nodes=15, random_state=42))
        ]
        weights = [rf_w, et_w, hgb_w]
        if xgb_w > 0:
            estimators.append(("xgb", XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss", verbosity=0, random_state=42, n_jobs=1)))
            weights.append(xgb_w)
        self.voting = VotingClassifier(estimators=estimators, voting="soft", weights=weights)

    def analyze(self, X_train, y_train, X_next):
        model = copy.deepcopy(self.voting)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_next)[0]
        result = np.zeros(10)
        for c, p in zip(model.classes_, probs): result[int(c)] = p
        total = result.sum()
        return np.ones(10) / 10 if total <= 0 else result / total

class EnsembleEngine:
    def __init__(self, df_raw, lottery_name, target_dow=None):
        self.df_raw = df_raw.copy()
        self.target_dow = target_dow
        self.lottery_name = lottery_name
        n = len(df_raw)

        if n >= 700:
            self.mode_name, self.trees, self.test_size, self.early_stop = "Mode 4 (700+ งวด) - Super Fast", 100, 30, 13
            self.lags, self.rolls, self.ai_weights = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20], (1.0, 1.0, 1.0, 1.0)
        elif n >= 400:
            self.mode_name, self.trees, self.test_size, self.early_stop = "Mode 3 (400-699 งวด) - Super Fast", 100, 25, 13
            self.lags, self.rolls, self.ai_weights = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20], (1.0, 0.9, 0.8, 1.0)
        elif n >= 200:
            self.mode_name, self.trees, self.test_size, self.early_stop = "Mode 2 (200-399 งวด) - Super Fast", 80, 20, 10
            self.lags, self.rolls, self.ai_weights = [1, 2, 3, 5, 8], [3, 5, 10, 20], (1.0, 0.8, 0.6, 0.5)
        else:
            self.mode_name, self.trees, self.test_size, self.early_stop = "Mode 1 (100-199 งวด) - Super Fast", 60, 15, 8
            self.lags, self.rolls, self.ai_weights = [1, 2, 3, 5], [3, 5, 10], (1.0, 0.8, 0.5, 0.1)

        if n < 100: self.test_size = min(5, max(0, n - 30))

        self.features = ["DayOfWeek", "Month", "Day", "WeekOfYear", "DayOfYear", "DrawIndex", "Gap", "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS", "DigitSum_3D"]
        for pos in ["H", "T", "O", "T2", "O2"]:
            self.features.extend([f"OddEven_{pos}", f"HighLow_{pos}", f"Mirror_{pos}", f"Mod3_{pos}", f"Skip_{pos}", f"Repeat_{pos}"])
            for lag in self.lags: self.features.append(f"Lag_{lag}_{pos}")
            for w in self.rolls: self.features.extend([f"Roll_{w}_Mean_{pos}", f"Roll_{w}_Std_{pos}"])
            for d in range(10): self.features.append(f"Hot20_{pos}_{d}")

        self.pos_sys, self.freq_sys = PositionalEquation(), FrequencyEngine()
        self.cond_sys, self.st_sys = ConditionalSystem(), StateTransitionSystem()
        self.ptn_sys = PatternBacktestSystem()
        self.ai_sys = AISystem(self.trees, *self.ai_weights)
        self.base_weights = {"AI": 0.35, "Freq": 0.20, "ST": 0.15, "Cal": 0.10, "BT": 0.10, "Eq": 0.10}

    def _process_single_position(self, pos, df_hist, X_all, next_x, next_date):
        bt_size = self.test_size
        if len(df_hist) < bt_size + 30 or bt_size <= 0:
            norm_weights, bt_msg = self.base_weights.copy(), "(ข้อมูลน้อย ข้าม Backtest)"
        else:
            ai_hits, fq_hits, cal_hits, st_hits, ptn_hits, steps_run = 0, 0, 0, 0, 0, 0
            for i in range(bt_size):
                curr_train_len = len(X_all) - bt_size + i
                if curr_train_len < 30: continue
                X_train_step, y_train_step = X_all.iloc[:curr_train_len], df_hist[pos].iloc[:curr_train_len]
                X_test_step, actual_val = X_all.iloc[[curr_train_len]], df_hist[pos].iloc[curr_train_len]

                proxy_model = ExtraTreesClassifier(n_estimators=10, max_depth=3, min_samples_leaf=2, n_jobs=1, random_state=42)
                proxy_model.fit(X_train_step, y_train_step)
                probs = proxy_model.predict_proba(X_test_step)[0]
                ai_res = np.zeros(10)
                for idx, c in enumerate(proxy_model.classes_): ai_res[int(c)] = probs[idx]
                if actual_val in np.argsort(ai_res)[::-1][:5]: ai_hits += 1

                curr_df, target_date = df_hist.iloc[:curr_train_len], df_hist["Date"].iloc[curr_train_len]
                if actual_val in np.argsort(self.freq_sys.analyze(curr_df, pos))[::-1][:5]: fq_hits += 1
                if actual_val in np.argsort(self.cond_sys.analyze(curr_df, pos, target_date))[::-1][:5]: cal_hits += 1
                if actual_val in np.argsort(self.st_sys.analyze(curr_df, pos))[::-1][:5]: st_hits += 1
                if actual_val in np.argsort(self.ptn_sys.analyze(curr_df, pos))[::-1][:5]: ptn_hits += 1

                steps_run += 1
                if steps_run >= self.early_stop: break

            if steps_run <= 0:
                norm_weights, bt_msg = self.base_weights.copy(), "(Backtest ไม่สามารถทำงานได้)"
            else:
                w_ai = self.base_weights["AI"] * max(0.1, ai_hits / steps_run) ** 2
                w_fq = self.base_weights["Freq"] * max(0.1, fq_hits / steps_run) ** 2
                w_cal = self.base_weights["Cal"] * max(0.1, cal_hits / steps_run) ** 2
                w_st = self.base_weights["ST"] * max(0.1, st_hits / steps_run) ** 2
                w_bt = self.base_weights["BT"] * max(0.1, ptn_hits / steps_run) ** 2
                w_eq = self.base_weights["Eq"] * 0.1
                total = w_ai + w_fq + w_cal + w_st + w_bt + w_eq
                norm_weights = {"AI": w_ai / total, "Freq": w_fq / total, "Cal": w_cal / total, "ST": w_st / total, "BT": w_bt / total, "Eq": w_eq / total}
                bt_msg = f"(Backtest {steps_run} งวด: AI {int(ai_hits / steps_run * 100)}% | Freq {int(fq_hits / steps_run * 100)}% | ST {int(st_hits / steps_run * 100)}% | Cal {int(cal_hits / steps_run * 100)}% | Pattern {int(ptn_hits / steps_run * 100)}%)"

        p_ai = self.ai_sys.analyze(X_all, df_hist[pos], next_x)
        p_fq = self.freq_sys.analyze(df_hist, pos)
        p_cal = self.cond_sys.analyze(df_hist, pos, next_date)
        p_st = self.st_sys.analyze(df_hist, pos)
        p_bt = self.ptn_sys.analyze(df_hist, pos)
        p_eq = self.pos_sys.analyze(df_hist)

        W = norm_weights
        final_score = W["AI"]*p_ai + W["Freq"]*p_fq + W["Cal"]*p_cal + W["ST"]*p_st + W["BT"]*p_bt + W["Eq"]*p_eq
        total = final_score.sum()
        final_score = np.ones(10) / 10 if total <= 0 else final_score / total

        def get_top5(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]
        return pos, {"AI": get_top5(p_ai), "Calendar": get_top5(p_cal), "Frequency": get_top5(p_fq), "Final": get_top5(final_score), "Probs_For_Graph": final_score, "Weights": norm_weights, "BT_Msg": bt_msg}

    def predict_all(self):
        last_date = self.df_raw["Date"].iloc[-1]
        if self.target_dow is not None:
            days_ahead = self.target_dow - last_date.dayofweek
            if days_ahead <= 0: days_ahead += 7
            next_date = last_date + timedelta(days=days_ahead)
        else:
            gap_days = 7 if len(self.df_raw) <= 1 else (self.df_raw["Date"].iloc[-1] - self.df_raw["Date"].iloc[-2]).days
            if gap_days <= 0: gap_days = 7
            next_date = last_date + timedelta(days=gap_days)

        dummy = pd.DataFrame([{"Date": next_date, "Result_3D": "000", "Result_2D": "00"}])
        df_ext = pd.concat([self.df_raw, dummy], ignore_index=True)
        df_ext = build_features_den(df_ext, self.lags, self.rolls)

        df_hist = df_ext.iloc[:-1].copy()
        X_all = df_hist[self.features].copy()
        next_x = df_ext.iloc[[-1]][self.features].copy()

        results = []
        for pos in ["H", "T", "O", "T2", "O2"]:
            results.append(self._process_single_position(pos, df_hist, X_all, next_x, next_date))
        return {pos: data for pos, data in results}, next_date


# ============================================================
# ================= CORE LOGIC: เลขดับ =======================
# ============================================================
def fetch_data_dub(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        if not response.content: return None
        soup = BeautifulSoup(response.content, "html.parser")
        post_body = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content"))
        if post_body is None: post_body = soup
        text_content = post_body.get_text(separator="\n")

        pattern = re.compile(r"\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d{2})")
        matches = pattern.findall(text_content)
        data = []
        for date_str, prize1, bot2 in matches:
            p1, p2 = str(prize1).zfill(3), str(bot2).zfill(2)
            data.append({"date": date_str, "draw_num": p1, "hundred": int(p1[0]), "ten": int(p1[1]), "unit": int(p1[2]), "bot_ten": int(p2[0]), "bot_unit": int(p2[1])})

        if len(data) < 30: return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ ดึงข้อมูลไม่สำเร็จ: {e}")
        return None

def get_adaptive_config_dub(n):
    if n >= 700: return {"mode": "Mode 4 (700+ งวด)", "trees": 100, "test_size": 30, "early_stop": 15, "lags": [1, 2, 3, 5, 8, 13], "rolls": [3, 5, 10, 20], "rf": 1.0, "et": 1.0, "hgb": 1.0, "xgb": 1.0}
    elif n >= 400: return {"mode": "Mode 3 (400-699 งวด)", "trees": 100, "test_size": 25, "early_stop": 13, "lags": [1, 2, 3, 5, 8, 13], "rolls": [3, 5, 10, 20], "rf": 1.0, "et": 0.9, "hgb": 0.8, "xgb": 1.0}
    elif n >= 200: return {"mode": "Mode 2 (200-399 งวด)", "trees": 80, "test_size": 20, "early_stop": 10, "lags": [1, 2, 3, 5, 8], "rolls": [3, 5, 10, 20], "rf": 1.0, "et": 0.8, "hgb": 0.6, "xgb": 0.5}
    else: return {"mode": "Mode 1 (30-199 งวด)", "trees": 60, "test_size": 15, "early_stop": 8, "lags": [1, 2, 3, 5], "rolls": [3, 5, 10], "rf": 1.0, "et": 0.8, "hgb": 0.5, "xgb": 0.1}

def build_features_dub(df, target_col, lags, rolls):
    df_feat = df.copy()
    n = len(df_feat)
    df_feat["prev_val"] = df_feat[target_col].shift(1)
    prev = df_feat["prev_val"]
    df_feat["mirror"] = ((prev + 5) % 10)
    df_feat["is_even"] = (prev % 2 == 0).astype(int)
    df_feat["is_high"] = (prev >= 5).astype(int)
    df_feat["mod3"] = (prev % 3)
    df_feat["weekday"] = df_feat["date"].dt.weekday
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["day"] = df_feat["date"].dt.day

    for lag in lags: df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

    if "lag_1" in df_feat.columns: df_feat["repeat_2"] = (df_feat["lag_1"] == df_feat.get("lag_2", -999)).astype(int)
    if "lag_1" in df_feat.columns and "lag_2" in df_feat.columns and "lag_3" in df_feat.columns:
        df_feat["repeat_3"] = ((df_feat["lag_1"] == df_feat["lag_2"]) & (df_feat["lag_2"] == df_feat["lag_3"])).astype(int)

    for w in rolls:
        shifted = df_feat[target_col].shift(1)
        df_feat[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        df_feat[f"rolling_std_{w}"] = shifted.rolling(w).std()

    windows = list(rolls)
    if n >= 500 and 50 not in windows: windows.append(50)
    history = df_feat[target_col].values
    for w in windows:
        for d in range(10):
            hot_values, cold_values = np.zeros(n), np.zeros(n)
            for i in range(n):
                start = max(0, i - w)
                window = history[start:i]
                count = np.sum(window == d)
                hot_values[i] = count
                if w < 50: cold_values[i] = len(window) - count
            df_feat[f"hot{w}_{d}"] = hot_values
            if w < 50: df_feat[f"cold{w}_{d}"] = cold_values

    for d in range(10):
        skip_values, last_seen = np.full(n, 100.0), -1
        for i in range(n):
            if last_seen >= 0: skip_values[i] = i - last_seen
            if history[i] == d: last_seen = i
        df_feat[f"skip_{d}"] = skip_values
    return df_feat.fillna(-1)

class OptimizedEliminationSystemV4:
    def __init__(self, df, target_col, lotto_name):
        self.df, self.target_col, self.lotto_name, self.n = df.copy(), target_col, lotto_name, len(df)
        self.cfg = get_adaptive_config_dub(self.n)
        self.mode_name, self.trees, self.test_size, self.early_stop = self.cfg["mode"], self.cfg["trees"], min(self.cfg["test_size"], max(0, self.n - 30)), self.cfg["early_stop"]
        self.lags, self.rolls = self.cfg["lags"], self.cfg["rolls"]
        self.ai_weights = (self.cfg["rf"], self.cfg["et"], self.cfg["hgb"], self.cfg["xgb"])
        self.models = self.create_models()

    def create_models(self):
        return {
            "rf": RandomForestClassifier(n_estimators=self.trees, max_depth=5, min_samples_leaf=2, random_state=42, n_jobs=1),
            "et": ExtraTreesClassifier(n_estimators=self.trees, max_depth=5, min_samples_leaf=2, random_state=42, n_jobs=1),
            "hgb": HistGradientBoostingClassifier(max_iter=50, max_depth=5, learning_rate=0.08, random_state=42),
            "xgb": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss", verbosity=0, random_state=42, n_jobs=1)
        }

    @staticmethod
    def convert_probs(model, probs):
        result = np.zeros(10)
        for idx, cls in enumerate(model.classes_):
            try:
                digit = int(cls)
                if 0 <= digit <= 9: result[digit] = probs[idx]
            except: pass
        total = result.sum()
        return np.ones(10) / 10 if total <= 0 else result / total

    def train_ai(self, X_train, y_train, X_predict):
        ai_probs, total_weight = np.zeros(10), 0.0
        for idx, (name, base_model) in enumerate(self.models.items()):
            weight = self.ai_weights[idx]
            if weight <= 0: continue
            model = type(base_model)(**base_model.get_params())
            try:
                model.fit(X_train, y_train)
                model_probs = self.convert_probs(model, model.predict_proba(X_predict)[0])
                ai_probs += (model_probs * weight)
                total_weight += weight
            except Exception: continue
        if total_weight <= 0: return np.ones(10) / 10
        ai_probs /= total_weight
        total = ai_probs.sum()
        return np.ones(10) / 10 if total <= 0 else ai_probs / total

    def markov(self, df_hist):
        seq = df_hist[self.target_col].astype(int).values
        n = len(seq)
        if n < 5: return np.ones(10) / 10
        last1 = seq[-1]

        p1, total1 = np.zeros(10), 0
        for i in range(0, n - 1):
            if seq[i] == last1: p1[seq[i + 1]] += 1; total1 += 1
        p1 = p1 / total1 if total1 > 0 else np.full(10, 0.1)

        if n < 200: return p1
        last2 = seq[-2]
        p2, total2 = np.zeros(10), 0
        for i in range(1, n - 1):
            if seq[i - 1] == last2 and seq[i] == last1: p2[seq[i + 1]] += 1; total2 += 1
        p2 = p2 / total2 if total2 > 0 else p1.copy()

        if n < 500: return 0.6 * p2 + 0.4 * p1
        last3 = seq[-3]
        p3, total3 = np.zeros(10), 0
        for i in range(2, n - 1):
            if seq[i - 2] == last3 and seq[i - 1] == last2 and seq[i] == last1: p3[seq[i + 1]] += 1; total3 += 1
        p3 = p3 / total3 if total3 > 0 else p2.copy()
        return 0.5 * p3 + 0.3 * p2 + 0.2 * p1

    def freq_skip(self, df_hist):
        result, series, n = np.zeros(10), df_hist[self.target_col].astype(int).values, len(df_hist)
        for d in range(10):
            count = np.sum(series == d)
            freq = count / max(n, 1)
            positions = np.where(series == d)[0]
            skip = (n - positions[-1] - 1) if len(positions) > 0 else 100
            result[d] = 0.5 * min(freq * 10, 1.0) + 0.5 * max(1.0 - skip / 30.0, 0.0)
        total = result.sum()
        return np.ones(10) / 10 if total <= 0 else result / total

    def day_probability(self, df_hist, target_dow):
        day_df = df_hist[df_hist["date"].dt.weekday == target_dow]
        if len(day_df) == 0: return np.ones(10) / 10
        counts = day_df[self.target_col].value_counts(normalize=True)
        probs = np.zeros(10)
        for d in range(10): probs[d] = counts.get(d, 0.0)
        total = probs.sum()
        return np.ones(10) / 10 if total <= 0 else probs / total

    def run_backtest(self, X_all, y_all, df_all):
        if self.test_size <= 0 or len(X_all) <= self.test_size + 30: return {"ai": 0.5, "stat": 0.5, "day": 0.5, "steps": 0}
        start, ai_hits, stat_hits, day_hits, steps = len(X_all) - self.test_size, 0, 0, 0, 0
        for i in range(start, len(X_all)):
            X_train, y_train, X_test, actual, hist = X_all.iloc[:i], y_all.iloc[:i], X_all.iloc[[i]], int(y_all.iloc[i]), df_all.iloc[:i].copy()
            if len(hist) < 30: continue
            
            ai = self.train_ai(X_train, y_train, X_test)
            if actual in np.argsort(ai)[::-1][:5]: ai_hits += 1

            stat = 0.5 * self.markov(hist) + 0.5 * self.freq_skip(hist)
            stat /= (stat.sum() + 1e-12)
            if actual in np.argsort(stat)[::-1][:5]: stat_hits += 1

            day = self.day_probability(hist, df_all.iloc[i]["date"].weekday())
            if actual in np.argsort(day)[::-1][:5]: day_hits += 1

            steps += 1
            if steps >= self.early_stop: break
        if steps <= 0: return {"ai": 0.5, "stat": 0.5, "day": 0.5, "steps": 0}
        return {"ai": ai_hits / steps, "stat": stat_hits / steps, "day": day_hits / steps, "steps": steps}

    def analyze(self, target_date, target_dow):
        if self.n < 30: return None
        dummy = {"date": target_date, "draw_num": "000", "hundred": 0, "ten": 0, "unit": 0, "bot_ten": 0, "bot_unit": 0}
        df_extended = pd.concat([self.df, pd.DataFrame([dummy])], ignore_index=True)
        df_feat = build_features_dub(df_extended, self.target_col, self.lags, self.rolls)

        X_all, X_next, y_all = df_feat.iloc[:-1].copy(), df_feat.iloc[[-1]].copy(), self.df[self.target_col].astype(int)
        feature_cols = [c for c in X_all.columns if c not in ["date", "draw_num", "hundred", "ten", "unit", "bot_ten", "bot_unit", self.target_col]]
        X_train, X_predict = X_all[feature_cols], X_next[feature_cols]

        bt = self.run_backtest(X_train, y_all, self.df)
        w_ai, w_stat, w_day = (0.30, 0.50, 0.20) if self.n < 200 else (0.40, 0.40, 0.20) if self.n < 500 else (0.50, 0.35, 0.15)
        
        if bt["steps"] > 0:
            wa, ws, wd = w_ai * max(0.10, bt["ai"])**2, w_stat * max(0.10, bt["stat"])**2, w_day * max(0.10, bt["day"])**2
            total = wa + ws + wd
            if total > 0: w_ai, w_stat, w_day = wa / total, ws / total, wd / total

        ai_probs = self.train_ai(X_train, y_all, X_predict)
        stat_probs = 0.5 * self.markov(self.df) + 0.5 * self.freq_skip(self.df)
        stat_probs /= (stat_probs.sum() + 1e-12)
        day_probs = self.day_probability(self.df, target_dow)

        final_probs = w_ai * ai_probs + w_stat * stat_probs + w_day * day_probs
        final_probs /= (final_probs.sum() + 1e-12)

        return {"ai": ai_probs, "stat": stat_probs, "day": day_probs, "final": final_probs, "w_ai": w_ai, "w_stat": w_stat, "w_day": w_day, "bt_msg": f"BT-WalkForward {bt['steps']} งวด | AI {bt['ai']*100:.1f}% | Stat {bt['stat']*100:.1f}% | Day {bt['day']*100:.1f}%"}

def get_dead_numbers(probs, k=7):
    idx = np.argsort(probs)[:k]
    return [(int(i), float(probs[i])) for i in idx]

def format_dead_output(dead_list):
    return " - ".join(str(num) for num, prob in dead_list)


# ============================================================
# ================= UI: แสดงผลระบบเลขเด่น ======================
# ============================================================
def show_den_system():
    st.title("🚀 ระบบวิเคราะห์เลขเด่น Ultimate Ensemble")
    st.markdown("**Sequential Draw-to-Draw Edition**")
    st.caption("🔄 ทุกครั้งที่วิเคราะห์จะคำนวณใหม่ทั้งหมด โดยไม่บันทึก Model / Weight / Prediction")
    st.divider()

    col1, col2 = st.columns(2)
    with col1: selected_lotto = st.selectbox("🎯 เลือกหวย:", list(LOTTERY_SOURCES.keys()), index=0, key="den_lotto")
    with col2:
        day_options = {"อัตโนมัติ (คำนวณจากงวดล่าสุด)": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6}
        selected_day_label = st.selectbox("📅 ออกวัน:", list(day_options.keys()), index=0, key="den_day")
        target_dow = day_options[selected_day_label]

    if st.button("🚀 วิเคราะห์เลขเด่น V.Max Sequential", type="primary", use_container_width=True):
        with st.spinner("⏳ กำลังดึงข้อมูลและคำนวณใหม่ทั้งหมด..."):
            df_raw = fetch_and_clean_data_den(LOTTERY_SOURCES[selected_lotto])
            if df_raw.empty: st.stop()
            engine = EnsembleEngine(df_raw, selected_lotto, target_dow=target_dow)

            st.info(f"**⚙️ สเตตัสระบบ [{engine.mode_name}]**\n- 📚 ข้อมูลทั้งหมด: **{len(df_raw)} งวด**\n- 🌲 Trees = **{engine.trees}**\n- 🔄 Backtest = **{engine.test_size}**\n- 🤖 AI = **RF + ET + HGB + XGB**\n- 🧠 Ensemble = **AI + Frequency + Calendar + ST + Pattern + Equation**")

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
            def get_top5_final(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]

            st.divider()
            st.subheader("🔥 สรุปฟันธง เลขเด่นมาแรง")
            st.markdown("🚀 **เด่นบนรวม (ร้อย-สิบ-หน่วย)** : `" + " , ".join([str(x[0]) for x in get_top5_final(probs_top)]) + "`")
            st.markdown("⬇️ **เด่นล่างรวม (สิบ-หน่วย)** : `" + " , ".join([str(x[0]) for x in get_top5_final(probs_bot)]) + "`")

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


# ============================================================
# ================= UI: แสดงผลระบบเลขดับ =======================
# ============================================================
def show_dub_system():
    st.title("🛑 ระบบวิเคราะห์เลขดับ PRO V4.1")
    st.markdown("**Candidate Elimination - 7 ดับ**")
    st.caption("Sequential / No Memory / Walk-Forward")
    st.divider()

    col1, col2 = st.columns(2)
    with col1: target_lotto = st.selectbox("🎯 เลือกหวย:", list(LOTTERY_SOURCES.keys()), index=0, key="dub_lotto")
    with col2:
        day_options = {"อัตโนมัติ (คำนวณจากงวดล่าสุด)": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6}
        selected_day = st.selectbox("📅 ออกวัน:", list(day_options.keys()), index=0, key="dub_day")
        dow_input = day_options[selected_day]

    if st.button("🛑 วิเคราะห์เลขดับ PRO V4.1", type="primary", use_container_width=True):
        with st.spinner("⏳ กำลังดึงข้อมูล + สร้างโมเดลใหม่ + Walk-Forward..."):
            df = fetch_data_dub(LOTTERY_SOURCES[target_lotto])
            if df is None or df.empty:
                st.error("❌ ไม่สามารถดึงข้อมูลได้"); st.stop()
            
            last_date = df["date"].iloc[-1]
            if dow_input is not None:
                days_ahead = dow_input - last_date.weekday()
                if days_ahead <= 0: days_ahead += 7
                target_date, target_dow = last_date + timedelta(days=days_ahead), dow_input
            else:
                gap = (df["date"].iloc[-1] - df["date"].iloc[-2]).days if len(df) >= 2 else 7
                if gap <= 0: gap = 7
                target_date = last_date + timedelta(days=gap)
                target_dow = target_date.weekday()

            cfg = get_adaptive_config_dub(len(df))
            st.info(f"**⚙️ ระบบ [{cfg['mode']}]**\n- 📊 ข้อมูลย้อนหลัง: **{len(df)} งวด**\n- 🔄 Backtest: **{cfg['test_size']} งวด**\n- 🛑 โหมดความจำ: **NO MEMORY / NO MODEL CACHE**")

            dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
            st.markdown(f"### 🔮 ผลการวิเคราะห์เลขดับ\n**งวดเป้าหมาย:** วัน{dow_names[target_dow]} ที่ {target_date.strftime('%d/%m/%Y')}\n**ข้อมูลอ้างอิง:** {len(df)} งวด")
            st.divider()

            positions = {"💯 3 ตัวบน (ร้อย)": "hundred", "🔟 3 ตัวบน (สิบ)": "ten", "1️⃣ 3 ตัวบน (หน่วย)": "unit", "🔽 2 ตัวล่าง (สิบ)": "bot_ten", "⬇️ 2 ตัวล่าง (หน่วย)": "bot_unit"}

            for position_name, col in positions.items():
                system = OptimizedEliminationSystemV4(df, col, target_lotto)
                result = system.analyze(target_date, target_dow)
                if result is None: st.warning(f"⚠️ ข้อมูลไม่เพียงพอ: {position_name}"); continue

                dead_ai, dead_stat, dead_day, dead_final = get_dead_numbers(result["ai"], 7), get_dead_numbers(result["stat"], 7), get_dead_numbers(result["day"], 7), get_dead_numbers(result["final"], 7)
                w_ai, w_stat, w_day = int(result["w_ai"] * 100), int(result["w_stat"] * 100), int(result["w_day"] * 100)

                with st.expander(f"📌 {position_name} (AI {w_ai}% | Stat {w_stat}% | Day {w_day}%)", expanded=True):
                    st.caption(result["bt_msg"])
                    st.markdown(f"- 🤖 **ดับ AI:** `{format_dead_output(dead_ai)}`")
                    st.markdown(f"- 📊 **ดับสถิติ:** `{format_dead_output(dead_stat)}`")
                    st.markdown(f"- 📅 **ดับกำลังวัน:** `{format_dead_output(dead_day)}`")
                    st.success(f"🌟 **ดับสรุปรวม 7 ตัว:** `{format_dead_output(dead_final)}`")


# ============================================================
# ================= MAIN APP (Sidebar) =======================
# ============================================================
st.sidebar.title("🛠️ เมนูการใช้งาน")
st.sidebar.markdown("---")

menu_selection = st.sidebar.radio(
    "กรุณาเลือกระบบที่ต้องการ:",
    ("🟢 วิเคราะห์เลขเด่น", "🔴 วิเคราะห์เลขดับ")
)

st.sidebar.markdown("---")
st.sidebar.caption("แอปพลิเคชันวิเคราะห์สถิติด้วย AI")

if menu_selection == "🟢 วิเคราะห์เลขเด่น":
    show_den_system()
elif menu_selection == "🔴 วิเคราะห์เลขดับ":
    show_dub_system()
