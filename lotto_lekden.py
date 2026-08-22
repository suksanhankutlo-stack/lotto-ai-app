# ============================================================
# 🚀 LOTTO AI ULTIMATE V.MAX 5-TOP TURBO (UPDATED)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import warnings

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

# ============================================================
# 0. CONFIG
# ============================================================
st.set_page_config(
    page_title="Lotto AI V.MAX 5-TOP TURBO",
    page_icon="🚀",
    layout="centered"
)
RANDOM_SEED = 42

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
# 2. UI STYLE
# ============================================================
st.markdown("""
<style>
.main-title { text-align:center; font-size:29px; font-weight:900; color:#D32F2F; margin-top:5px; }
.sub-title { text-align:center; color:#666; font-size:14px; margin-bottom:18px; }
.hot-card { padding:18px; border-radius:16px; border:2px solid #ff4b4b; margin:10px 0; background:linear-gradient(to bottom right,#ffffff,#fff5f5); box-shadow:0 4px 8px rgba(255,75,75,.10); }
.number-highlight { font-size:36px; font-weight:900; color:#D32F2F; letter-spacing:2px; }
.dot-sep { color:#FFCDD2; font-size:26px; margin:0 8px; }
.badge-ai { background:#E3F2FD; color:#1565C0; padding:4px 10px; border-radius:15px; font-weight:800; border:1px solid #BBDEFB; }
.badge-stat { background:#E8F5E9; color:#2E7D32; padding:4px 10px; border-radius:15px; font-weight:800; border:1px solid #C8E6C9; }
.badge-cal { background:#FFF3E0; color:#E65100; padding:4px 10px; border-radius:15px; font-weight:800; border:1px solid #FFE0B2; }
.position-title { font-size:20px; font-weight:800; margin-top:20px; color:#333; border-bottom:2px solid #eee; padding-bottom:6px; }
.info-row { margin:8px 0; font-size:14px; }
.stat-box { padding:10px; border-radius:12px; background:#f8f9fa; border:1px solid #e5e5e5; text-align:center; margin-bottom:8px; }
.small-muted { font-size:12px; color:#888; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. DATA FETCH (Updated with Retry & Session)
# ============================================================
@st.cache_data(ttl=180, show_spinner=False)
def fetch_and_clean_data(url):
    try:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        main = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content"))
        if main is None: main = soup

        lines = main.get_text(separator="\n").split("\n")
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
        num_pattern = re.compile(r"\b(\d{3})\b.*?\b(\d{2})\b|\b(\d{5,6})\b.*?\b(\d{2})\b")
        current_date = pd.Timestamp(datetime.now())
        rows = []

        for line in lines:
            line = line.strip()
            if not line: continue
            dm = date_pattern.search(line)
            if dm:
                try:
                    d = pd.to_datetime(dm.group(1), errors="coerce")
                    if not pd.isna(d): current_date = d
                except Exception: pass

            nm = num_pattern.search(line)
            if not nm: continue

            if nm.group(1):
                r3 = nm.group(1)
                r2 = nm.group(2)
            elif nm.group(3):
                r3 = nm.group(3)[-3:]
                r2 = nm.group(4)
            else: continue

            rows.append({
                "Date": current_date,
                "Result_3D": str(r3).zfill(3),
                "Result_2D": str(r2).zfill(2)
            })

        if len(rows) < 10: raise ValueError("ข้อมูลน้อยเกินไป")

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna().drop_duplicates().sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ ดึงข้อมูลไม่ได้: {e}")
        return pd.DataFrame()

# ============================================================
# 4. SAFE FEATURE ENGINEERING (Optimized Lags & Rolls)
# ============================================================
@st.cache_data(show_spinner=False)
def build_features(df, lags=(1, 2, 3), rolls=(5, 10)):
    x = df.copy()
    r3 = x["Result_3D"].astype(str)
    r2 = x["Result_2D"].astype(str)

    x["H"] = r3.str[0].astype(np.int8)
    x["T"] = r3.str[1].astype(np.int8)
    x["O"] = r3.str[2].astype(np.int8)
    x["T2"] = r2.str[0].astype(np.int8)
    x["O2"] = r2.str[1].astype(np.int8)

    x["DOW"] = x["Date"].dt.dayofweek.astype(np.int8)
    x["Month"] = x["Date"].dt.month.astype(np.int8)
    x["Day"] = x["Date"].dt.day.astype(np.int8)
    x["DayOfYear"] = x["Date"].dt.dayofyear.astype(np.int16)
    x["Gap"] = x["Date"].diff().dt.days.fillna(7).clip(0, 60).astype(np.int16)

    x["DOW_SIN"] = np.sin(2 * np.pi * x["DOW"] / 7)
    x["DOW_COS"] = np.cos(2 * np.pi * x["DOW"] / 7)
    x["MONTH_SIN"] = np.sin(2 * np.pi * x["Month"] / 12)
    x["MONTH_COS"] = np.cos(2 * np.pi * x["Month"] / 12)

    ph = x["H"].shift(1)
    pt = x["T"].shift(1)
    po = x["O"].shift(1)

    x["PrevSum"] = (ph + pt + po)
    x["PrevOdd"] = ((ph % 2) + (pt % 2) + (po % 2))
    x["PrevHigh"] = ((ph >= 5).astype(np.int8) + (pt >= 5).astype(np.int8) + (po >= 5).astype(np.int8))
    x["DistHT"] = (ph - pt).abs()
    x["DistTO"] = (pt - po).abs()

    positions = ["H", "T", "O", "T2", "O2"]
    for pos in positions:
        s = x[pos]
        prev = s.shift(1)
        x[f"Odd_{pos}"] = (prev % 2)
        x[f"High_{pos}"] = (prev >= 5).astype(np.int8)
        x[f"Prime_{pos}"] = (prev.isin([2, 3, 5, 7])).astype(np.int8)

        for lag in lags: x[f"L{lag}_{pos}"] = s.shift(lag)
        for w in rolls: x[f"RM{w}_{pos}"] = s.shift(1).rolling(w, min_periods=1).mean()

        arr = s.to_numpy()
        skip = np.full(len(arr), -1, dtype=np.float32)
        last_seen = np.full(10, -1, dtype=np.int32)
        for i in range(len(arr)):
            if i == 0: continue
            prev_value = int(arr[i - 1])
            if 0 <= prev_value <= 9:
                if last_seen[prev_value] >= 0: skip[i] = (i - 1) - last_seen[prev_value]
                else: skip[i] = i
                last_seen[prev_value] = i - 1
        x[f"Skip_{pos}"] = skip
        x[f"RepeatPrev_{pos}"] = (prev == s.shift(2)).astype(np.int8)

    return x.replace([np.inf, -np.inf], np.nan).fillna(-1)

# ============================================================
# 5. ENGINES (Freq, Cal, Trans, Pattern, Eq) 
# ============================================================
class FrequencyEngine:
    def analyze(self, df, pos):
        s = df[pos].astype(int)
        if len(s) == 0: return np.ones(10) / 10
        r15, r30, all_f = s.tail(15).value_counts(normalize=True), s.tail(30).value_counts(normalize=True), s.value_counts(normalize=True)
        score = np.array([(r15.get(d, 0) * 0.55 + r30.get(d, 0) * 0.30 + all_f.get(d, 0) * 0.15) for d in range(10)])
        score += 0.02
        return score / score.sum()

class CalendarEngine:
    def analyze(self, df, pos, next_date):
        subset = df[df["DOW"] == next_date.dayofweek]
        if len(subset) == 0: return np.ones(10) / 10
        all_freq, recent_freq = subset[pos].astype(int).value_counts(normalize=True), subset.tail(min(25, len(subset)))[pos].astype(int).value_counts(normalize=True)
        w_recent = 0.70 if len(subset) >= 25 else (0.55 if len(subset) >= 15 else 0.35)
        score = np.array([(all_freq.get(d, 0) * (1.0 - w_recent) + recent_freq.get(d, 0) * w_recent) for d in range(10)])
        score += 0.02
        return score / score.sum()

class TransitionEngine:
    def analyze(self, df, pos):
        if len(df) < 8: return np.ones(10) / 10
        current, prev = int(df[pos].iloc[-1]), df[pos].shift(1)
        subset = df[prev == current]
        if len(subset) < 2: return np.ones(10) / 10
        freq = subset[pos].astype(int).value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)]) + 0.02
        return score / score.sum()

class PatternEngine:
    def analyze(self, df, pos):
        if len(df) < 10: return np.ones(10) / 10
        a, b = int(df[pos].iloc[-1]), int(df[pos].iloc[-2])
        p1, p2 = df[pos].shift(1), df[pos].shift(2)
        subset = df[(p1 == a) & (p2 == b)]
        if len(subset) < 2: subset = df[p1 == a]
        if len(subset) < 1: return np.ones(10) / 10
        freq = subset[pos].astype(int).value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)]) + 0.02
        return score / score.sum()

class EquationEngine:
    def analyze(self, df):
        if len(df) == 0: return np.ones(10) / 10
        h, t, o = int(df.iloc[-1]["H"]), int(df.iloc[-1]["T"]), int(df.iloc[-1]["O"])
        vals = [(h + t) % 10, (t + o) % 10, abs(h - o) % 10, (h * t) % 10, (h + t + o) % 10, (h * 2 + o) % 10]
        score = np.ones(10, dtype=np.float64) * 0.05
        for v in vals: score[v] += 1.0
        return score / score.sum()

# ============================================================
# 6. AI ENSEMBLE (Updated with Regularization)
# ============================================================
class FastAI:
    def __init__(self, trees=55):
        self.trees = trees
        self.model_weights = {"RF": 0.35, "ET": 0.35, "HGB": 0.30}

    def create_models(self, backtest=False):
        rf_trees, et_trees, hgb_iter = (18, 18, 25) if backtest else (self.trees, self.trees, 60)
        return {
            "RF": RandomForestClassifier(n_estimators=rf_trees, max_depth=4, min_samples_leaf=5, max_samples=0.85, max_features="sqrt", n_jobs=-1, random_state=42),
            "ET": ExtraTreesClassifier(n_estimators=et_trees, max_depth=4, min_samples_leaf=5, bootstrap=True, max_samples=0.85, max_features="sqrt", n_jobs=-1, random_state=43),
            "HGB": HistGradientBoostingClassifier(max_iter=hgb_iter, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=3.0, random_state=44)
        }

    def predict(self, X, y, X_next):
        result = np.zeros(10, dtype=np.float64)
        total_weight = 0.0
        models = self.create_models(backtest=False)
        for name, model in models.items():
            weight = self.model_weights[name]
            try:
                if len(np.unique(y)) >= 2:
                    model.fit(X, y)
                    probs = model.predict_proba(X_next)[0]
                    for cls, p in zip(model.classes_, probs): result[int(cls)] += (float(p) * weight)
                    total_weight += weight
            except Exception: continue
        if total_weight <= 0: return np.ones(10) / 10
        result /= total_weight
        return (result + 1e-9) / (result + 1e-9).sum()

def get_config(n):
    if n >= 700: return {"trees": 55, "bt": 20, "min_train": 60}
    elif n >= 400: return {"trees": 50, "bt": 18, "min_train": 55}
    elif n >= 200: return {"trees": 45, "bt": 15, "min_train": 45}
    elif n >= 100: return {"trees": 40, "bt": 12, "min_train": 40}
    else: return {"trees": 35, "bt": 10, "min_train": 30}

# ============================================================
# 7. ENSEMBLE ENGINE (Batch Retraining Added)
# ============================================================
class EnsembleEngine:
    def __init__(self, df, lottery_name, target_dow=None):
        self.df = df.copy()
        self.lottery_name = lottery_name
        self.target_dow = target_dow
        cfg = get_config(len(df))
        self.trees, self.bt, self.min_train = cfg["trees"], cfg["bt"], cfg["min_train"]
        self.lags, self.rolls = (1, 2, 3), (5, 10)
        
        self.features = ["DOW", "Month", "Day", "DayOfYear", "Gap", "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS", "PrevSum", "PrevOdd", "PrevHigh", "DistHT", "DistTO"]
        for pos in ["H", "T", "O", "T2", "O2"]:
            self.features.extend([f"Odd_{pos}", f"High_{pos}", f"Prime_{pos}", f"Skip_{pos}", f"RepeatPrev_{pos}"])
            for lag in self.lags: self.features.append(f"L{lag}_{pos}")
            for w in self.rolls: self.features.append(f"RM{w}_{pos}")

        self.freq, self.calendar, self.transition, self.pattern, self.equation = FrequencyEngine(), CalendarEngine(), TransitionEngine(), PatternEngine(), EquationEngine()
        self.ai = FastAI(self.trees)
        self.base_weights = {"AI": 0.50, "Freq": 0.16, "ST": 0.11, "Cal": 0.11, "Pattern": 0.08, "Eq": 0.04}

    @staticmethod
    def top_hit(prob, actual, n): return actual in np.argsort(prob)[::-1][:n]

    def backtest(self, pos, X, df_hist):
        n = len(X)
        if n < self.min_train + 5: return self.base_weights.copy(), "ข้อมูลน้อย", [], {"top1":0, "top3":0, "top5":0, "logloss":None}
        
        start = max(self.min_train, n - self.bt)
        scores = {"AI": 0.0, "Freq": 0.0, "ST": 0.0, "Cal": 0.0, "Pattern": 0.0}
        total_decay = 0.0
        history, logloss_values = [], []
        top1_count, top3_count, top5_count = 0, 0, 0
        last_trained_models = None

        for step, idx in enumerate(range(start, n)):
            decay = (1.06 ** step)
            total_decay += decay
            Xtr, ytr = X.iloc[:idx], df_hist[pos].iloc[:idx].astype(int)
            xt, actual, target_date = X.iloc[[idx]], int(df_hist[pos].iloc[idx]), df_hist["Date"].iloc[idx]

            # Batch Retraining every 5 steps
            if step % 5 == 0 or last_trained_models is None:
                models = self.ai.create_models(backtest=True)
                for name, model in models.items():
                    try:
                        if len(np.unique(ytr)) >= 2: model.fit(Xtr, ytr)
                    except Exception: pass
                last_trained_models = models
            else:
                models = last_trained_models

            ai_prob, tmp, total_model_weight = np.ones(10)/10, np.zeros(10, dtype=np.float64), 0.0
            for name, model in models.items():
                mw = self.ai.model_weights[name]
                try:
                    p = model.predict_proba(xt)[0]
                    for cls, val in zip(model.classes_, p): tmp[int(cls)] += float(val) * mw
                    total_model_weight += mw
                except: pass
            if total_model_weight > 0: ai_prob = tmp / total_model_weight

            hist = df_hist.iloc[:idx].copy()
            fq, cal, stp, ptn = self.freq.analyze(hist, pos), self.calendar.analyze(hist, pos, target_date), self.transition.analyze(hist, pos), self.pattern.analyze(hist, pos)

            if self.top_hit(ai_prob, actual, 5): scores["AI"] += decay
            if self.top_hit(fq, actual, 5): scores["Freq"] += decay
            if self.top_hit(cal, actual, 5): scores["Cal"] += decay
            if self.top_hit(stp, actual, 5): scores["ST"] += decay
            if self.top_hit(ptn, actual, 5): scores["Pattern"] += decay

            ai_rank = np.argsort(ai_prob)[::-1]
            if actual == ai_rank[0]: top1_count += 1
            if actual in ai_rank[:3]: top3_count += 1
            if actual in ai_rank[:5]: top5_count += 1

            try: logloss_values.append(log_loss([actual], [ai_prob], labels=list(range(10))))
            except: pass

            combined = (self.base_weights["AI"]*ai_prob + self.base_weights["Freq"]*fq + self.base_weights["Cal"]*cal + self.base_weights["ST"]*stp + self.base_weights["Pattern"]*ptn)
            combined = (combined + 1e-9) / (combined + 1e-9).sum()
            top5 = np.argsort(combined)[::-1][:5].tolist()

            history.append({
                "date_str": target_date.strftime("%d/%m/%Y"),
                "actual": actual,
                "top_5_ordered": top5,
                "is_success": actual in top5,
                "prob": combined.copy() # Store prob for overall backtest calculation
            })

        if total_decay <= 0: return self.base_weights.copy(), "Error", [], {}
        accuracy = {k: scores[k]/total_decay for k in scores}
        
        shrink = 0.60
        weighted = {}
        for k in accuracy:
            stable = shrink*0.50 + (1.0-shrink)*accuracy[k]
            weighted[k] = self.base_weights[k]*(0.40 + 0.60*stable)
        weighted["Eq"] = self.base_weights["Eq"] * 0.40

        total = sum(weighted.values())
        weights_pct = {k: v/total for k,v in weighted.items()} if total > 0 else self.base_weights.copy()
        
        if weights_pct["AI"] > 0.58:
            diff = weights_pct["AI"] - 0.58
            weights_pct["AI"] = 0.58
            others = [k for k in weights_pct if k != "AI"]
            oth_sum = sum(weights_pct[k] for k in others)
            if oth_sum > 0:
                for k in others: weights_pct[k] += diff * (weights_pct[k]/oth_sum)

        metrics = {"top1": top1_count, "top3": top3_count, "top5": top5_count, "total": max(0, n-start), "logloss": np.mean(logloss_values) if logloss_values else None}
        bt_msg = f"WF {metrics['total']} งวด | AI Top-1 {top1_count}/{metrics['total']} | Top-3 {top3_count}/{metrics['total']} | Top-5 {top5_count}/{metrics['total']}"
        
        return weights_pct, bt_msg, history[-10:], metrics

    def process_position(self, pos, hist, X, X_next, next_date):
        weights, bt_msg, history, metrics = self.backtest(pos, X, hist)
        ai = self.ai.predict(X, hist[pos].astype(int), X_next)
        fq, cal, stp, ptn, eq = self.freq.analyze(hist, pos), self.calendar.analyze(hist, pos, next_date), self.transition.analyze(hist, pos), self.pattern.analyze(hist, pos), self.equation.analyze(hist)
        
        final = (weights["AI"]*ai + weights["Freq"]*fq + weights["Cal"]*cal + weights["ST"]*stp + weights["Pattern"]*ptn + weights["Eq"]*eq)
        final = (final + 1e-9) / (final + 1e-9).sum()

        def top_n(p, n): return [(int(i), float(p[i])) for i in np.argsort(p)[::-1][:n]]
        return {"Final": top_n(final, 5), "AI": top_n(ai, 3), "Freq": top_n(fq, 3), "Calendar": top_n(cal, 3), "Prob": final, "Weights": weights, "BT": bt_msg, "Metrics": metrics, "History": history}

    def get_next_date(self):
        last_date = self.df["Date"].iloc[-1]
        if self.target_dow is not None:
            days = (self.target_dow - last_date.dayofweek) % 7
            return last_date + timedelta(days=days if days != 0 else 7)
        if len(self.df) >= 3:
            gaps = self.df["Date"].diff().dt.days.dropna().tail(5)
            if len(gaps) > 0: return last_date + timedelta(days=max(1, min(int(round(gaps.median())), 14)))
        return last_date + timedelta(days=7)

    def predict_all(self):
        next_date = self.get_next_date()
        hist = build_features(self.df, self.lags, self.rolls)
        X = hist[self.features].astype(np.float32)

        next_row = {}
        prev_h, prev_t, prev_o = int(hist["H"].iloc[-1]), int(hist["T"].iloc[-1]), int(hist["O"].iloc[-1])
        next_row["DOW"] = next_date.dayofweek
        next_row["Month"] = next_date.month
        next_row["Day"] = next_date.day
        next_row["DayOfYear"] = next_date.dayofyear
        next_row["Gap"] = max(0, min(60, int((next_date - self.df["Date"].iloc[-1]).days)))
        next_row["DOW_SIN"] = np.sin(2 * np.pi * next_row["DOW"] / 7)
        next_row["DOW_COS"] = np.cos(2 * np.pi * next_row["DOW"] / 7)
        next_row["MONTH_SIN"] = np.sin(2 * np.pi * next_row["Month"] / 12)
        next_row["MONTH_COS"] = np.cos(2 * np.pi * next_row["Month"] / 12)
        next_row["PrevSum"] = (prev_h + prev_t + prev_o)
        next_row["PrevOdd"] = ((prev_h % 2) + (prev_t % 2) + (prev_o % 2))
        next_row["PrevHigh"] = (int(prev_h >= 5) + int(prev_t >= 5) + int(prev_o >= 5))
        next_row["DistHT"] = abs(prev_h - prev_t)
        next_row["DistTO"] = abs(prev_t - prev_o)

        for pos in ["H", "T", "O", "T2", "O2"]:
            s = hist[pos].astype(int)
            prev = int(s.iloc[-1])
            next_row[f"Odd_{pos}"] = prev % 2
            next_row[f"High_{pos}"] = int(prev >= 5)
            next_row[f"Prime_{pos}"] = int(prev in [2, 3, 5, 7])
            for lag in self.lags: next_row[f"L{lag}_{pos}"] = int(s.iloc[-lag]) if len(s) >= lag else -1
            for w in self.rolls: next_row[f"RM{w}_{pos}"] = s.tail(w).mean() if len(s) else -1
            
            indices = np.where(s.to_numpy() == prev)[0]
            next_row[f"Skip_{pos}"] = float(indices[-1] - indices[-2]) if len(indices) >= 2 else float(len(s))
            next_row[f"RepeatPrev_{pos}"] = int(s.iloc[-1] == s.iloc[-2]) if len(s) >= 2 else 0

        X_next = pd.DataFrame([next_row])[self.features].astype(np.float32)
        results = {pos: self.process_position(pos, hist, X, X_next, next_date) for pos in ["H", "T", "O", "T2", "O2"]}
        return results, next_date

# ============================================================
# 8. HTML HELPERS
# ============================================================
def html_top5(items): return '<span class="dot-sep">•</span>'.join([f'<span class="number-highlight">{n}</span>' for n, p in items])
def html_badge(items, badge_class): return f'<span class="{badge_class}">' + " &nbsp;•&nbsp; ".join([str(n) for n, p in items]) + '</span>'
def nums_prob(items): return " | ".join(f"{n} ({p:.1%})" for n, p in items)
def combine_top_n(preds, positions, n=5):
    score = sum(preds[pos]["Prob"] for pos in positions) / len(positions)
    return [(int(i), float(score[i])) for i in np.argsort(score)[::-1][:n]]

def render_overall_history(preds, pos_list, title):
    # Retrieve and merge histories based on the requested positions
    if not preds[pos_list[0]].get("History"): return ""
    
    n_hist = len(preds[pos_list[0]]["History"])
    recent_hist = []
    
    for i in range(n_hist):
        hist_data = [preds[p]["History"][i] for p in pos_list]
        avg_prob = sum(h["prob"] for h in hist_data) / len(pos_list)
        top_5 = np.argsort(avg_prob)[::-1][:5].tolist()
        actuals = [h["actual"] for h in hist_data]
        matches = [n for n in top_5 if n in actuals]
        
        recent_hist.append({
            "date": hist_data[0]["date_str"],
            "top_5": top_5,
            "actuals": actuals,
            "is_success": len(matches) > 0,
            "matches": matches
        })
    
    recent_hist = recent_hist[::-1]
    wins = sum(1 for h in recent_hist if h["is_success"])
    total_hist = len(recent_hist)
    rate = wins / total_hist if total_hist else 0

    html = f'<div class="stat-box">🏆 TOP-5 รูด/วิ่ง {title} เข้า <b>{wins}/{total_hist}</b> ({rate:.0%})</div>'
    html += '<div style="overflow-x:auto;"><table style="width:100%; text-align:center; border-collapse:collapse; font-family:sans-serif; font-size:13px;">'
    html += '<tr style="background:#f1f3f4; color:#333;"><th style="padding:9px; border-bottom:2px solid #ccc;">วันที่</th><th style="padding:9px; border-bottom:2px solid #ccc;">5-TOP</th><th style="padding:9px; border-bottom:2px solid #ccc;">ผลจริง</th><th style="padding:9px; border-bottom:2px solid #ccc;">สถานะ</th></tr>'
    
    for h in recent_hist:
        bg = "#F1F8E9" if h["is_success"] else "#FFEBEE"
        icon = "✅ WIN" if h["is_success"] else "❌ หลุด"
        
        parts = []
        for n in h["top_5"]:
            if n in h["actuals"]: parts.append(f'<span style="color:#D32F2F; font-weight:900; font-size:16px;">{n}</span>')
            else: parts.append(str(n))
        top5_str = " - ".join(parts)
        actuals_str = " - ".join(str(a) for a in h["actuals"])

        html += f'<tr style="background:{bg}; border-bottom:1px solid #ddd;"><td style="padding:9px;">{h["date"]}</td><td style="padding:9px; font-weight:700;">{top5_str}</td><td style="padding:9px; font-weight:800; color:#555;">{actuals_str}</td><td style="padding:9px; font-weight:800;">{icon}</td></tr>'
    
    html += '</table></div>'
    return html

# ============================================================
# 9. UI APP
# ============================================================
st.markdown('<div class="main-title">🚀 LOTTO AI V.MAX</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">5-TOP TURBO • Leakage Safe • Walk-Forward<br><b>RF + ExtraTrees + HGB | AI + สถิติ + วัน + Pattern</b></div>', unsafe_allow_html=True)
st.divider()

c1, c2 = st.columns(2)
selected_lotto = c1.selectbox("🎯 เลือกหวย", list(LOTTERY_SOURCES.keys()))
day_options = {"อัตโนมัติ": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6}
day_label = c2.selectbox("📅 วันออกรางวัล", list(day_options.keys()))

if st.button("🚀 วิเคราะห์เลขเด่นด้วย AI TURBO", type="primary", use_container_width=True):
    with st.spinner("⚡ กำลังดึงข้อมูล + Batch Walk-Forward + AI Ensemble..."):
        df = fetch_and_clean_data(LOTTERY_SOURCES[selected_lotto])
        if df.empty: st.stop()

        st.success(f"โหลดข้อมูลสำเร็จ {len(df)} งวด")
        engine = EnsembleEngine(df, selected_lotto, day_options[day_label])
        preds, next_date = engine.predict_all()

        labels = {"H": "หลักร้อย 3 ตัวบน", "T": "หลักสิบ 3 ตัวบน", "O": "หลักหน่วย 3 ตัวบน", "T2": "หลักสิบ 2 ตัวล่าง", "O2": "หลักหน่วย 2 ตัวล่าง"}
        days = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

        st.divider()
        st.info(f"📅 งวดเป้าหมาย: วัน{days[next_date.dayofweek]} {next_date.strftime('%d-%m-%Y')} | ข้อมูล {len(df)} งวด")

        for pos in ["H", "T", "O", "T2", "O2"]:
            res = preds[pos]
            st.markdown(f'<div class="position-title">📍 {labels[pos]}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="hot-card"><div style="font-weight:700; color:#444; margin-bottom:8px;">🔥 HOT TOP-5</div><div style="text-align:center; margin:10px 0;">{html_top5(res["Final"])}</div><div style="font-size:13px; color:#888; text-align:center; margin-top:8px;">{nums_prob(res["Final"])}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-row">🤖 <b>AI TOP-3:</b> &nbsp; {html_badge(res["AI"], "badge-ai")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-row">📊 <b>สถิติ TOP-3:</b> &nbsp; {html_badge(res["Freq"], "badge-stat")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-row">📅 <b>กำลังวัน TOP-3:</b> &nbsp; {html_badge(res["Calendar"], "badge-cal")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-muted">📈 {res["BT"]}</div>', unsafe_allow_html=True) W = res["Weights"]
            
            st.markdown(f'<div class="small-muted">⚖️ น้ำหนัก: AI {W["AI"]:.0%} | สถิติ {W["Freq"]:.0%} | วัน {W["Cal"]:.0%} | ก้าวเดิน {W["ST"]:.0%} | Pattern {W["Pattern"]:.0%} | Eq {W["Eq"]:.0%}</div>', unsafe_allow_html=True)

        hot_top = combine_top_n(preds, ["H", "T", "O"])
        hot_bottom = combine_top_n(preds, ["T2", "O2"])

        st.divider()
        st.subheader("🔥 สรุปเลขเด่นภาพรวม")
        
        # ภาพรวมบน
        st.markdown(f'<div class="hot-card"><div style="font-weight:700; color:#444; margin-bottom:8px;">🔥 HOT 5-TOP รูด/วิ่ง — บน</div><div style="text-align:center; margin:10px 0;">{html_top5(hot_top)}</div><div style="font-size:13px; color:#888; text-align:center; margin-top:8px;">{nums_prob(hot_top)}</div></div>', unsafe_allow_html=True)
        with st.expander("🕰️ ประวัติย้อนหลัง 10 งวด — รูด/วิ่ง บน", expanded=False):
            st.markdown(render_overall_history(preds, ["H", "T", "O"], "บน"), unsafe_allow_html=True)

        # ภาพรวมล่าง
        st.markdown(f'<div class="hot-card"><div style="font-weight:700; color:#444; margin-bottom:8px;">🔥 HOT 5-TOP รูด/วิ่ง — ล่าง</div><div style="text-align:center; margin:10px 0;">{html_top5(hot_bottom)}</div><div style="font-size:13px; color:#888; text-align:center; margin-top:8px;">{nums_prob(hot_bottom)}</div></div>', unsafe_allow_html=True)
        with st.expander("🕰️ ประวัติย้อนหลัง 10 งวด — รูด/วิ่ง ล่าง", expanded=False):
            st.markdown(render_overall_history(preds, ["T2", "O2"], "ล่าง"), unsafe_allow_html=True)

        st.success("✅ วิเคราะห์เสร็จสิ้น • Leakage Safe • Walk-Forward • Dynamic Weight • AI Ensemble • Top-1/3/5")
        st.caption("หมายเหตุ: ผลลัพธ์เป็นการจัดอันดับความน่าจะเป็นจากข้อมูลย้อนหลัง ไม่สามารถรับประกันผลรางวัลจริงได้")
