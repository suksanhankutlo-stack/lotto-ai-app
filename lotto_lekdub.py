# ==============================================================================
# 🛑 LOTTO AI PRO V7.8 FAST CONSENSUS EDITION
# FAST WF ENSEMBLE • RECENT+LONG WF • STABILITY WEIGHT
# ADAPTIVE VARIANCE PENALTY • CONSENSUS BONUS
# LEAKAGE SAFE • NO PERSISTENT MODEL • STREAMLIT CACHE
# ENSEMBLE: ET + RF + HGB + LR
# ============================================================================

import streamlit as st
import requests
import warnings
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. STREAMLIT SETUP
# ==============================================================================

st.set_page_config(
    page_title="ระบบวิเคราะห์เลขดับ PRO V7.8 FAST CONSENSUS",
    page_icon="🛑",
    layout="centered",
)

st.markdown("""
<style>
.main-title{text-align:center;font-size:32px;font-weight:900;background:-webkit-linear-gradient(45deg,#000,#B71C1C,#4A148C);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:5px;letter-spacing:1.5px}
.sub-title{text-align:center;color:#555;font-size:14px;margin-bottom:20px;font-weight:bold}
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
    "หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html",
}

# ==============================================================================
# 2. FETCH DATA
# ==============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(lotto_name):
    url = LOTTO_URLS.get(lotto_name)
    if not url:
        return None
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        post_body = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content")) or soup
        text = post_body.get_text(separator="\n")

        pattern = re.compile(r"\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d{2})")
        matches = pattern.findall(text)
        if len(matches) < 30:
            return None

        data = []
        for date_str, prize1, bottom2 in matches:
            top = str(prize1).zfill(3)
            bot = str(bottom2).zfill(2)
            data.append({
                "date": date_str,
                "draw_num": top,
                "hundred": int(top[0]),
                "ten": int(top[1]),
                "unit": int(top[2]),
                "bot_ten": int(bot[0]),
                "bot_unit": int(bot[1]),
            })

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = (
            df.dropna(subset=["date"])
              .drop_duplicates(subset=["date"], keep="last")
              .sort_values("date")
              .reset_index(drop=True)
        )
        return df
    except Exception as e:
        st.error(f"❌ ดึงข้อมูลไม่สำเร็จ: {e}")
        return None

# ==============================================================================
# 3. ADAPTIVE CONFIG — SPEED PRESERVED
# ==============================================================================

def get_adaptive_config(n):
    if n >= 700:
        return {
            "mode": "FAST CONSENSUS 700+", "trees": 110, "rf_trees": 55,
            "max_depth": 7, "bt_steps": 10, "min_train": 60,
            "lags": [1, 2, 3, 5, 8, 13], "rolls": [3, 5, 10, 20],
            "wf_et": 16, "wf_recent": 5,
        }
    if n >= 400:
        return {
            "mode": "FAST CONSENSUS 400-699", "trees": 90, "rf_trees": 45,
            "max_depth": 6, "bt_steps": 9, "min_train": 50,
            "lags": [1, 2, 3, 5, 8], "rolls": [3, 5, 10, 15],
            "wf_et": 15, "wf_recent": 5,
        }
    if n >= 200:
        return {
            "mode": "FAST CONSENSUS 200-399", "trees": 70, "rf_trees": 35,
            "max_depth": 6, "bt_steps": 8, "min_train": 40,
            "lags": [1, 2, 3, 5], "rolls": [3, 5, 10],
            "wf_et": 14, "wf_recent": 4,
        }
    return {
        "mode": "FAST CONSENSUS 30-199", "trees": 50, "rf_trees": 25,
        "max_depth": 5, "bt_steps": 7, "min_train": 30,
        "lags": [1, 2, 3], "rolls": [3, 5],
        "wf_et": 12, "wf_recent": 4,
    }

# ==============================================================================
# 4. LEAKAGE-SAFE FEATURE ENGINEERING
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
    x["prev_prime"] = prev.isin([2, 3, 5, 7]).astype(np.float32)
    x["prev_mod3"] = prev % 3

    dt = x["date"].dt
    weekday = dt.weekday.astype(float)
    x["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    x["day_of_month"] = dt.day.astype(float)
    x["month"] = dt.month.astype(float)
    x["month_sin"] = np.sin(2 * np.pi * dt.month.astype(float) / 12)
    x["month_cos"] = np.cos(2 * np.pi * dt.month.astype(float) / 12)

    for lag in lags:
        x[f"lag_{lag}"] = target.shift(lag)
    x["diff_1"] = (prev - target.shift(2)).abs()
    x["repeat_prev2"] = (prev == target.shift(2)).astype(np.float32)
    x["repeat_prev3"] = (prev == target.shift(3)).astype(np.float32)

    shifted = target.shift(1)
    for w in rolls:
        r = shifted.rolling(w, min_periods=1)
        x[f"roll_mean_{w}"] = r.mean()
        x[f"roll_std_{w}"] = r.std().fillna(0)
        x[f"ema_{w}"] = shifted.ewm(span=w, adjust=False).mean()
        x[f"repeat_rate_{w}"] = (shifted.eq(shifted.shift(1)).astype(float).rolling(w, min_periods=1).mean())

        # Recent digit frequencies — all based on shifted history only.
        for d in range(10):
            x[f"freq_{w}_{d}"] = shifted.eq(d).astype(np.float32).rolling(w, min_periods=1).mean()

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
# 5. PROBABILITY HELPERS
# ==============================================================================

def normalize_probs(p, temperature=1.0):
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size != 10:
        out = np.ones(10, dtype=float) / 10
        out[:min(10, p.size)] = p[:min(10, p.size)]
        p = out
    p[~np.isfinite(p)] = 0
    p = np.maximum(p, 1e-9)
    p = np.power(p, 1.0 / max(float(temperature), 1e-6))
    total = p.sum()
    return p / total if total > 0 else np.ones(10) / 10


def model_probs(model, X):
    try:
        raw = model.predict_proba(X)[0]
        result = np.zeros(10, dtype=float)
        for i, cls in enumerate(model.classes_):
            c = int(cls)
            if 0 <= c <= 9:
                result[c] = raw[i]
        return normalize_probs(result)
    except Exception:
        return np.ones(10) / 10

# ==============================================================================
# 6. STATISTICAL SYSTEM
# ==============================================================================

class SingularityStatSystem:
    @staticmethod
    def markov_blend(seq):
        seq = np.asarray(seq, dtype=int)
        if len(seq) < 15:
            return np.ones(10) / 10
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
        if n == 0:
            return np.ones(10) / 10
        result = np.zeros(10, dtype=float)
        global_repeat_rate = np.mean(seq[1:] == seq[:-1]) if n > 1 else 0.1
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
            prob_z = 1 / (1 + np.exp(-np.clip(z, -12, 12)))
            if current_gap == 0 and len(pos) > 0:
                specific_repeats = np.sum((seq[:-1] == d) & (seq[1:] == d))
                specific_repeat_rate = specific_repeats / max(len(pos), 1)
                rescue_score = (0.6 * specific_repeat_rate) + (0.4 * global_repeat_rate)
                prob_z = max(prob_z, rescue_score * 2.5)
            elif current_gap == 1 and len(pos) > 0:
                prob_z = max(prob_z, 0.15)
            freq = np.mean(seq == d)
            result[d] = (0.7 * prob_z) + (0.3 * freq)
        return normalize_probs(result)

    @staticmethod
    def day_probability(df, target_col, target_dow):
        mask = df["date"].dt.weekday == target_dow
        values = df.loc[mask, target_col].astype(int).to_numpy()
        if len(values) < 5:
            return np.ones(10) / 10
        counts = np.bincount(values, minlength=10).astype(float)
        counts += 1.0
        return normalize_probs(counts)

# ==============================================================================
# 7. MODEL BUILDERS
# NOTE: No class_weight='balanced' — digit classes are structurally near-uniform.
# ==============================================================================

def make_models(cfg):
    return {
        "LR": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=160, C=0.12, random_state=42)
        ),
        "ET": ExtraTreesClassifier(
            n_estimators=cfg["trees"], max_depth=cfg["max_depth"],
            min_samples_leaf=4, max_features="sqrt", bootstrap=True,
            max_samples=0.88, random_state=43, n_jobs=-1
        ),
        "RF": RandomForestClassifier(
            n_estimators=cfg["rf_trees"], max_depth=cfg["max_depth"],
            min_samples_leaf=4, max_features="log2", bootstrap=True,
            max_samples=0.88, random_state=44, n_jobs=-1
        ),
        "HGB": HistGradientBoostingClassifier(
            max_iter=60, max_depth=min(5, cfg["max_depth"]),
            learning_rate=0.035, min_samples_leaf=5,
            l2_regularization=2.5, random_state=45
        ),
    }

# ==============================================================================
# 8. CACHED FULL AI TRAIN — unchanged data => no retraining in Streamlit cache
# ==============================================================================

@st.cache_data(show_spinner=False)
def cached_ai_predict(X_train_np, y_train_np, X_predict_np, cfg_tuple):
    trees, rf_trees, max_depth = map(int, cfg_tuple)
    cfg = {"trees": trees, "rf_trees": rf_trees, "max_depth": max_depth}
    models = make_models(cfg)
    weights = {"LR": 0.15, "ET": 0.40, "RF": 0.15, "HGB": 0.30}
    result = np.zeros(10, dtype=float)
    total = 0.0
    Xtr = pd.DataFrame(X_train_np)
    Xpr = pd.DataFrame(X_predict_np)
    y = np.asarray(y_train_np, dtype=int)
    for name, base in models.items():
        w = weights[name]
        try:
            m = clone(base)
            m.fit(Xtr, y)
            result += model_probs(m, Xpr) * w
            total += w
        except Exception:
            continue
    return normalize_probs(result / total) if total > 0 else np.ones(10) / 10

# ==============================================================================
# 9. FAST WALK-FORWARD BACKTEST
# Two tiny models only: ET + LR. Fewer steps than V7.7 keeps speed high.
# ==============================================================================

@st.cache_data(show_spinner=False)
def cached_walk_forward(X_np, y_np, dates_np, cfg_tuple, target_values_np, target_dow_values_np):
    n = len(y_np)
    min_train, steps, et_trees = map(int, cfg_tuple)
    if n <= min_train + 2:
        return {"ai": 0.5, "stat": 0.5, "day": 0.5, "steps": 0, "history": [], "ai_hits": [], "stat_hits": [], "day_hits": []}

    start = max(min_train, n - steps)
    indices = np.arange(start, n)
    values = np.asarray(y_np, dtype=int)
    dates = pd.to_datetime(dates_np)
    ai_hits, stat_hits, day_hits = [], [], []
    history = []

    for i in indices:
        X_train = pd.DataFrame(X_np[:i])
        X_one = pd.DataFrame(X_np[i:i+1])
        y_train = values[:i]
        actual = int(values[i])

        # Fast AI proxy ensemble.
        p_ai_parts = []
        try:
            et = ExtraTreesClassifier(
                n_estimators=et_trees, max_depth=5, min_samples_leaf=4,
                max_features="sqrt", bootstrap=True, max_samples=0.9,
                random_state=100 + int(i), n_jobs=-1
            )
            et.fit(X_train, y_train)
            p_ai_parts.append(model_probs(et, X_one))
        except Exception:
            pass
        try:
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=120, C=0.12, random_state=200 + int(i)))
            lr.fit(X_train, y_train)
            p_ai_parts.append(model_probs(lr, X_one))
        except Exception:
            pass
        p_ai = normalize_probs(np.mean(p_ai_parts, axis=0)) if p_ai_parts else np.ones(10) / 10

        hist_vals = values[:i]
        p_stat = normalize_probs(
            0.4 * SingularityStatSystem.markov_blend(hist_vals)
            + 0.6 * SingularityStatSystem.mtbo_skip(hist_vals)
        )
        target_day = int(dates[i].weekday())
        hist_df = pd.DataFrame({"date": dates[:i], "v": values[:i]})
        p_day = SingularityStatSystem.day_probability(hist_df, "v", target_day)

        ai_hit = int(actual in np.argsort(p_ai)[-7:])
        stat_hit = int(actual in np.argsort(p_stat)[-7:])
        day_hit = int(actual in np.argsort(p_day)[-7:])
        ai_hits.append(ai_hit); stat_hits.append(stat_hit); day_hits.append(day_hit)

        combined = normalize_probs((p_ai + p_stat + p_day) / 3.0)
        dead_7 = np.argsort(combined)[:7]
        history.append({
            "date": dates[i].strftime("%d/%m/%Y"),
            "actual": actual,
            "dead_nums": sorted(dead_7.tolist()),
            "is_success": bool(actual not in dead_7),
            "ai_hit": bool(ai_hit), "stat_hit": bool(stat_hit), "day_hit": bool(day_hit),
        })

    return {
        "ai": float(np.mean(ai_hits)) if ai_hits else 0.5,
        "stat": float(np.mean(stat_hits)) if stat_hits else 0.5,
        "day": float(np.mean(day_hits)) if day_hits else 0.5,
        "steps": len(ai_hits), "history": history,
        "ai_hits": ai_hits, "stat_hits": stat_hits, "day_hits": day_hits,
    }

# ==============================================================================
# 10. WEIGHT ENGINE — LONG + RECENT + STABILITY
# ==============================================================================

def source_score(long_hits, recent_k=5):
    a = np.asarray(long_hits, dtype=float)
    if len(a) == 0:
        return 0.5, 0.5, 0.0
    long_acc = float(np.mean(a))
    rk = min(recent_k, len(a))
    recent_acc = float(np.mean(a[-rk:]))
    # Stability: 1 for consistent sequence, lower for high fluctuation.
    if len(a) >= 2:
        stability = float(np.clip(1.0 - np.std(a), 0.0, 1.0))
    else:
        stability = 0.5
    score = 0.70 * long_acc + 0.30 * recent_acc
    score = 0.85 * score + 0.15 * stability
    return score, stability, recent_acc


def calculate_dynamic_weights(bt, base=(0.50, 0.35, 0.15), recent_k=5):
    names = ["ai", "stat", "day"]
    bases = np.asarray(base, dtype=float)
    scores, stabilities, recent = [], [], []
    for name in names:
        s, st, r = source_score(bt.get(f"{name}_hits", []), recent_k)
        scores.append(s); stabilities.append(st); recent.append(r)

    scores = np.asarray(scores)
    stabilities = np.asarray(stabilities)

    # Baseline is 70% because Top-7 coverage has a random baseline of 7/10.
    excess = np.clip(scores - 0.70, -0.15, 0.15)
    signal = np.exp(2.2 * excess)
    raw = bases * signal * (0.85 + 0.15 * stabilities)
    raw = raw / raw.sum()

    # Shrink toward baseline weights. More WF observations => less shrink.
    steps = int(bt.get("steps", 0))
    reliability = min(1.0, steps / 10.0)
    shrink = 0.20 + (1.0 - reliability) * 0.25
    w = (1 - shrink) * raw + shrink * bases
    w = w / w.sum()
    return {
        "w_ai": float(w[0]), "w_stat": float(w[1]), "w_day": float(w[2]),
        "score_ai": float(scores[0]), "score_stat": float(scores[1]), "score_day": float(scores[2]),
        "stab_ai": float(stabilities[0]), "stab_stat": float(stabilities[1]), "stab_day": float(stabilities[2]),
        "recent_ai": float(recent[0]), "recent_stat": float(recent[1]), "recent_day": float(recent[2]),
    }

# ==============================================================================
# 11. FINAL CONSENSUS ENGINE
# ==============================================================================

def final_consensus(ai, stat, day, w_ai, w_stat, w_day, n):
    stack = np.vstack([ai, stat, day])
    mean_probs = w_ai * ai + w_stat * stat + w_day * day
    std_probs = np.std(stack, axis=0)

    # Adaptive penalty: lower for small data, stronger for large stable data.
    penalty_factor = float(np.interp(n, [30, 100, 200, 400, 800], [0.22, 0.25, 0.28, 0.32, 0.36]))
    variance_penalty = penalty_factor * std_probs

    # Consensus bonus rewards agreement, but remains intentionally small.
    agreement = 1.0 - np.clip(std_probs / 0.20, 0, 1)
    consensus_bonus = 0.055 * agreement * mean_probs

    score = mean_probs - variance_penalty + consensus_bonus
    score = np.maximum(score, 1e-9)
    return normalize_probs(score), float(np.max(variance_penalty)), float(np.max(consensus_bonus)), float(penalty_factor)

# ==============================================================================
# 12. ANALYZER
# ==============================================================================

class SingularityAI:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.n = len(df)
        self.cfg = get_adaptive_config(self.n)

    def analyze(self, target_date, target_dow):
        if self.n < 30:
            return None

        future = {
            "date": target_date, "draw_num": "000", "hundred": np.nan,
            "ten": np.nan, "unit": np.nan, "bot_ten": np.nan, "bot_unit": np.nan,
        }
        extended = pd.concat([self.df, pd.DataFrame([future])], ignore_index=True)
        features = build_features_cached(extended, self.target_col, tuple(self.cfg["lags"]), tuple(self.cfg["rolls"]))

        drop_cols = ["date", "draw_num", "hundred", "ten", "unit", "bot_ten", "bot_unit", self.target_col]
        X_all_df = features.iloc[:-1].drop(columns=drop_cols, errors="ignore")
        X_predict_df = features.iloc[[-1]][X_all_df.columns]
        y_all = self.df[self.target_col].astype(int)

        # Fast WF cache. No retraining if the same data/features/config are unchanged.
        bt_cfg = (self.cfg["min_train"], self.cfg["bt_steps"], self.cfg["wf_et"])
        bt = cached_walk_forward(
            X_all_df.to_numpy(dtype=np.float32), y_all.to_numpy(dtype=int),
            self.df["date"].to_numpy(), bt_cfg,
            y_all.to_numpy(dtype=int), self.df["date"].dt.weekday.to_numpy(dtype=int)
        )

        weights = calculate_dynamic_weights(bt, base=(0.50, 0.35, 0.15), recent_k=self.cfg["wf_recent"])

        ai_probs = cached_ai_predict(
            X_all_df.to_numpy(dtype=np.float32),
            y_all.to_numpy(dtype=int),
            X_predict_df.to_numpy(dtype=np.float32),
            (self.cfg["trees"], self.cfg["rf_trees"], self.cfg["max_depth"]),
        )

        seq = y_all.to_numpy(dtype=int)
        p_stat = normalize_probs(
            0.4 * SingularityStatSystem.markov_blend(seq)
            + 0.6 * SingularityStatSystem.mtbo_skip(seq)
        )
        p_day = SingularityStatSystem.day_probability(self.df, self.target_col, target_dow)

        final, std_max, consensus_max, penalty_factor = final_consensus(
            ai_probs, p_stat, p_day,
            weights["w_ai"], weights["w_stat"], weights["w_day"], self.n
        )

        return {
            "ai": ai_probs, "stat": p_stat, "day": p_day, "final": final,
            **weights,
            "bt_ai": bt["ai"], "bt_stat": bt["stat"], "bt_day": bt["day"],
            "bt_steps": bt["steps"], "std_max": std_max,
            "consensus_max": consensus_max, "penalty_factor": penalty_factor,
            "history": bt["history"],
        }

# ==============================================================================
# 13. UI HELPERS
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
        if delta == 0:
            delta = 7
        target_date = last_date + timedelta(days=delta)
    return target_date, target_dow

# ==============================================================================
# 14. MAIN UI
# ==============================================================================

st.markdown('<div class="main-title">🛑 LOTTO AI PRO V7.8</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">FAST CONSENSUS • RECENT + LONG WF • STABILITY • ADAPTIVE VARIANCE</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    target_lotto = st.selectbox("🎯 เลือกหวย", list(LOTTO_URLS.keys()), index=0)
with col2:
    day_options = {
        "อัตโนมัติ (จากงวดล่าสุด)": None, "วันจันทร์": 0, "วันอังคาร": 1,
        "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4,
        "วันเสาร์": 5, "วันอาทิตย์": 6,
    }
    day_label = st.selectbox("📅 ออกวัน", list(day_options.keys()), index=0)
    dow_input = day_options[day_label]

if st.button("🛑 วิเคราะห์เลขดับ 7 ตัว ⚡ V7.8 FAST RUN", type="primary", use_container_width=True):
    with st.spinner("⚡ V7.8 กำลังคำนวณ Fast WF + Consensus + Adaptive Variance..."):
        df = fetch_data(target_lotto)
        if df is None or df.empty:
            st.error("❌ ไม่สามารถดึงข้อมูลได้")
            st.stop()

        target_date, target_dow = target_date_from_last(df, dow_input)
        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
        st.info(f"📅 **งวดเป้าหมาย:** วัน{dow_names[target_dow]} {target_date.strftime('%d/%m/%Y')} | อ้างอิง {len(df)} งวด")

        cfg = get_adaptive_config(len(df))
        st.caption(f"⚙️ {cfg['mode']} | Cached Training | Fast WF {cfg['bt_steps']} งวด | Baseline 70% | No class balancing")

        positions = {
            "💯 3 ตัวบน (ร้อย)": "hundred",
            "🔟 3 ตัวบน (สิบ)": "ten",
            "1️⃣ 3 ตัวบน (หน่วย)": "unit",
            "🔽 2 ตัวล่าง (สิบ)": "bot_ten",
            "⬇️ 2 ตัวล่าง (หน่วย)": "bot_unit",
        }

        store_final_probs = {}
        progress = st.progress(0, text="Init V7.8 Fast Consensus Engine...")

        for pos_idx, (position_name, col) in enumerate(positions.items(), start=1):
            progress.progress(pos_idx / len(positions), text=f"⚡ วิเคราะห์ {position_name}")
            system = SingularityAI(df, col)
            result = system.analyze(target_date, target_dow)
            if result is None:
                continue

            store_final_probs[col] = result["final"]
            dead_final = get_dead_numbers(result["final"], 7)
            dead_ai = get_dead_numbers(result["ai"], 7)
            dead_stat = get_dead_numbers(result["stat"], 7)
            dead_day = get_dead_numbers(result["day"], 7)

            edge_ai = (result["bt_ai"] - 0.70) * 100
            edge_stat = (result["bt_stat"] - 0.70) * 100
            edge_day = (result["bt_day"] - 0.70) * 100

            html_card = (
                '<div style="background:#fff;border-radius:12px;border:1px solid #e0e0e0;box-shadow:0 8px 20px rgba(0,0,0,.06);padding:20px;margin-bottom:20px">'
                f'<div style="font-size:20px;font-weight:900;color:#222;border-bottom:2px solid #f0f0f0;padding-bottom:10px;margin-bottom:15px">{position_name}</div>'
                '<div style="background:linear-gradient(135deg,#fff5f5,#ffebee);border:2px solid #ffcdd2;border-radius:12px;padding:20px;text-align:center;margin-bottom:20px">'
                '<div style="color:#D32F2F;font-weight:800;font-size:15px;margin-bottom:8px">🚫 ดับเอกฉันท์ 7 ตัว (V7.8 Consensus)</div>'
                f'<div style="font-size:36px;font-weight:900;color:#B71C1C;letter-spacing:8px">{format_dead(dead_final)}</div>'
                '</div>'
                '<div style="display:flex;flex-direction:column;gap:10px;margin-bottom:20px">'
                '<div style="display:flex;justify-content:space-between;align-items:center;background:#f8f9fa;padding:12px 15px;border-radius:8px;border-left:4px solid #1976D2">'
                '<span style="font-size:14px;color:#333">🤖 <b>AI Ensemble</b></span>'
                f'<span style="background:#E3F2FD;color:#1565C0;padding:6px 15px;border-radius:20px;font-weight:800;font-size:15px;letter-spacing:3px">{format_dead(dead_ai)}</span></div>'
                '<div style="display:flex;justify-content:space-between;align-items:center;background:#f8f9fa;padding:12px 15px;border-radius:8px;border-left:4px solid #388E3C">'
                '<span style="font-size:14px;color:#333">📊 <b>สถิติ MTBO + Markov</b></span>'
                f'<span style="background:#E8F5E9;color:#2E7D32;padding:6px 15px;border-radius:20px;font-weight:800;font-size:15px;letter-spacing:3px">{format_dead(dead_stat)}</span></div>'
                '<div style="display:flex;justify-content:space-between;align-items:center;background:#f8f9fa;padding:12px 15px;border-radius:8px;border-left:4px solid #F57C00">'
                '<span style="font-size:14px;color:#333">📅 <b>วัน</b></span>'
                f'<span style="background:#FFF3E0;color:#E65100;padding:6px 15px;border-radius:20px;font-weight:800;font-size:15px;letter-spacing:3px">{format_dead(dead_day)}</span></div>'
                '</div>'
                '<div style="background:#263238;border-radius:10px;padding:15px;color:#CFD8DC;font-family:sans-serif">'
                f'<div style="text-align:center;color:#fff;font-weight:bold;font-size:14px;margin-bottom:12px">⚡ Fast Walk-Forward ({result["bt_steps"]} งวด)</div>'
                '<div style="display:flex;justify-content:space-between;text-align:center;border-bottom:1px solid #37474F;padding-bottom:12px;margin-bottom:12px">'
                '<div style="flex:1"><div style="color:#64B5F6;font-size:12px">🤖 AI</div>'
                f'<div style="font-size:18px;color:#fff;font-weight:900">{result["bt_ai"]*100:.0f}%</div><div style="font-size:11px;color:#90A4AE">Edge {edge_ai:+.0f}% | W {(result["w_ai"]*100):.0f}%</div></div>'
                '<div style="flex:1;border-left:1px solid #37474F;border-right:1px solid #37474F"><div style="color:#81C784;font-size:12px">📊 สถิติ</div>'
                f'<div style="font-size:18px;color:#fff;font-weight:900">{result["bt_stat"]*100:.0f}%</div><div style="font-size:11px;color:#90A4AE">Edge {edge_stat:+.0f}% | W {(result["w_stat"]*100):.0f}%</div></div>'
                '<div style="flex:1"><div style="color:#FFB74D;font-size:12px">📅 วัน</div>'
                f'<div style="font-size:18px;color:#fff;font-weight:900">{result["bt_day"]*100:.0f}%</div><div style="font-size:11px;color:#90A4AE">Edge {edge_day:+.0f}% | W {(result["w_day"]*100):.0f}%</div></div>'
                '</div>'
                f'<div style="text-align:center;color:#FFB74D;font-size:12px;font-weight:700">📈 Recent AI {result["recent_ai"]*100:.0f}% | Stability {result["stab_ai"]*100:.0f}%</div>'
                f'<div style="text-align:center;color:#FF8A65;font-size:12px;margin-top:5px">⚠️ Variance Penalty {result["std_max"]*100:.2f}% | Consensus Bonus {result["consensus_max"]*100:.2f}%</div>'
                '</div></div>'
            )
            st.markdown(html_card, unsafe_allow_html=True)

            with st.expander(f"🕰️ ประวัติเลขดับย้อนหลัง 10 งวด ({position_name})", expanded=False):
                if result.get("history"):
                    recent_hist = result["history"][-10:][::-1]
                    rows = []
                    for h in recent_hist:
                        status = "✅ ดับอยู่" if h["is_success"] else "❌ ดับหลุด"
                        rows.append({
                            "งวดวันที่": h["date"],
                            "เลขดับ 7 ตัว": " - ".join(map(str, h["dead_nums"])),
                            "ออกจริง": h["actual"],
                            "ผลลัพธ์": status,
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("ข้อมูลไม่เพียงพอสำหรับการสร้างตารางย้อนหลัง")

        progress.empty()

        # SUMMARY
        st.subheader("🔥 สรุปเลขดับเอกฉันท์ (Top / Bottom)")
        col_sum1, col_sum2 = st.columns(2)

        if all(x in store_final_probs for x in ["hundred", "ten", "unit"]):
            top_probs = normalize_probs((store_final_probs["hundred"] + store_final_probs["ten"] + store_final_probs["unit"]) / 3.0)
            dead_top = get_dead_numbers(top_probs, 7)
            with col_sum1:
                st.markdown(
                    f'<div style="background:#fff5f5;padding:20px;border-radius:12px;border:2px solid #ffcdd2;text-align:center">'
                    f'<div style="font-weight:900;color:#d32f2f;font-size:18px">🚫 ดับบนรวม 7 ตัว</div>'
                    f'<div style="font-size:28px;font-weight:900;color:#B71C1C;margin-top:15px;letter-spacing:4px">{format_dead(dead_top)}</div></div>',
                    unsafe_allow_html=True,
                )

        if all(x in store_final_probs for x in ["bot_ten", "bot_unit"]):
            bot_probs = normalize_probs((store_final_probs["bot_ten"] + store_final_probs["bot_unit"]) / 2.0)
            dead_bot = get_dead_numbers(bot_probs, 7)
            with col_sum2:
                st.markdown(
                    f'<div style="background:#fff5f5;padding:20px;border-radius:12px;border:2px solid #ffcdd2;text-align:center">'
                    f'<div style="font-weight:900;color:#d32f2f;font-size:18px">🚫 ดับล่างรวม 7 ตัว</div>'
                    f'<div style="font-size:28px;font-weight:900;color:#B71C1C;margin-top:15px;letter-spacing:4px">{format_dead(dead_bot)}</div></div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.caption("🛡️ V7.8 FAST CONSENSUS: Fast WF Ensemble • Recent+Long WF • Stability Weight • Adaptive Variance • Consensus Bonus • Leakage Safe • Cached Training")
        st.warning("⚠️ ระบบเป็นการจัดอันดับเชิงสถิติ/แมชชีนเลิร์นนิง ไม่สามารถรับประกันผลสลากที่เป็นกระบวนการสุ่มได้")
