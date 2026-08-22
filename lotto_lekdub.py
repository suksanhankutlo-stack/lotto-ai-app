# ==============================================================================
# 🛑 LOTTO AI PRO V7.9 STABLE CONSENSUS EDITION
# UPGRADES: ANTI-OVERFITTING • HIGHER SMOOTHING • TEMPERATURE SCALING
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. STREAMLIT SETUP
# ==============================================================================

st.set_page_config(
    page_title="ระบบวิเคราะห์เลขดับ PRO V7.9 STABLE CONSENSUS",
    page_icon="🛑",
    layout="centered",
)

st.markdown("""
<style>
.main-title{text-align:center;font-size:32px;font-weight:900;background:-webkit-linear-gradient(45deg,#000,#B71C1C,#4A148C);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:5px;letter-spacing:1.5px}
.sub-title{text-align:center;color:#555;font-size:14px;margin-bottom:20px;font-weight:bold}
</style>
""", unsafe_allow_html=True)

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
                "bot_ten": int(bot[0]), "bot_unit": int(bot[1]),
            })

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ ดึงข้อมูลไม่สำเร็จ: {e}")
        return None

def get_adaptive_config(n):
    # ปรับ max_depth ให้ตื้นขึ้นเพื่อป้องกัน Overfitting จาก Noise
    if n >= 700:
        return {"mode": "STABLE CONSENSUS 700+", "trees": 120, "rf_trees": 60, "max_depth": 5, "bt_steps": 12, "min_train": 60, "lags": [1, 2, 3, 5, 8, 13], "rolls": [3, 5, 10, 20], "wf_recent": 5}
    if n >= 400:
        return {"mode": "STABLE CONSENSUS 400-699", "trees": 100, "rf_trees": 50, "max_depth": 4, "bt_steps": 10, "min_train": 50, "lags": [1, 2, 3, 5, 8], "rolls": [3, 5, 10, 15], "wf_recent": 5}
    if n >= 200:
        return {"mode": "STABLE CONSENSUS 200-399", "trees": 80, "rf_trees": 40, "max_depth": 4, "bt_steps": 8, "min_train": 40, "lags": [1, 2, 3, 5], "rolls": [3, 5, 10], "wf_recent": 4}
    return {"mode": "STABLE CONSENSUS 30-199", "trees": 60, "rf_trees": 30, "max_depth": 3, "bt_steps": 6, "min_train": 30, "lags": [1, 2, 3], "rolls": [3, 5], "wf_recent": 4}

@st.cache_data(show_spinner=False)
def build_features_cached(df, target_col, lags, rolls):
    x = df.copy()
    target = pd.to_numeric(x[target_col], errors="coerce")
    prev = target.shift(1)
    
    x["prev_val"] = prev
    x["prev_even"] = (prev % 2 == 0).astype(np.float32)
    x["prev_high"] = (prev >= 5).astype(np.float32)
    x["mirror"] = (prev + 5) % 10

    dt = x["date"].dt
    weekday = dt.weekday.astype(float)
    x["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    
    for lag in lags: x[f"lag_{lag}"] = target.shift(lag)
    shifted = target.shift(1)
    
    for w in rolls:
        x[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        for d in range(10): x[f"freq_{w}_{d}"] = shifted.eq(d).astype(np.float32).rolling(w, min_periods=1).mean()

    prev_arr = shifted.to_numpy()
    idx = np.arange(len(prev_arr))
    for d in range(10):
        hit = np.where(np.isfinite(prev_arr) & (prev_arr == d), idx, -1)
        last_seen = np.maximum.accumulate(hit)
        x[f"skip_{d}"] = np.clip(np.where(last_seen >= 0, idx - last_seen, 60), 0, 60)

    return x.replace([np.inf, -np.inf], np.nan).fillna(-1)

def normalize_probs(p, temperature=1.0):
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size != 10: p = np.ones(10)/10
    p = np.maximum(p, 1e-9)
    p = np.power(p, 1.0 / max(float(temperature), 1e-6))
    return p / p.sum() if p.sum() > 0 else np.ones(10)/10

class SingularityStatSystem:
    @staticmethod
    def markov_blend(seq):
        seq = np.asarray(seq, dtype=int)
        if len(seq) < 15: return np.ones(10) / 10
        mask = seq[:-1] == seq[-1]
        counts = np.bincount(seq[1:][mask], minlength=10).astype(float)
        counts += 2.5 # [UPDATE]: เพิ่ม Smoothing เพื่อลด Overconfidence ของความสุ่ม
        return normalize_probs(counts)

    @staticmethod
    def mtbo_skip(seq):
        seq = np.asarray(seq, dtype=int)
        if len(seq) == 0: return np.ones(10) / 10
        result = np.zeros(10, dtype=float)
        global_repeat_rate = np.mean(seq[1:] == seq[:-1]) if len(seq) > 1 else 0.1
        for d in range(10):
            pos = np.where(seq == d)[0]
            if len(pos) > 1:
                gaps = np.diff(pos)
                avg_gap, std_gap = np.mean(gaps), np.std(gaps) + 0.1
            else:
                avg_gap, std_gap = 10.0, 5.0
            current_gap = len(seq) - pos[-1] - 1 if len(pos) else 60
            z = (current_gap - avg_gap) / std_gap
            prob_z = 1 / (1 + np.exp(-np.clip(z, -8, 8))) # [UPDATE]: คลิป Z-score ให้แคบลง
            freq = np.mean(seq == d)
            result[d] = (0.7 * prob_z) + (0.3 * freq)
        return normalize_probs(result)

    @staticmethod
    def day_probability(df, target_col, target_dow):
        mask = df["date"].dt.weekday == target_dow
        values = df.loc[mask, target_col].astype(int).to_numpy()
        counts = np.bincount(values, minlength=10).astype(float)
        counts += 3.0 # [UPDATE]: เพิ่มตัวหารดึงเข้าค่า Uniform ลดปัญหาจำข้อมูลจาก Sample size เล็กๆ
        return normalize_probs(counts)

def make_models(cfg):
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=160, C=0.08, random_state=42)),
        "ET": ExtraTreesClassifier(n_estimators=cfg["trees"], max_depth=cfg["max_depth"], min_samples_leaf=6, max_features="sqrt", random_state=43, n_jobs=-1), # [UPDATE] min_samples_leaf=6
        "RF": RandomForestClassifier(n_estimators=cfg["rf_trees"], max_depth=cfg["max_depth"], min_samples_leaf=6, max_features="log2", random_state=44, n_jobs=-1),
        "HGB": HistGradientBoostingClassifier(max_iter=50, max_depth=cfg["max_depth"], learning_rate=0.03, min_samples_leaf=8, l2_regularization=3.5, random_state=45), # [UPDATE] regularize แน่นขึ้น
    }

def model_probs(model, X):
    try:
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(X)[0]
            result = np.zeros(10, dtype=float)
            for i, cls in enumerate(model.classes_): result[int(cls)] = raw[i]
            return normalize_probs(result)
        else: # For RidgeClassifier
            raw = model.decision_function(X)[0]
            exp_raw = np.exp(raw - np.max(raw))
            result = np.zeros(10, dtype=float)
            for i, cls in enumerate(model.classes_): result[int(cls)] = exp_raw[i]
            return normalize_probs(result)
    except: return np.ones(10)/10

@st.cache_data(show_spinner=False)
def cached_walk_forward(X_np, y_np, dates_np, cfg_tuple, target_values_np, target_dow_values_np):
    min_train, steps = map(int, cfg_tuple)
    n = len(y_np)
    if n <= min_train + 2: return {"ai":0.5, "stat":0.5, "day":0.5, "steps":0, "history":[], "ai_hits":[], "stat_hits":[], "day_hits":[]}

    indices = np.arange(max(min_train, n - steps), n)
    dates = pd.to_datetime(dates_np)
    ai_hits, stat_hits, day_hits, history = [], [], [], []

    for i in indices:
        X_train, y_train, actual = pd.DataFrame(X_np[:i]), y_np[:i], int(y_np[i])
        X_one = pd.DataFrame(X_np[i:i+1])

        # [UPDATE] ใช้ RidgeClassifier ที่เสถียรและเร็วมากๆ ในการทำ WF Proxy (ลดความแปรปรวนจาก Tree)
        proxy_model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=5.0, random_state=i))
        try:
            proxy_model.fit(X_train, y_train)
            p_ai = model_probs(proxy_model, X_one)
        except: p_ai = np.ones(10)/10
        
        hist_vals = y_np[:i]
        p_stat = normalize_probs(0.4 * SingularityStatSystem.markov_blend(hist_vals) + 0.6 * SingularityStatSystem.mtbo_skip(hist_vals))
        p_day = SingularityStatSystem.day_probability(pd.DataFrame({"date": dates[:i], "v": hist_vals}), "v", target_dow_values_np[i])

        ai_hit = int(actual in np.argsort(p_ai)[-7:])
        stat_hit = int(actual in np.argsort(p_stat)[-7:])
        day_hit = int(actual in np.argsort(p_day)[-7:])
        ai_hits.append(ai_hit); stat_hits.append(stat_hit); day_hits.append(day_hit)
        
        dead_7 = sorted(np.argsort(normalize_probs((p_ai + p_stat + p_day) / 3.0))[:7].tolist())
        history.append({"date": dates[i].strftime("%d/%m/%Y"), "actual": actual, "dead_nums": dead_7, "is_success": actual not in dead_7})

    return {"ai": np.mean(ai_hits), "stat": np.mean(stat_hits), "day": np.mean(day_hits), "steps": len(ai_hits), "history": history, "ai_hits": ai_hits, "stat_hits": stat_hits, "day_hits": day_hits}

@st.cache_data(show_spinner=False)
def cached_ai_predict(X_train_np, y_train_np, X_predict_np, cfg_tuple):
    cfg = {"trees": int(cfg_tuple[0]), "rf_trees": int(cfg_tuple[1]), "max_depth": int(cfg_tuple[2])}
    models = make_models(cfg)
    weights = {"LR": 0.20, "ET": 0.35, "RF": 0.15, "HGB": 0.30}
    result, total = np.zeros(10), 0.0
    for name, base in models.items():
        w = weights[name]
        try:
            m = clone(base).fit(pd.DataFrame(X_train_np), y_train_np)
            result += model_probs(m, pd.DataFrame(X_predict_np)) * w
            total += w
        except: continue
    return normalize_probs(result / total) if total > 0 else np.ones(10)/10

def calculate_dynamic_weights(bt, base=(0.50, 0.35, 0.15), recent_k=5):
    scores, stabilities = [], []
    for name in ["ai", "stat", "day"]:
        hits = np.asarray(bt.get(f"{name}_hits", []), dtype=float)
        if len(hits) == 0: scores.append(0.5); stabilities.append(0.5); continue
        rk = min(recent_k, len(hits))
        score = (0.75 * np.mean(hits)) + (0.25 * np.mean(hits[-rk:]))
        stab = float(np.clip(1.0 - np.std(hits), 0.0, 1.0)) if len(hits) >= 2 else 0.5
        scores.append(score); stabilities.append(stab)

    scores = np.asarray(scores)
    excess = np.clip(scores - 0.70, -0.10, 0.10) # ลด excess cap
    signal = np.exp(1.8 * excess)
    raw = np.asarray(base) * signal * (0.85 + 0.15 * np.asarray(stabilities))
    
    shrink = 0.25 + (1.0 - min(1.0, bt.get("steps", 0) / 10.0)) * 0.30
    w = ((1 - shrink) * (raw / raw.sum())) + (shrink * np.asarray(base))
    return {"w_ai": w[0], "w_stat": w[1], "w_day": w[2], "bt_ai": scores[0], "bt_stat": scores[1], "bt_day": scores[2]}

def final_consensus(ai, stat, day, w_ai, w_stat, w_day, n):
    stack = np.vstack([ai, stat, day])
    mean_probs = w_ai * ai + w_stat * stat + w_day * day
    std_probs = np.std(stack, axis=0)

    penalty_factor = float(np.interp(n, [30, 200, 800], [0.15, 0.22, 0.30])) # ลด penalty เล็กน้อย
    variance_penalty = penalty_factor * std_probs
    consensus_bonus = 0.05 * (1.0 - np.clip(std_probs / 0.20, 0, 1)) * mean_probs

    score = mean_probs - variance_penalty + consensus_bonus
    # [UPDATE]: ใส่ temperature=1.15 เพื่อเกลี่ยค่าลด Overconfidence
    return normalize_probs(np.maximum(score, 1e-9), temperature=1.15), float(np.max(variance_penalty)), float(np.max(consensus_bonus))

class SingularityAI:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.n = len(df)
        self.cfg = get_adaptive_config(self.n)

    def analyze(self, target_date, target_dow):
        if self.n < 30: return None
        extended = pd.concat([self.df, pd.DataFrame([{"date": target_date, "draw_num": "000", **{k:np.nan for k in ["hundred", "ten", "unit", "bot_ten", "bot_unit"]} }])], ignore_index=True)
        features = build_features_cached(extended, self.target_col, tuple(self.cfg["lags"]), tuple(self.cfg["rolls"]))

        drop_cols = ["date", "draw_num", "hundred", "ten", "unit", "bot_ten", "bot_unit", self.target_col]
        X_all = features.iloc[:-1].drop(columns=drop_cols, errors="ignore")
        X_pred = features.iloc[[-1]][X_all.columns]
        y_all = self.df[self.target_col].astype(int)

        bt = cached_walk_forward(X_all.to_numpy(dtype=np.float32), y_all.to_numpy(dtype=int), self.df["date"].to_numpy(), (self.cfg["min_train"], self.cfg["bt_steps"]), y_all.to_numpy(dtype=int), self.df["date"].dt.weekday.to_numpy(dtype=int))
        weights = calculate_dynamic_weights(bt, recent_k=self.cfg["wf_recent"])
        
        ai_probs = cached_ai_predict(X_all.to_numpy(dtype=np.float32), y_all.to_numpy(dtype=int), X_pred.to_numpy(dtype=np.float32), (self.cfg["trees"], self.cfg["rf_trees"], self.cfg["max_depth"]))
        
        seq = y_all.to_numpy(dtype=int)
        p_stat = normalize_probs(0.4 * SingularityStatSystem.markov_blend(seq) + 0.6 * SingularityStatSystem.mtbo_skip(seq))
        p_day = SingularityStatSystem.day_probability(self.df, self.target_col, target_dow)

        final, std_max, consensus_max = final_consensus(ai_probs, p_stat, p_day, weights["w_ai"], weights["w_stat"], weights["w_day"], self.n)

        return {"ai": ai_probs, "stat": p_stat, "day": p_day, "final": final, **weights, "bt_steps": bt["steps"], "std_max": std_max, "consensus_max": consensus_max, "history": bt["history"]}

def get_dead_numbers(probs, k=7): return [(int(i), float(probs[i])) for i in np.argsort(probs)[:k]]
def format_dead(dead_list): return " • ".join(str(num) for num, _ in dead_list)

def target_date_from_last(df, dow_input):
    last_date = df["date"].iloc[-1]
    if dow_input is None:
        gap = min(31, max(1, (df["date"].iloc[-1] - df["date"].iloc[-2]).days)) if len(df) >= 2 else 7
        return last_date + timedelta(days=gap), (last_date + timedelta(days=gap)).weekday()
    delta = 7 if (dow_input - last_date.weekday()) % 7 == 0 else (dow_input - last_date.weekday()) % 7
    return last_date + timedelta(days=delta), dow_input

st.markdown('<div class="main-title">🛑 LOTTO AI PRO V7.9</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">STABLE CONSENSUS • ANTI-OVERFITTING • SMOOTHED PROBABILITY</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
target_lotto = col1.selectbox("🎯 เลือกหวย", list(LOTTO_URLS.keys()), index=0)
day_options = {"อัตโนมัติ (จากงวดล่าสุด)": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6}
dow_input = day_options[col2.selectbox("📅 ออกวัน", list(day_options.keys()), index=0)]

if st.button("🛑 วิเคราะห์เลขดับ 7 ตัว ⚡ V7.9 STABLE RUN", type="primary", use_container_width=True):
    with st.spinner("⚡ V7.9 กำลังคำนวณ Stable Consensus & Anti-Overfitting..."):
        df = fetch_data(target_lotto)
        if df is None or df.empty: st.error("❌ ไม่สามารถดึงข้อมูลได้"); st.stop()

        target_date, target_dow = target_date_from_last(df, dow_input)
        st.info(f"📅 **งวดเป้าหมาย:** วัน{['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์'][target_dow]} {target_date.strftime('%d/%m/%Y')} | อ้างอิง {len(df)} งวด")

        store_final_probs = {}
        progress = st.progress(0, text="Init V7.9 Stable Engine...")
        positions = {"💯 3 ตัวบน (ร้อย)": "hundred", "🔟 3 ตัวบน (สิบ)": "ten", "1️⃣ 3 ตัวบน (หน่วย)": "unit", "🔽 2 ตัวล่าง (สิบ)": "bot_ten", "⬇️ 2 ตัวล่าง (หน่วย)": "bot_unit"}

        for pos_idx, (position_name, col) in enumerate(positions.items(), start=1):
            progress.progress(pos_idx / len(positions), text=f"⚡ วิเคราะห์ {position_name}")
            result = SingularityAI(df, col).analyze(target_date, target_dow)
            if not result: continue

            store_final_probs[col] = result["final"]
            dead_final, dead_ai, dead_stat, dead_day = map(lambda p: get_dead_numbers(p, 7), [result["final"], result["ai"], result["stat"], result["day"]])

            html_card = (
                f'<div style="background:#fff;border-radius:12px;border:1px solid #e0e0e0;box-shadow:0 8px 20px rgba(0,0,0,.06);padding:20px;margin-bottom:20px">'
                f'<div style="font-size:20px;font-weight:900;color:#222;border-bottom:2px solid #f0f0f0;padding-bottom:10px;margin-bottom:15px">{position_name}</div>'
                f'<div style="background:linear-gradient(135deg,#fff5f5,#ffebee);border:2px solid #ffcdd2;border-radius:12px;padding:20px;text-align:center;margin-bottom:20px">'
                f'<div style="color:#D32F2F;font-weight:800;font-size:15px;margin-bottom:8px">🚫 ดับเอกฉันท์ 7 ตัว (V7.9)</div>'
                f'<div style="font-size:36px;font-weight:900;color:#B71C1C;letter-spacing:8px">{format_dead(dead_final)}</div></div>'
                f'<div style="display:flex;flex-direction:column;gap:10px;margin-bottom:20px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;background:#f8f9fa;padding:12px 15px;border-radius:8px;border-left:4px solid #1976D2"><span style="font-size:14px">🤖 <b>AI Ensemble</b></span><span style="color:#1565C0;font-weight:800">{format_dead(dead_ai)}</span></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;background:#f8f9fa;padding:12px 15px;border-radius:8px;border-left:4px solid #388E3C"><span style="font-size:14px">📊 <b>Stat/MTBO</b></span><span style="color:#2E7D32;font-weight:800">{format_dead(dead_stat)}</span></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;background:#f8f9fa;padding:12px 15px;border-radius:8px;border-left:4px solid #F57C00"><span style="font-size:14px">📅 <b>วันออก</b></span><span style="color:#E65100;font-weight:800">{format_dead(dead_day)}</span></div></div>'
                f'<div style="background:#263238;border-radius:10px;padding:15px;color:#CFD8DC;font-size:12px;text-align:center">'
                f'WF {result["bt_steps"]} งวด | AI Win {result["bt_ai"]*100:.0f}% (W: {result["w_ai"]*100:.0f}%) | Stat {result["bt_stat"]*100:.0f}% (W: {result["w_stat"]*100:.0f}%) | Day {result["bt_day"]*100:.0f}% (W: {result["w_day"]*100:.0f}%)<br>'
                f'<span style="color:#FF8A65">Variance Penalty: {result["std_max"]*100:.2f}% | Consensus Bonus: {result["consensus_max"]*100:.2f}%</span></div></div>'
            )
            st.markdown(html_card, unsafe_allow_html=True)
            
            with st.expander(f"🕰️ ประวัติเลขดับย้อนหลัง ({position_name})"):
                if result.get("history"):
                    st.dataframe(pd.DataFrame([{"งวดวันที่": h["date"], "เลขดับ": " - ".join(map(str, h["dead_nums"])), "ออก": h["actual"], "ผลลัพธ์": "✅ ดับอยู่" if h["is_success"] else "❌ ดับหลุด"} for h in result["history"][-10:][::-1]]), use_container_width=True, hide_index=True)

        progress.empty()

        st.subheader("🔥 สรุปเลขดับเอกฉันท์ (รวม)")
        c1, c2 = st.columns(2)
        if all(x in store_final_probs for x in ["hundred", "ten", "unit"]):
            c1.markdown(f'<div style="background:#fff5f5;padding:20px;border-radius:12px;border:2px solid #ffcdd2;text-align:center;font-weight:900;color:#B71C1C">🚫 ดับบนรวม 7 ตัว<br><span style="font-size:28px;letter-spacing:4px">{format_dead(get_dead_numbers(normalize_probs(sum(store_final_probs[k] for k in ["hundred", "ten", "unit"])/3), 7))}</span></div>', unsafe_allow_html=True)
        if all(x in store_final_probs for x in ["bot_ten", "bot_unit"]):
            c2.markdown(f'<div style="background:#fff5f5;padding:20px;border-radius:12px;border:2px solid #ffcdd2;text-align:center;font-weight:900;color:#B71C1C">🚫 ดับล่างรวม 7 ตัว<br><span style="font-size:28px;letter-spacing:4px">{format_dead(get_dead_numbers(normalize_probs(sum(store_final_probs[k] for k in ["bot_ten", "bot_unit"])/2), 7))}</span></div>', unsafe_allow_html=True)

        st.divider()
        st.caption("🛡️ V7.9 STABLE UPGRADE: ควบคุม Overfitting • เพิ่ม Laplace Smoothing • ลดความสวิงของค่าสถิติด้วย Temperature Scaling")
