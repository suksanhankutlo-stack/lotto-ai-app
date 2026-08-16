# ============================================================
# 🚀 LOTTO AI ULTIMATE ENSEMBLE (TOP 3 EDITION)
# SEQUENTIAL DRAW-TO-DRAW + HIGH PRECISION UI
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
    page_title="AI หวยขั้นเทพ (Top 3)",
    page_icon="🎯",
    layout="centered" # หรือเปลี่ยนเป็น "wide" ถ้าชอบจอใหญ่
)

# Custom CSS ทำให้อ่านง่ายขึ้น
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; color: #E63946; }
    .number-box { padding: 10px; border-radius: 10px; background-color: #f0f2f6; text-align: center; margin-bottom: 10px;}
    .number-highlight { font-size: 30px; font-weight: 900; color: #1D3557; }
    </style>
""", unsafe_allow_html=True)

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
@st.cache_data(ttl=3600) # แคชข้อมูล 1 ชม. ช่วยให้โหลดเร็วขึ้น
def fetch_and_clean_data(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content"))
        if main_content is None: main_content = soup
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
                try: current_date = pd.to_datetime(date_match.group(1), errors="coerce")
                except Exception: pass

            num_match = num_pattern.search(line)
            if not num_match: continue

            if num_match.group(1) and num_match.group(2): res3d, res2d = num_match.group(1), num_match.group(2)
            elif num_match.group(3) and num_match.group(4): res3d, res2d = num_match.group(3)[-3:], num_match.group(4)
            else: continue

            extracted.append({"Date": current_date, "Result_3D": str(res3d).zfill(3), "Result_2D": str(res2d).zfill(2)})

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
# 3. FEATURE ENGINEERING
# ============================================================
def build_features(df, lags, rolls):
    df_feat = df.copy()
    df_feat["H"] = df_feat["Result_3D"].astype(str).str[0].astype(int)
    df_feat["T"] = df_feat["Result_3D"].astype(str).str[1].astype(int)
    df_feat["O"] = df_feat["Result_3D"].astype(str).str[2].astype(int)
    df_feat["T2"] = df_feat["Result_2D"].astype(str).str[0].astype(int)
    df_feat["O2"] = df_feat["Result_2D"].astype(str).str[1].astype(int)

    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["Day"] = df_feat["Date"].dt.day
    df_feat["Gap"] = df_feat["Date"].diff().dt.days.fillna(7).clip(lower=0).astype(int)

    df_feat["Dist_HT"] = (df_feat["H"].shift(1) - df_feat["T"].shift(1)).abs().fillna(0)
    df_feat["Dist_TO"] = (df_feat["T"].shift(1) - df_feat["O"].shift(1)).abs().fillna(0)

    prime_digits = [2, 3, 5, 7]
    for pos in ["H", "T", "O", "T2", "O2"]:
        prev = df_feat[pos].shift(1)
        df_feat[f"OddEven_{pos}"] = (prev % 2).fillna(0)
        df_feat[f"HighLow_{pos}"] = (prev >= 5).fillna(0).astype(int)
        df_feat[f"IsPrime_{pos}"] = prev.isin(prime_digits).astype(int)
        df_feat[f"Mirror_{pos}"] = ((prev + 5) % 10).fillna(0)

        for lag in lags: df_feat[f"Lag_{lag}_{pos}"] = df_feat[pos].shift(lag)
        for w in rolls:
            shifted = df_feat[pos].shift(1)
            df_feat[f"Roll_{w}_Mean_{pos}"] = shifted.rolling(w).mean()
            df_feat[f"Roll_{w}_Std_{pos}"] = shifted.rolling(w).std()

        if f"Lag_1_{pos}" in df_feat.columns and f"Lag_2_{pos}" in df_feat.columns:
            df_feat[f"Repeat_{pos}"] = (df_feat[f"Lag_1_{pos}"] == df_feat[f"Lag_2_{pos}"]).astype(int)
        else: df_feat[f"Repeat_{pos}"] = 0

        shifted = df_feat[pos].shift(1)
        for d in range(10): df_feat[f"Hot20_{pos}_{d}"] = shifted.eq(d).rolling(20).sum()

        skips, last_seen, values = np.zeros(len(df_feat)), {}, df_feat[pos].values
        for i in range(len(values)):
            if i == 0: skips[i] = 100
            else:
                prev_value = values[i - 1]
                skips[i] = (i - last_seen[prev_value]) if prev_value in last_seen else i
            last_seen[values[i]] = i
        df_feat[f"Skip_{pos}"] = skips

    return df_feat.replace([np.inf, -np.inf], np.nan).fillna(-1)

# ============================================================
# 4. SUB-SYSTEMS (Eq, Freq, Cal, ST, PTN)
# ============================================================
class PositionalEquation:
    def analyze(self, df):
        latest = df.iloc[-1]
        H, T, O = latest["H"], latest["T"], latest["O"]
        probs = np.zeros(10)
        values = [(H+T)%10, (T+O)%10, abs(H-O)%10, (H*T)%10, (H+T+O)%10, (H*2+O)%10]
        for v in values: probs[int(v)] += 1
        probs += 0.1
        return probs / probs.sum()

class FrequencyEngine:
    def analyze(self, df, pos):
        series = df[pos].dropna()
        probs = np.zeros(10)
        if len(series) == 0: return np.ones(10) / 10
        freq_all = series.value_counts(normalize=True).to_dict()
        freq_15 = series.tail(15).value_counts(normalize=True).to_dict()
        for i in range(10):
            idxs = np.where(series.values == i)[0]
            skip = (len(series) - 1 - idxs[-1]) if len(idxs) > 0 else len(series)
            probs[i] = (freq_all.get(i, 0) * 0.3) + (freq_15.get(i, 0) * 0.5) + ((1.0 / (skip + 1)) * 0.2)
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
            ("rf", RandomForestClassifier(n_estimators=trees, max_depth=6, min_samples_leaf=2, class_weight='balanced', random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=trees, max_depth=6, min_samples_leaf=2, class_weight='balanced', random_state=42)),
            ("hgb", HistGradientBoostingClassifier(max_iter=50, learning_rate=0.05, max_leaf_nodes=15, random_state=42))
        ]
        weights = [rf_w, et_w, hgb_w]
        if xgb_w > 0:
            estimators.append(("xgb", XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.05, tree_method="hist", random_state=42)))
            weights.append(xgb_w)
        self.voting = VotingClassifier(estimators=estimators, voting="soft", weights=weights)

    def analyze(self, X_train, y_train, X_next):
        model = copy.deepcopy(self.voting)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_next)[0]
        result = np.zeros(10)
        for c, p in zip(model.classes_, probs): result[int(c)] = p
        if result.sum() <= 0: return np.ones(10) / 10
        return result / result.sum()

# ============================================================
# 5. ENSEMBLE ENGINE (TOP 3 TUNED)
# ============================================================
class EnsembleEngine:
    def __init__(self, df_raw, lottery_name, target_dow=None):
        self.df_raw = df_raw.copy()
        self.target_dow = target_dow
        n = len(df_raw)

        # ปรับค่า Lags/Rolls ให้จับระยะสั้นได้ดีขึ้น
        self.lags = [1, 2, 3, 4, 5, 8]
        self.rolls = [2, 4, 10, 20] 

        if n >= 400: self.trees, self.test_size, self.early_stop = 100, 25, 13; self.ai_weights = (1.0, 1.0, 1.0, 1.0)
        elif n >= 200: self.trees, self.test_size, self.early_stop = 80, 20, 10; self.ai_weights = (1.0, 0.8, 0.6, 0.5)
        else: self.trees, self.test_size, self.early_stop = 60, 15, 8; self.ai_weights = (1.0, 0.8, 0.5, 0.1)

        if n < 100: self.test_size = min(5, max(0, n - 30))

        self.features = ["DayOfWeek", "Month", "Day", "Gap", "Dist_HT", "Dist_TO"]
        for pos in ["H", "T", "O", "T2", "O2"]:
            self.features.extend([f"OddEven_{pos}", f"HighLow_{pos}", f"IsPrime_{pos}", f"Mirror_{pos}", f"Skip_{pos}", f"Repeat_{pos}"])
            for lag in self.lags: self.features.append(f"Lag_{lag}_{pos}")
            for w in self.rolls: self.features.extend([f"Roll_{w}_Mean_{pos}", f"Roll_{w}_Std_{pos}"])
            for d in range(10): self.features.append(f"Hot20_{pos}_{d}")

        self.pos_sys, self.freq_sys, self.cond_sys = PositionalEquation(), FrequencyEngine(), ConditionalSystem()
        self.st_sys, self.ptn_sys = StateTransitionSystem(), PatternBacktestSystem()
        self.ai_sys = AISystem(self.trees, *self.ai_weights)
        
        # ปรับ Base Weights ให้น้ำหนัก AI และ ความถี่ (ความร้อนแรง) มากขึ้น
        self.base_weights = {"AI": 0.40, "Freq": 0.25, "ST": 0.10, "Cal": 0.10, "BT": 0.10, "Eq": 0.05}

    def _process_single_position(self, pos, df_hist, X_all, next_x, next_date):
        bt_size = self.test_size
        if len(df_hist) < bt_size + 30 or bt_size <= 0:
            norm_weights = self.base_weights.copy()
            bt_msg = "(ข้อมูลน้อย ข้าม Backtest)"
        else:
            ai_hits, fq_hits, cal_hits, st_hits, ptn_hits = 0, 0, 0, 0, 0
            steps_run, total_decay = 0, 0

            for i in range(bt_size):
                curr_train_len = len(X_all) - bt_size + i
                if curr_train_len < 30: continue
                
                # Decay Weight ดันความสำคัญของงวดล่าสุดให้สูงขึ้น
                decay_weight = 1.10 ** i
                total_decay += decay_weight

                X_train_step = X_all.iloc[:curr_train_len]
                y_train_step = df_hist[pos].iloc[:curr_train_len]
                actual_val = df_hist[pos].iloc[curr_train_len]

                proxy_model = ExtraTreesClassifier(n_estimators=15, max_depth=4, random_state=42)
                proxy_model.fit(X_train_step, y_train_step)
                probs = proxy_model.predict_proba(X_all.iloc[[curr_train_len]])[0]
                ai_res = np.zeros(10)
                for idx, c in enumerate(proxy_model.classes_): ai_res[int(c)] = probs[idx]
                
                # เช็คความแม่นระดับ Top 3 (เดิม Top 4) บังคับให้คัดแต่โมเดลที่แม่นจริงๆ
                if actual_val in np.argsort(ai_res)[::-1][:3]: ai_hits += decay_weight

                curr_df = df_hist.iloc[:curr_train_len]
                t_date = df_hist["Date"].iloc[curr_train_len]

                if actual_val in np.argsort(self.freq_sys.analyze(curr_df, pos))[::-1][:3]: fq_hits += decay_weight
                if actual_val in np.argsort(self.cond_sys.analyze(curr_df, pos, t_date))[::-1][:3]: cal_hits += decay_weight
                if actual_val in np.argsort(self.st_sys.analyze(curr_df, pos))[::-1][:3]: st_hits += decay_weight
                if actual_val in np.argsort(self.ptn_sys.analyze(curr_df, pos))[::-1][:3]: ptn_hits += decay_weight

                steps_run += 1
                if steps_run >= self.early_stop: break

            if steps_run <= 0 or total_decay == 0:
                norm_weights = self.base_weights.copy()
                bt_msg = "(Backtest ไม่พร้อม)"
            else:
                w_ai = self.base_weights["AI"] * max(0.1, ai_hits / total_decay) ** 2
                w_fq = self.base_weights["Freq"] * max(0.1, fq_hits / total_decay) ** 2
                w_cal = self.base_weights["Cal"] * max(0.1, cal_hits / total_decay) ** 2
                w_st = self.base_weights["ST"] * max(0.1, st_hits / total_decay) ** 2
                w_bt = self.base_weights["BT"] * max(0.1, ptn_hits / total_decay) ** 2
                w_eq = self.base_weights["Eq"] * 0.1
                
                total = w_ai + w_fq + w_cal + w_st + w_bt + w_eq
                norm_weights = {"AI": w_ai/total, "Freq": w_fq/total, "Cal": w_cal/total, "ST": w_st/total, "BT": w_bt/total, "Eq": w_eq/total}
                bt_msg = f"🔍 Backtest {steps_run} งวด (Top 3 Precision): AI {int(ai_hits/total_decay*100)}% | Freq {int(fq_hits/total_decay*100)}% | Ptn {int(ptn_hits/total_decay*100)}%"

        p_ai = self.ai_sys.analyze(X_all, df_hist[pos], next_x)
        p_fq = self.freq_sys.analyze(df_hist, pos)
        p_cal = self.cond_sys.analyze(df_hist, pos, next_date)
        p_st = self.st_sys.analyze(df_hist, pos)
        p_bt = self.ptn_sys.analyze(df_hist, pos)
        p_eq = self.pos_sys.analyze(df_hist)

        W = norm_weights
        final_score = (W["AI"]*p_ai + W["Freq"]*p_fq + W["Cal"]*p_cal + W["ST"]*p_st + W["BT"]*p_bt + W["Eq"]*p_eq)
        total = final_score.sum()
        final_score = (np.ones(10)/10) if total <= 0 else (final_score / total)

        # ฟังก์ชันคัดเฉพาะ 3 อันดับแรก
        def get_top3(probs):
            return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:3]

        return pos, {"AI": get_top3(p_ai), "Stat": get_top3(p_fq), "Final": get_top3(final_score), "Probs_For_Graph": final_score, "Weights": norm_weights, "BT_Msg": bt_msg}

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

        results = {}
        for pos in ["H", "T", "O", "T2", "O2"]:
            _, data = self._process_single_position(pos, df_hist, X_all, next_x, next_date)
            results[pos] = data
        return results, next_date

# ============================================================
# 6. STREAMLIT UI
# ============================================================
st.title("🎯 AI วิเคราะห์หวยขั้นเทพ (TOP 3)")
st.caption("✨ ปรับปรุงความแม่นยำด้วย Top 3 Time-Decay Backtest & อัปเกรดการแสดงผลให้อ่านง่าย")
st.divider()

col1, col2 = st.columns(2)
with col1: selected_lotto = st.selectbox("📌 เลือกหวย:", list(LOTTERY_SOURCES.keys()), index=0)
with col2:
    day_options = {"อัตโนมัติ": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6}
    target_dow = day_options[st.selectbox("📅 วันที่ออก:", list(day_options.keys()), index=0)]

if st.button("🚀 เริ่มวิเคราะห์เลขเด่น (Top 3)", type="primary", use_container_width=True):
    with st.spinner("⏳ AI กำลังรวบรวมสถิติและคัด 3 ตัวตึง..."):
        df_raw = fetch_and_clean_data(LOTTERY_SOURCES[selected_lotto])
        if df_raw.empty: st.stop()

        engine = EnsembleEngine(df_raw, selected_lotto, target_dow=target_dow)
        preds, next_date = engine.predict_all()
        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
        labels = {"H": "หลักร้อย (บน)", "T": "หลักสิบ (บน)", "O": "หลักหน่วย (บน)", "T2": "หลักสิบ (ล่าง)", "O2": "หลักหน่วย (ล่าง)"}

        # ----------------------------------------------------
        # ส่วนที่ 1: สรุปฟันธง (ดึงมาไว้บนสุด)
        # ----------------------------------------------------
        st.markdown(f"### 🔮 ผลฟันธง ประจำวัน{dow_names[next_date.dayofweek]} ที่ {next_date.strftime('%d/%m/%Y')}")
        
        def get_top3_combined(probs):
            return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:3]
            
        probs_top = (preds["H"]["Probs_For_Graph"] + preds["T"]["Probs_For_Graph"] + preds["O"]["Probs_For_Graph"]) / 3
        probs_bot = (preds["T2"]["Probs_For_Graph"] + preds["O2"]["Probs_For_Graph"]) / 2
        
        top_comb = get_top3_combined(probs_top)
        bot_comb = get_top3_combined(probs_bot)

        # การ์ดแสดงผลสรุปรวม
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown('<div class="number-box">🚀 <b>เด่นบนรวม (ร้อย-สิบ-หน่วย)</b><br>'
                        f'<span class="number-highlight">{top_comb[0][0]} - {top_comb[1][0]} - {top_comb[2][0]}</span></div>', 
                        unsafe_allow_html=True)
        with cc2:
            st.markdown('<div class="number-box">⬇️ <b>เด่นล่างรวม (สิบ-หน่วย)</b><br>'
                        f'<span class="number-highlight">{bot_comb[0][0]} - {bot_comb[1][0]} - {bot_comb[2][0]}</span></div>', 
                        unsafe_allow_html=True)
        st.divider()

        # ----------------------------------------------------
        # ส่วนที่ 2: เจาะลึกรายหลัก
        # ----------------------------------------------------
        st.subheader("📍 เจาะลึกความแม่นแต่ละหลัก (Top 3)")
        
        for pos in ["H", "T", "O", "T2", "O2"]:
            final_nums = [x[0] for x in preds[pos]["Final"]]
            ai_nums = [x[0] for x in preds[pos]["AI"]]
            stat_nums = [x[0] for x in preds[pos]["Stat"]]
            
            with st.expander(f"👉 {labels[pos]} | ชี้เป้า: {final_nums[0]}, {final_nums[1]}, {final_nums[2]}", expanded=False):
                st.caption(preds[pos]["BT_Msg"])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("🥇 เต็ง 1", final_nums[0], f"{preds[pos]['Final'][0][1]*100:.1f}%")
                c2.metric("🥈 เต็ง 2", final_nums[1], f"{preds[pos]['Final'][1][1]*100:.1f}%")
                c3.metric("🥉 เต็ง 3", final_nums[2], f"{preds[pos]['Final'][2][1]*100:.1f}%")
                
                st.markdown(f"- 🤖 **คัดโดย AI เพียวๆ:** `{ai_nums[0]}, {ai_nums[1]}, {ai_nums[2]}`")
                st.markdown(f"- 📊 **คัดจากสถิติหวย:** `{stat_nums[0]}, {stat_nums[1]}, {stat_nums[2]}`")

        # ----------------------------------------------------
        # ส่วนที่ 3: กราฟความน่าจะเป็น
        # ----------------------------------------------------
        st.divider()
        st.subheader("📊 กราฟความน่าจะเป็น (3 ตัวเต็ง)")
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes = axes.flatten()

        for idx, pos in enumerate(["H", "T", "O", "T2", "O2"]):
            ax = axes[idx]
            top_3 = preds[pos]["Final"]
            numbers = [str(x[0]) for x in top_3]
            probabilities = [x[1] * 100 for x in top_3]
            
            bars = ax.bar(numbers, probabilities, color=['#FF4B4B', '#FF7F7F', '#FFBABA'])
            ax.set_title(labels[pos], fontweight='bold')
            ax.set_ylim(0, max(probabilities) * 1.3 if probabilities else 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # ใส่ตัวเลขบนแท่งกราฟ
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)

        fig.delaxes(axes[5])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
