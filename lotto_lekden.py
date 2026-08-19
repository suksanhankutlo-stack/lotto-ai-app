# ============================================================
# 🚀 LOTTO AI ULTIMATE V.MAX 5-TOP (HIGH ACCURACY & SUPER CLEAR UI)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import warnings

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Lotto AI V.MAX 5-TOP",
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
# 2. UI STYLE (ปรับตัวเลขให้แจ่มชัด)
# ============================================================
st.markdown("""
<style>
.main-title { text-align:center; font-size:28px; font-weight:900; color: #D32F2F; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); }
.sub-title { text-align:center; color:#555; font-size:14px; margin-bottom: 25px; }

/* กล่องหลัก */
.hot-card {
    padding: 18px;
    border-radius: 16px;
    border: 2px solid #ff4b4b;
    margin: 10px 0;
    background: linear-gradient(to bottom right, #ffffff, #fff5f5);
    box-shadow: 0 4px 6px rgba(255, 75, 75, 0.1);
}

/* ตัวเลข 5-TOP ให้ใหญ่และเด่นสุดๆ */
.number-highlight {
    font-size: 36px;
    font-weight: 900;
    color: #D32F2F;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
    letter-spacing: 2px;
}
.dot-sep {
    color: #FFCDD2;
    font-size: 26px;
    margin: 0 10px;
}

/* ป้ายกำกับ 3-TOP (Badges) */
.badge-ai { background: #E3F2FD; color: #1565C0; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 16px; border: 1px solid #BBDEFB;}
.badge-stat { background: #E8F5E9; color: #2E7D32; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 16px; border: 1px solid #C8E6C9;}
.badge-cal { background: #FFF3E0; color: #E65100; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 16px; border: 1px solid #FFE0B2;}

.position-title { font-size:20px; font-weight:800; margin-top: 20px; color: #333; border-bottom: 2px solid #eee; padding-bottom: 5px;}
.info-row { margin: 8px 0; font-size: 15px; display: flex; align-items: center;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. FETCH DATA
# ============================================================
@st.cache_data(ttl=180, show_spinner=False)
def fetch_and_clean_data(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
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
            if nm.group(1): r3, r2 = nm.group(1), nm.group(2)
            elif nm.group(3): r3, r2 = nm.group(3)[-3:], nm.group(4)
            else: continue
            
            rows.append({"Date": current_date, "Result_3D": str(r3).zfill(3), "Result_2D": str(r2).zfill(2)})
            
        if len(rows) < 10: raise ValueError("ข้อมูลน้อยเกินไป")
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna().drop_duplicates().sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ ดึงข้อมูลไม่ได้: {e}")
        return pd.DataFrame()

# ============================================================
# 4. TUNED FEATURE ENGINEERING
# ============================================================
def build_features(df, lags, rolls):
    x = df.copy()
    r3, r2 = x["Result_3D"].astype(str), x["Result_2D"].astype(str)
    x["H"], x["T"], x["O"] = r3.str[0].astype(np.int8), r3.str[1].astype(np.int8), r3.str[2].astype(np.int8)
    x["T2"], x["O2"] = r2.str[0].astype(np.int8), r2.str[1].astype(np.int8)

    x["DOW"] = x["Date"].dt.dayofweek.astype(np.int8)
    x["Month"] = x["Date"].dt.month.astype(np.int8)
    x["Day"] = x["Date"].dt.day.astype(np.int8)
    x["Gap"] = x["Date"].diff().dt.days.fillna(7).clip(0, 60).astype(np.int16)
    x["DOW_SIN"], x["DOW_COS"] = np.sin(2*np.pi*x["DOW"]/7), np.cos(2*np.pi*x["DOW"]/7)
    x["MONTH_SIN"], x["MONTH_COS"] = np.sin(2*np.pi*x["Month"]/12), np.cos(2*np.pi*x["Month"]/12)

    ph, pt, po = x["H"].shift(1), x["T"].shift(1), x["O"].shift(1)
    x["PrevSum"] = ph + pt + po
    x["PrevOdd"] = (ph % 2) + (pt % 2) + (po % 2)
    x["DistHT"], x["DistTO"] = (ph - pt).abs(), (pt - po).abs()

    for pos in ["H", "T", "O", "T2", "O2"]:
        s = x[pos]
        prev = s.shift(1)
        x[f"Odd_{pos}"] = (prev % 2)
        x[f"High_{pos}"] = (prev >= 5).astype(np.int8)
        x[f"Prime_{pos}"] = (prev.isin([2, 3, 5, 7])).astype(np.int8)
        
        for lag in lags: x[f"L{lag}_{pos}"] = s.shift(lag)
        for w in rolls: 
            x[f"RM{w}_{pos}"] = s.shift(1).rolling(w, min_periods=1).mean()

        arr = s.to_numpy()
        skip = np.zeros(len(arr), dtype=np.float32)
        last = np.full(10, -1, dtype=np.int32)
        for i, val in enumerate(arr):
            v = int(val)
            skip[i] = i if last[v] < 0 else i - last[v]
            last[v] = i
        x[f"Skip_{pos}"] = skip

    return x.replace([np.inf, -np.inf], np.nan).fillna(-1)

# ============================================================
# 5-9. STATISTIC ENGINES 
# ============================================================
class FrequencyEngine:
    def analyze(self, df, pos):
        s = df[pos].astype(int)
        if len(s) == 0: return np.ones(10) / 10
        r15 = s.tail(15).value_counts(normalize=True) 
        r30 = s.tail(30).value_counts(normalize=True)
        all_f = s.value_counts(normalize=True)
        score = np.array([r15.get(d, 0)*0.55 + r30.get(d, 0)*0.30 + all_f.get(d, 0)*0.15 for d in range(10)])
        score += 0.01
        return score / score.sum()

class CalendarEngine:
    def analyze(self, df, pos, next_date):
        subset = df[df["DOW"] == next_date.dayofweek]
        
        # กฎใหม่: ถ้าข้อมูลวันนั้นๆ มีไม่ถึง 25 งวด "ไม่ต้องคำนวณ"
        if len(subset) < 25:
            # คืนค่าความน่าจะเป็นแบบเป็นกลาง (ให้ทุกเลข 10% เท่ากัน)
            # เพื่อให้ไม่มีเลขไหนได้เปรียบเสียเปรียบ
            return np.ones(10) / 10
            
        # ถ้าข้อมูลถึง 25 งวด ค่อยคำนวณตามปกติ
        a = subset[pos].value_counts(normalize=True)
        b = subset.tail(25)[pos].value_counts(normalize=True)
        score = np.array([a.get(d, 0)*0.3 + b.get(d, 0)*0.7 for d in range(10)])
        score += 0.01
        return score / score.sum()

class TransitionEngine:
    def analyze(self, df, pos):
        if len(df) < 6: return np.ones(10)/10
        subset = df[df[pos].shift(1) == int(df[pos].iloc[-1])]
        if len(subset) < 2: return np.ones(10)/10
        freq = subset[pos].value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)])
        return (score + 0.01) / (score + 0.01).sum()

class PatternEngine:
    def analyze(self, df, pos):
        if len(df) < 7: return np.ones(10)/10
        a, b = int(df[pos].iloc[-1]), int(df[pos].iloc[-2])
        subset = df[(df[pos].shift(1) == a) & (df[pos].shift(2) == b)]
        if len(subset) < 2: subset = df[df[pos].shift(1) == a]
        if len(subset) < 1: return np.ones(10)/10
        freq = subset[pos].value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)])
        return (score + 0.01) / (score + 0.01).sum()

class EquationEngine:
    def analyze(self, df):
        row = df.iloc[-1]
        h, t, o = int(row["H"]), int(row["T"]), int(row["O"])
        vals = [(h+t)%10, (t+o)%10, abs(h-o)%10, (h*t)%10, (h+t+o)%10, (h*2+o)%10]
        score = np.ones(10) * 0.05
        for v in vals: score[v] += 1
        return score / score.sum()

# ============================================================
# 10. TUNED AI MODEL
# ============================================================
class FastAI:
    def __init__(self, trees, weights):
        self.trees = trees
        self.weights = weights

    def predict(self, X, y, X_next):
        rf_w, et_w, hgb_w = self.weights
        result = np.zeros(10)
        total_w = 0

        if rf_w > 0:
            model = RandomForestClassifier(n_estimators=self.trees, max_depth=6, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
            model.fit(X, y)
            for c, p in zip(model.classes_, model.predict_proba(X_next)[0]): result[int(c)] += p * rf_w
            total_w += rf_w

        if et_w > 0:
            model = ExtraTreesClassifier(n_estimators=self.trees, max_depth=6, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=43)
            model.fit(X, y)
            for c, p in zip(model.classes_, model.predict_proba(X_next)[0]): result[int(c)] += p * et_w
            total_w += et_w

        if hgb_w > 0:
            model = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=3, l2_regularization=0.5, random_state=44)
            model.fit(X, y)
            for c, p in zip(model.classes_, model.predict_proba(X_next)[0]): result[int(c)] += p * hgb_w
            total_w += hgb_w

        if total_w <= 0: return np.ones(10)/10
        result /= total_w
        return result / result.sum()

# ============================================================
# 11. ENSEMBLE ENGINE
# ============================================================
class EnsembleEngine:
    def __init__(self, df, lottery_name, target_dow=None):
        self.df, self.target_dow = df.copy(), target_dow
        n = len(df)

        self.trees = 55
        self.lags = [1, 2, 3, 5]
        self.rolls = [3, 5, 10]
        self.mode = "V.MAX TUNED"
        
        if n >= 700: self.bt = 10
        elif n >= 400: self.bt = 9
        else: self.bt = 8

        self.features = ["DOW", "Month", "Gap", "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS", "PrevSum", "PrevOdd", "DistHT", "DistTO"]
        for pos in ["H", "T", "O", "T2", "O2"]:
            self.features.extend([f"Odd_{pos}", f"High_{pos}", f"Prime_{pos}", f"Skip_{pos}"])
            for lag in self.lags: self.features.append(f"L{lag}_{pos}")
            for w in self.rolls: 
                self.features.append(f"RM{w}_{pos}")

        self.freq, self.calendar, self.transition, self.pattern, self.equation = FrequencyEngine(), CalendarEngine(), TransitionEngine(), PatternEngine(), EquationEngine()
        self.ai = FastAI(self.trees, (0.35, 0.35, 0.30))
        self.base_weights = {"AI": 0.50, "Freq": 0.15, "ST": 0.12, "Cal": 0.12, "BT": 0.08, "Eq": 0.03} 

    def backtest(self, pos, X, df_hist):
        n = len(X)
        if n < 45: return self.base_weights.copy(), "Backtest ข้อมูลน้อย"
        start = max(35, n - self.bt)
        scores = {"AI": 0.0, "Freq": 0.0, "ST": 0.0, "Cal": 0.0, "BT": 0.0}
        total_decay = 0.0

        for step, idx in enumerate(range(start, n)):
            decay = 1.08 ** step
            total_decay += decay
            Xtr, ytr, xt, actual = X.iloc[:idx], df_hist[pos].iloc[:idx], X.iloc[[idx]], int(df_hist[pos].iloc[idx])

            try:
                proxy = ExtraTreesClassifier(
                    n_estimators=10,
                    max_depth=5,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    random_state=200+step
                )
                proxy.fit(Xtr, ytr)
                tmp = np.zeros(10)
                for c, p in zip(proxy.classes_, proxy.predict_proba(xt)[0]): tmp[int(c)] = p
                if actual in np.argsort(tmp)[::-1][:5]: scores["AI"] += decay
            except: pass

            hist, target_date = df_hist.iloc[:idx].copy(), df_hist["Date"].iloc[idx]
            f, c, s, b = self.freq.analyze(hist, pos), self.calendar.analyze(hist, pos, target_date), self.transition.analyze(hist, pos), self.pattern.analyze(hist, pos)
            
            if actual in np.argsort(f)[::-1][:5]: scores["Freq"] += decay
            if actual in np.argsort(c)[::-1][:5]: scores["Cal"] += decay
            if actual in np.argsort(s)[::-1][:5]: scores["ST"] += decay
            if actual in np.argsort(b)[::-1][:5]: scores["BT"] += decay

        if total_decay <= 0: return self.base_weights.copy(), "Backtest error"
        accuracy = {k: v / total_decay for k, v in scores.items()}

        weighted = {}
        for k in accuracy: 
            weighted[k] = self.base_weights[k] * (0.35 + 0.65 * max(0.10, accuracy[k]))
        weighted["Eq"] = self.base_weights["Eq"] * 0.35
        
        total = sum(weighted.values())
        weights_pct = {k: v / total for k, v in weighted.items()}

        if weights_pct["AI"] > 0.58:
            diff = weights_pct["AI"] - 0.58
            weights_pct["AI"] = 0.58
            other_sum = sum(v for k, v in weights_pct.items() if k != "AI")
            if other_sum > 0:
                for k in weights_pct:
                    if k != "AI":
                        weights_pct[k] += diff * (weights_pct[k] / other_sum)

        return weights_pct, f"WF HitRate {self.bt} งวด | AI {accuracy['AI']:.0%} | Freq {accuracy['Freq']:.0%} | Cal {accuracy['Cal']:.0%}"

    def process_position(self, pos, hist, X, X_next, next_date):
        weights, bt_msg = self.backtest(pos, X, hist)
        ai, fq, cal = self.ai.predict(X, hist[pos], X_next), self.freq.analyze(hist, pos), self.calendar.analyze(hist, pos, next_date)
        stp, ptn, eq = self.transition.analyze(hist, pos), self.pattern.analyze(hist, pos), self.equation.analyze(hist)

        final = (weights["AI"]*ai + weights["Freq"]*fq + weights["Cal"]*cal + weights["ST"]*stp + weights["BT"]*ptn + weights["Eq"]*eq)
        final /= final.sum()

        top_n = lambda p, n: [(int(i), float(p[i])) for i in np.argsort(p)[::-1][:n]]
        return {"Final": top_n(final, 5), "AI": top_n(ai, 3), "Freq": top_n(fq, 3), "Calendar": top_n(cal, 3), "Prob": final, "Weights": weights, "BT": bt_msg}

    def predict_all(self):
        last_date = self.df["Date"].iloc[-1]
        days = (self.target_dow - last_date.dayofweek) % 7 if self.target_dow is not None else max(1, (last_date - self.df["Date"].iloc[-2]).days) if len(self.df)>=2 else 7
        next_date = last_date + timedelta(days=days if days > 0 else 7)
        
        ext = pd.concat([self.df, pd.DataFrame([{"Date": next_date, "Result_3D": "000", "Result_2D": "00"}])], ignore_index=True)
        ext = build_features(ext, self.lags, self.rolls)
        hist, X = ext.iloc[:-1].copy(), ext.iloc[:-1][self.features].astype(np.float32)
        X_next = ext.iloc[[-1]][self.features].astype(np.float32)

        return {pos: self.process_position(pos, hist, X, X_next, next_date) for pos in ["H", "T", "O", "T2", "O2"]}, next_date

# ============================================================
# 12. UI FORMATTING HELPERS 
# ============================================================
def html_top5(items):
    parts = [f'<span class="number-highlight">{n}</span>' for n, p in items]
    return f'<span class="dot-sep">•</span>'.join(parts)

def html_badge(items, badge_class):
    parts = [str(n) for n, p in items]
    return f'<span class="{badge_class}">{" &nbsp;•&nbsp; ".join(parts)}</span>'

def nums_prob(items): 
    return " | ".join(f"{n} ({p:.1%})" for n, p in items)

def combine_top_n(preds, positions, n=5):
    score = sum([preds[pos]["Prob"] for pos in positions]) / len(positions)
    return [(int(i), float(score[i])) for i in np.argsort(score)[::-1][:n]]

# ============================================================
# 13. UI HEADER
# ============================================================
st.markdown('<div class="main-title"> LOTTO AI V.MAX </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">High Accuracy Tuned • Advanced Feature Engineering<br><b>สรุปเด่น 5-TOP | AI & สถิติ 3-TOP (UI แจ่มชัด)</b></div>', unsafe_allow_html=True)
st.divider()

# ============================================================
# 14. APP RUN
# ============================================================
c1, c2 = st.columns(2)
selected_lotto = c1.selectbox("🎯 เลือกหวย", list(LOTTERY_SOURCES.keys()))
day_options = {"อัตโนมัติ": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6}
day_label = c2.selectbox("📅 วันออกรางวัล", list(day_options.keys()))

if st.button("🚀 วิเคราะห์เลขเด่นด้วย AI (Turbo)", type="primary", use_container_width=True):
    with st.spinner("⚡ AI กำลังดึงข้อมูลและประมวลผลเชิงลึก..."):
        df = fetch_and_clean_data(LOTTERY_SOURCES[selected_lotto])
        if df.empty: st.stop()

        engine = EnsembleEngine(df, selected_lotto, day_options[day_label])
        preds, next_date = engine.predict_all()
        labels = {"H":"หลักร้อย 3 ตัวบน", "T":"หลักสิบ 3 ตัวบน", "O":"หลักหน่วย 3 ตัวบน", "T2":"หลักสิบ 2 ตัวล่าง", "O2":"หลักหน่วย 2 ตัวล่าง"}
        days = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

        st.divider()
        st.info(f"📅 วิเคราะห์งวดเป้าหมาย: วัน{days[next_date.dayofweek]} {next_date.strftime('%d-%m-%Y')} (จากข้อมูล {len(df)} งวด)")

        for pos in ["H", "T", "O", "T2", "O2"]:
            res = preds[pos]
            st.markdown(f'<div class="position-title">📍 {labels[pos]}</div>', unsafe_allow_html=True)
            
            # การ์ด HOT TOP-5 
            st.markdown(f'''
                <div class="hot-card">
                    <div style="font-weight:700; color:#444; margin-bottom:8px;">🔥 HOT TOP-5 (สรุปเด่นหลัก)</div>
                    <div style="text-align:center; margin: 10px 0;">{html_top5(res["Final"])}</div>
                    <div style="font-size:13px; color:#888; text-align:center; margin-top:8px;">{nums_prob(res["Final"])}</div>
                </div>
            ''', unsafe_allow_html=True)

            # ป้ายกำกับ 3-TOP 
            st.markdown(f'<div class="info-row">🤖 <b>AI ท๊อป 3:</b> &nbsp; {html_badge(res["AI"], "badge-ai")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-row">📊 <b>สถิติ ท๊อป 3:</b> &nbsp; {html_badge(res["Freq"], "badge-stat")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-row">📅 <b>กำลังวัน ท๊อป 3:</b> &nbsp; {html_badge(res["Calendar"], "badge-cal")}</div>', unsafe_allow_html=True)
            
            # ข้อมูลประกอบ
            st.markdown(f'<div style="font-size:13px; color:#999; margin-top:10px;">📈 {res["BT"]}</div>', unsafe_allow_html=True)
            W = res["Weights"]
            st.markdown(f'<div style="font-size:13px; color:#999;">⚖️ น้ำหนัก: AI {W["AI"]:.0%} | สถิติ {W["Freq"]:.0%} | วัน {W["Cal"]:.0%} | ก้าวเดิน {W["ST"]:.0%} | แพทเทิร์น {W["BT"]:.0%}</div>', unsafe_allow_html=True)
            st.write("")

        # ภาพรวม
        hot_top, hot_bot = combine_top_n(preds, ["H","T","O"]), combine_top_n(preds, ["T2","O2"])
        st.subheader("🔥 สรุปเลขเด่นภาพรวม (บน-ล่าง)")
        st.markdown(f'<div class="hot-card"><div style="font-weight:700; color:#444;">🔥 HOT 5-TOP รูด/วิ่ง (บน)</div><div style="text-align:center; margin:10px 0;">{html_top5(hot_top)}</div><div style="font-size:13px; color:#888; text-align:center;">{nums_prob(hot_top)}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hot-card"><div style="font-weight:700; color:#444;">🔥 HOT 5-TOP รูด/วิ่ง (ล่าง)</div><div style="text-align:center; margin:10px 0;">{html_top5(hot_bot)}</div><div style="font-size:13px; color:#888; text-align:center;">{nums_prob(hot_bot)}</div></div>', unsafe_allow_html=True)

        st.success("✅ วิเคราะห์เสร็จสิ้น • อัปเกรดความแม่นยำและ UI เรียบร้อยแล้ว")
