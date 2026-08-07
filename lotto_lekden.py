import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from joblib import Memory
import hashlib
import os
import glob
import copy

# --- Machine Learning Modules ---
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. Setup Streamlit Page & Caching State
# ==============================================================================

st.set_page_config(page_title="ระบบวิเคราะห์เลขเด่น Ultimate Ensemble", page_icon="🚀", layout="centered")

os.makedirs('model_cache', exist_ok=True)
os.makedirs('/tmp/lotto_memory_cache', exist_ok=True)
memory = Memory(location='/tmp/lotto_memory_cache', verbose=0)

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

# ==========================================
# 1. ระบบจัดการข้อมูล & Feature Engineering
# ==========================================
@memory.cache
def fetch_and_clean_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('div', class_=re.compile(r'post-body|entry-content|post-content|content'))
        if not main_content: main_content = soup

        text_lines = main_content.get_text(separator='\n').split('\n')
        extracted = []
        date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})')
        num_pattern = re.compile(r'\b(\d{3})\b.*?\b(\d{2})\b|\b(\d{5,6})\b.*?\b(\d{2})\b')
        current_date = datetime.now().strftime('%Y-%m-%d')

        for line in text_lines:
            line = line.strip()
            if not line: continue

            date_match = date_pattern.search(line)
            if date_match:
                current_date = date_match.group(1).replace('/', '-')

            num_match = num_pattern.search(line)
            if num_match:
                if num_match.group(1) and num_match.group(2):
                    res3d, res2d = num_match.group(1), num_match.group(2)
                elif num_match.group(3) and num_match.group(4):
                    res3d, res2d = num_match.group(3)[-3:], num_match.group(4)
                else:
                    continue
                extracted.append({'Date': current_date, 'Result_3D': res3d, 'Result_2D': res2d})

        if len(extracted) < 10: raise Exception("ข้อมูลน้อยเกินไป")
        df = pd.DataFrame(extracted)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.dropna().sort_values('Date').reset_index(drop=True)
    except Exception:
        dates = pd.date_range(end=datetime.now(), periods=200, freq='W')
        df = pd.DataFrame({'Date': dates, 'Result_3D': np.random.randint(0, 1000, 200), 'Result_2D': np.random.randint(0, 100, 200)})
        df['Result_3D'] = df['Result_3D'].apply(lambda x: f"{x:03d}")
        df['Result_2D'] = df['Result_2D'].apply(lambda x: f"{x:02d}")
        return df.sort_values('Date').reset_index(drop=True)

@memory.cache
def build_features(df, lags, rolls):
    df_feat = df.copy()
    df_feat['H'] = df_feat['Result_3D'].str[0].astype(int)
    df_feat['T'] = df_feat['Result_3D'].str[1].astype(int)
    df_feat['O'] = df_feat['Result_3D'].str[2].astype(int)
    df_feat['T2'] = df_feat['Result_2D'].str[0].astype(int)
    df_feat['O2'] = df_feat['Result_2D'].str[1].astype(int)

    df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['DrawIndex'] = df_feat.index
    df_feat['Gap'] = df_feat['Date'].diff().dt.days.fillna(7).astype(int)

    df_feat['DigitSum_3D'] = (df_feat['H'].shift(1) + df_feat['T'].shift(1) + df_feat['O'].shift(1)).fillna(0) % 10

    for pos in ['H', 'T', 'O', 'T2', 'O2']:
        prev = df_feat[pos].shift(1)

        df_feat[f'OddEven_{pos}'] = (prev % 2).fillna(0).astype(int)
        df_feat[f'HighLow_{pos}'] = (prev >= 5).fillna(0).astype(int)
        df_feat[f'Mirror_{pos}'] = (prev + 5).fillna(0) % 10
        df_feat[f'Mod3_{pos}'] = (prev % 3).fillna(0).astype(int)

        for lag in lags:
            df_feat[f'Lag_{lag}_{pos}'] = df_feat[pos].shift(lag)

        for w in rolls:
            df_feat[f'Roll_{w}_Mean_{pos}'] = df_feat[pos].shift(1).rolling(w).mean()
            df_feat[f'Roll_{w}_Std_{pos}'] = df_feat[pos].shift(1).rolling(w).std()

        if f'Lag_1_{pos}' in df_feat.columns and f'Lag_2_{pos}' in df_feat.columns:
            df_feat[f'Repeat_{pos}'] = (df_feat[f'Lag_1_{pos}'] == df_feat[f'Lag_2_{pos}']).astype(int)

        for d in range(10):
            df_feat[f'Hot20_{pos}_{d}'] = (df_feat[pos].shift(1) == d).rolling(20).sum()

        last_seen = {}
        skips = np.zeros(len(df_feat))
        pos_values = df_feat[pos].values
        for i in range(len(df_feat)):
            if i == 0:
                skips[i] = 100
            else:
                curr_val = pos_values[i-1]
                skips[i] = i - last_seen.get(curr_val, 0)
            last_seen[pos_values[i]] = i
        df_feat[f'Skip_{pos}'] = skips

    return df_feat.fillna(-1)

# ==========================================
# 2. ระบบวิเคราะห์ 5 สำนัก
# ==========================================
class PositionalEquation:
    def analyze(self, df):
        latest = df.iloc[-1]
        H, T, O = latest['H'], latest['T'], latest['O']
        probs = np.zeros(10)
        for v in [(H + T) % 10, (T + O) % 10, abs(H - O) % 10, (H * T) % 10]: probs[int(v)] += 1.0
        return (probs + 0.1) / (probs + 0.1).sum()

class FrequencyEngine:
    def analyze(self, df, pos):
        series = df[pos].dropna()
        probs = np.zeros(10)
        freq_all = series.value_counts(normalize=True).to_dict()
        freq_10 = series.tail(10).value_counts(normalize=True).to_dict()
        for i in range(10):
            idxs = np.where(series == i)[0]
            skip = (len(series) - 1 - idxs[-1]) if len(idxs) > 0 else len(series)
            probs[i] = (freq_all.get(i, 0) * 0.4) + (freq_10.get(i, 0) * 0.4) + ((1.0 / (skip + 1)) * 0.2)
        return (probs + 0.01) / (probs + 0.01).sum()

class ConditionalSystem:
    def analyze(self, df, pos, next_date):
        probs = np.zeros(10)
        subset = df[(df['DayOfWeek'] == next_date.dayofweek)]
        if len(subset) == 0: subset = df
        freq = subset[pos].value_counts(normalize=True).to_dict()
        for i in range(10): probs[i] = freq.get(i, 0)
        return (probs + 0.01) / (probs + 0.01).sum()

class StateTransitionSystem:
    def analyze(self, df, pos):
        probs = np.zeros(10)
        if len(df) < 2: return np.ones(10) / 10
        subset = df[df[f'Lag_1_{pos}'] == df[pos].iloc[-1]]
        if len(subset) > 0:
            freq = subset[pos].value_counts(normalize=True).to_dict()
            for i in range(10): probs[i] = freq.get(i, 0)
        return (probs + 0.01) / (probs + 0.01).sum()

class PatternBacktestSystem:
    def analyze(self, df, pos):
        probs = np.zeros(10)
        if len(df) < 3: return np.ones(10) / 10
        l1, l2 = df[pos].iloc[-1], df[pos].iloc[-2]
        subset = df[(df[f'Lag_1_{pos}'] == l1) & (df[f'Lag_2_{pos}'] == l2)]
        if len(subset) == 0: subset = df[df[f'Lag_1_{pos}'] == l1]
        if len(subset) > 0:
            freq = subset[pos].value_counts(normalize=True).to_dict()
            for i in range(10): probs[i] = freq.get(i, 0)
        return (probs + 0.01) / (probs + 0.01).sum()

class AISystem:
    def __init__(self, lottery_id, trees, rf_w, et_w, hgb_w, xgb_w):
        self.lottery_id = lottery_id
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=trees, max_depth=5, n_jobs=1, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=trees, max_depth=5, n_jobs=1, random_state=42)),
            ('hgb', HistGradientBoostingClassifier(max_iter=50, random_state=42))
        ]
        weights = [rf_w, et_w, hgb_w]
        if xgb_w > 0:
            estimators.append(('xgb', XGBClassifier(n_estimators=50, max_depth=3, tree_method="hist", verbosity=0, random_state=42, n_jobs=1)))
            weights.append(xgb_w)
        self.voting = VotingClassifier(estimators=estimators, voting='soft', weights=weights)

    def analyze(self, X_train, y_train, X_next, pos, data_hash):
        prefix_path = f"model_cache/m_{self.lottery_id}_{pos}_"
        model_path = f"{prefix_path}{data_hash}.joblib"

        if not os.path.exists(model_path):
            for old_file in glob.glob(f"{prefix_path}*.joblib"):
                try: os.remove(old_file)
                except: pass
            model = self.voting
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
        else:
            model = joblib.load(model_path)

        probs = model.predict_proba(X_next)[0]
        res = np.zeros(10)
        for c, p in zip(model.classes_, probs): res[int(c)] = p
        return res / res.sum()

class EnsembleEngine:
    def __init__(self, df_raw, lottery_name, target_dow=None):
        self.df_raw = df_raw
        self.target_dow = target_dow
        self.lottery_name = lottery_name
        self.lottery_id = lottery_name.split(".")[0].strip()
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
            self.ai_weights = (1.0, 0.8, 0.5, 0.10)

        if n < 100: self.test_size = min(5, max(0, n - 30))

        self.features = ['DayOfWeek', 'Month', 'DrawIndex', 'Gap', 'DigitSum_3D']
        for pos in ['H', 'T', 'O', 'T2', 'O2']:
            self.features.extend([f'OddEven_{pos}', f'HighLow_{pos}', f'Mirror_{pos}', f'Mod3_{pos}', f'Skip_{pos}', f'Repeat_{pos}'])
            for lag in self.lags: self.features.append(f'Lag_{lag}_{pos}')
            for w in self.rolls:
                self.features.extend([f'Roll_{w}_Mean_{pos}', f'Roll_{w}_Std_{pos}'])
            for d in range(10): self.features.append(f'Hot20_{pos}_{d}')

        hash_array = pd.util.hash_pandas_object(df_raw[['Result_3D', 'Result_2D']], index=False).values
        base_hash = hashlib.md5(hash_array).hexdigest()
        self.data_hash = f"{base_hash}_{self.trees}_{self.test_size}_{len(self.features)}_{str(self.ai_weights)}"

        self.pos_sys, self.freq_sys = PositionalEquation(), FrequencyEngine()
        self.cond_sys, self.st_sys = ConditionalSystem(), StateTransitionSystem()
        self.ptn_sys = PatternBacktestSystem()
        self.ai_sys = AISystem(self.lottery_id, self.trees, *self.ai_weights)
        self.base_weights = {'AI': 0.35, 'Freq': 0.20, 'ST': 0.15, 'Cal': 0.10, 'BT': 0.10, 'Eq': 0.10}

    def _process_single_position(self, pos, df_hist, X_all, next_x, next_date):
        bt_size = self.test_size
        cache_key = f"model_cache/bt_{self.lottery_id}_{pos}_{self.data_hash}.joblib"

        if os.path.exists(cache_key):
            norm_weights, bt_msg = joblib.load(cache_key)
        elif len(df_hist) < bt_size + 30 or bt_size <= 0:
            norm_weights, bt_msg = self.base_weights, "(ข้อมูลน้อย ข้าม Backtest)"
        else:
            ai_hits, fq_hits, cal_hits, st_hits, ptn_hits = 0, 0, 0, 0, 0
            steps_run = 0
            proxy_model = ExtraTreesClassifier(n_estimators=10, max_depth=3, n_jobs=1, random_state=42)

            for i in range(bt_size):
                curr_train_len = len(X_all) - bt_size + i
                X_train_step = X_all.iloc[:curr_train_len]
                y_train_step = df_hist[pos].iloc[:curr_train_len]
                X_test_step = X_all.iloc[[curr_train_len]]
                actual_val = df_hist[pos].iloc[curr_train_len]

                proxy_model.fit(X_train_step, y_train_step)
                probs = proxy_model.predict_proba(X_test_step)[0]
                ai_res = np.zeros(10)
                for idx, c in enumerate(proxy_model.classes_): ai_res[int(c)] = probs[idx]
                if actual_val in np.argsort(ai_res)[::-1][:5]: ai_hits += 1

                curr_df = df_hist.iloc[:curr_train_len]
                target_date = df_hist.iloc[curr_train_len]['Date']

                if actual_val in np.argsort(self.freq_sys.analyze(curr_df, pos))[::-1][:5]: fq_hits += 1
                if actual_val in np.argsort(self.cond_sys.analyze(curr_df, pos, target_date))[::-1][:5]: cal_hits += 1
                if actual_val in np.argsort(self.st_sys.analyze(curr_df, pos))[::-1][:5]: st_hits += 1
                if actual_val in np.argsort(self.ptn_sys.analyze(curr_df, pos))[::-1][:5]: ptn_hits += 1

                steps_run += 1
                if steps_run >= self.early_stop:
                    break

            w_ai = self.base_weights['AI'] * max(0.1, (ai_hits / steps_run))**2
            w_fq = self.base_weights['Freq'] * max(0.1, (fq_hits / steps_run))**2
            w_cal = self.base_weights['Cal'] * max(0.1, (cal_hits / steps_run))**2
            w_st = self.base_weights['ST'] * max(0.1, (st_hits / steps_run))**2
            w_bt = self.base_weights['BT'] * max(0.1, (ptn_hits / steps_run))**2
            w_eq = self.base_weights['Eq'] * 0.1

            total = w_ai + w_fq + w_cal + w_st + w_bt + w_eq
            norm_weights = {'AI': w_ai/total, 'Freq': w_fq/total, 'Cal': w_cal/total, 'ST': w_st/total, 'BT': w_bt/total, 'Eq': w_eq/total}
            msg = f"(Backtest {steps_run} งวด: AI {int((ai_hits/steps_run)*100)}% | Freq {int((fq_hits/steps_run)*100)}% | ST {int((st_hits/steps_run)*100)}%)"

            joblib.dump((norm_weights, msg), cache_key)
            bt_msg = msg

        p_ai = self.ai_sys.analyze(X_all, df_hist[pos], next_x, pos, self.data_hash)
        p_fq = self.freq_sys.analyze(df_hist, pos)
        p_cal = self.cond_sys.analyze(df_hist, pos, next_date)
        p_st = self.st_sys.analyze(df_hist, pos)
        p_bt = self.ptn_sys.analyze(df_hist, pos)
        p_eq = self.pos_sys.analyze(df_hist)

        W = norm_weights
        final_score = (W['AI']*p_ai + W['Freq']*p_fq + W['Cal']*p_cal + W['ST']*p_st + W['BT']*p_bt + W['Eq']*p_eq)
        final_score = final_score / final_score.sum()

        def get_top5(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]

        return pos, {
            'AI': get_top5(p_ai),
            'Calendar': get_top5(p_cal),
            'Frequency': get_top5(p_fq),
            'Final': get_top5(final_score),
            'Probs_For_Graph': final_score,
            'BT_Msg': bt_msg
        }

    def predict_all(self):
        last_date = self.df_raw['Date'].iloc[-1]
        if self.target_dow is not None:
            days_ahead = self.target_dow - last_date.dayofweek
            if days_ahead <= 0: days_ahead += 7
            next_date = last_date + timedelta(days=days_ahead)
        else:
            next_date = last_date + timedelta(days=7 if len(self.df_raw) <= 1 else (last_date - self.df_raw['Date'].iloc[-2]).days)

        dummy = pd.DataFrame([{'Date': next_date, 'Result_3D': '000', 'Result_2D': '00'}])
        df_ext = pd.concat([self.df_raw, dummy], ignore_index=True)

        df_ext = build_features(df_ext, self.lags, self.rolls)
        next_x = df_ext.iloc[[-1]][self.features]
        df_hist = df_ext.iloc[:-1]
        X_all = df_hist[self.features]

        results = []
        for pos in ['H', 'T', 'O', 'T2', 'O2']:
            res = self._process_single_position(pos, df_hist, X_all, next_x, next_date)
            results.append(res)

        return {pos: data for pos, data in results}, next_date

# ==========================================
# 3. Streamlit User Interface
# ==========================================

st.title("🚀 ระบบวิเคราะห์เลขเด่น Ultimate Ensemble")
st.markdown("**(Super Fast Mobile Edition)**")
st.divider()

col1, col2 = st.columns(2)
with col1:
    selected_lotto = st.selectbox('🎯 เลือกหวย:', list(LOTTERY_SOURCES.keys()), index=0)
with col2:
    day_options = {
        'อัตโนมัติ (คำนวณจากงวดล่าสุด)': None, 'วันจันทร์': 0, 'วันอังคาร': 1,
        'วันพุธ': 2, 'วันพฤหัสบดี': 3, 'วันศุกร์': 4, 'วันเสาร์': 5, 'วันอาทิตย์': 6
    }
    selected_day_label = st.selectbox('📅 ออกวัน:', list(day_options.keys()), index=0)
    target_dow = day_options[selected_day_label]

if st.button("🚀 วิเคราะห์เลขเด่น V.Max (Sequential)", type="primary", use_container_width=True):
    with st.spinner("⏳ กำลังดึงข้อมูลและคำนวณโมเดล AI โปรดรอสักครู่..."):
        url = LOTTERY_SOURCES[selected_lotto]
        df_raw = fetch_and_clean_data(url)
        engine = EnsembleEngine(df_raw, selected_lotto, target_dow=target_dow)

        weights_str = f"RF={engine.ai_weights[0]} | ET={engine.ai_weights[1]} | HGB={engine.ai_weights[2]} | XGB={engine.ai_weights[3]}"

        st.info(f"""
        **⚙️ สเตตัสระบบ [{engine.mode_name}]:**
        - 🌲 Trees = {engine.trees} | 🔄 BT = {engine.test_size} (Stop: {engine.early_stop})
        - 📊 โครงสร้างข้อมูล: Lags {engine.lags} | Rolling {engine.rolls} | ฟีเจอร์ Modulo 3: เปิดใช้งาน
        - 🤖 สัดส่วนโหวต AI (4 สำนัก): {weights_str}
        """)

        preds, next_date = engine.predict_all()
        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
        labels = {'H': 'หลักร้อย (บน)', 'T': 'หลักสิบ (บน)', 'O': 'หลักหน่วย (บน)', 'T2': 'หลักสิบ (ล่าง)', 'O2': 'หลักหน่วย (ล่าง)'}

        st.markdown(f"### 🔮 ผลการวิเคราะห์ ประจำวัน{dow_names[next_date.dayofweek]}ที่ {next_date.strftime('%d-%m-%Y')}")
        st.divider()

        for pos in ['H', 'T', 'O', 'T2', 'O2']:
            nums_ai = ", ".join([str(num) for num, prob in preds[pos]['AI']])
            nums_day = ", ".join([str(num) for num, prob in preds[pos]['Calendar']])
            nums_stat = ", ".join([str(num) for num, prob in preds[pos]['Frequency']])
            nums_final = ", ".join([str(num) for num, prob in preds[pos]['Final']])

            with st.expander(f"📍 ตำแหน่ง: {labels[pos]}", expanded=True):
                st.caption(f"{preds[pos]['BT_Msg']}")
                st.markdown(f"- 🤖 **เลขเด่น AI** : `{nums_ai}`")
                st.markdown(f"- 📅 **เลขเด่น กำลังวัน** : `{nums_day}`")
                st.markdown(f"- 📊 **เลขเด่น สถิติ** : `{nums_stat}`")
                st.success(f"🌟 **เด่นสรุปรวม 5 ตัว**: `{nums_final}`")

        # สรุปฟันธง
        probs_top = (preds['H']['Probs_For_Graph'] + preds['T']['Probs_For_Graph'] + preds['O']['Probs_For_Graph']) / 3
        probs_bot = (preds['T2']['Probs_For_Graph'] + preds['O2']['Probs_For_Graph']) / 2

        def get_top5(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]
        top5_top = get_top5(probs_top)
        top5_bot = get_top5(probs_bot)

        st.divider()
        st.subheader("🔥 สรุปฟันธง เลขเด่นมาแรง")
        st.markdown(f"🚀 **เด่นบนรวม (ร้อย-สิบ-หน่วย)** : `" + " , ".join([str(x[0]) for x in top5_top]) + "`")
        st.markdown(f"⬇️ **เด่นล่างรวม (สิบ-หน่วย)** : `" + " , ".join([str(x[0]) for x in top5_bot]) + "`")

        # กราฟแสดงผล
        st.divider()
        st.subheader("📊 กราฟความน่าจะเป็นแต่ละหลัก")
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        colors_list = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

        for idx, pos in enumerate(['H', 'T', 'O', 'T2', 'O2']):
            ax = axes[idx]
            top_5_items = preds[pos]['Final']
            ax.bar([str(x[0]) for x in top_5_items], [x[1]*100 for x in top_5_items], color=colors_list)
            ax.set_title(labels[pos])
            ax.set_ylabel('โอกาส (%)')

        if len(axes) > 5:
            fig.delaxes(axes[5])

        plt.tight_layout()
        st.pyplot(fig)
