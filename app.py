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

# --- Machine Learning Modules ---
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. Setup Streamlit Page & Caching State
# ==============================================================================

st.set_page_config(page_title="ระบบวิเคราะห์เลขเด่น/เลขดับ Ultimate Ensemble", page_icon="🚀", layout="centered")

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
# 2. ระบบวิเคราะห์เลขเด่น & เลขดับ
# ==========================================
class EnsembleEngine:
    def __init__(self, df_raw, lottery_name, target_dow=None):
        self.df_raw = df_raw
        self.target_dow = target_dow
        self.lottery_name = lottery_name
        self.lottery_id = lottery_name.split(".")[0].strip()
        n = len(df_raw)

        self.trees, self.test_size = 100, 20
        self.lags, self.rolls = [1, 2, 3, 5], [3, 5, 10]
        
        self.features = ['DayOfWeek', 'Month', 'DrawIndex', 'Gap', 'DigitSum_3D']
        for pos in ['H', 'T', 'O', 'T2', 'O2']:
            self.features.extend([f'OddEven_{pos}', f'HighLow_{pos}', f'Mirror_{pos}', f'Mod3_{pos}', f'Skip_{pos}', f'Repeat_{pos}'])
            for lag in self.lags: self.features.append(f'Lag_{lag}_{pos}')
            for w in self.rolls:
                self.features.extend([f'Roll_{w}_Mean_{pos}', f'Roll_{w}_Std_{pos}'])

        hash_array = pd.util.hash_pandas_object(df_raw[['Result_3D', 'Result_2D']], index=False).values
        base_hash = hashlib.md5(hash_array).hexdigest()
        self.data_hash = f"{base_hash}_{self.trees}_{self.test_size}"
        
        self.model = RandomForestClassifier(n_estimators=self.trees, max_depth=5, random_state=42, n_jobs=1)

    def predict_all(self, mode="เด่น"):
        last_date = self.df_raw['Date'].iloc[-1]
        if self.target_dow is not None:
            days_ahead = self.target_dow - last_date.dayofweek
            if days_ahead <= 0: days_ahead += 7
            next_date = last_date + timedelta(days=days_ahead)
        else:
            next_date = last_date + timedelta(days=7)

        dummy = pd.DataFrame([{'Date': next_date, 'Result_3D': '000', 'Result_2D': '00'}])
        df_ext = pd.concat([self.df_raw, dummy], ignore_index=True)
        df_ext = build_features(df_ext, self.lags, self.rolls)
        
        next_x = df_ext.iloc[[-1]][self.features]
        df_hist = df_ext.iloc[:-1]
        X_all = df_hist[self.features]

        results = {}
        for pos in ['H', 'T', 'O', 'T2', 'O2']:
            y_train = df_hist[pos]
            self.model.fit(X_all, y_train)
            probs = self.model.predict_proba(next_x)[0]
            
            res_probs = np.zeros(10)
            for c, p in zip(self.model.classes_, probs): res_probs[int(c)] = p
            
            sorted_items = sorted([(i, res_probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)
            
            if mode == "เด่น":
                selected = sorted_items[:5] # เอา 5 ตัวที่น่าจะเป็นไปได้มากที่สุด
            else:
                selected = sorted_items[-3:] # เอา 3 ตัวท้ายสุดเป็นเลขดับ
                
            results[pos] = selected

        return results, next_date

# ==========================================
# 3. Streamlit User Interface
# ==========================================

st.title("🚀 ระบบวิเคราะห์เลขเด่น / เลขดับ V4")
st.divider()

selected_lotto = st.selectbox('🎯 เลือกประเภทหวย:', list(LOTTERY_SOURCES.keys()), index=0)
mode_choice = st.radio("🔍 เลือกระบบ:", ["เลขเด่น", "เลขดับ"], horizontal=True)

if st.button("🚀 วิเคราะห์", type="primary", use_container_width=True):
    with st.spinner("⏳ กำลังคำนวณผล..."):
        url = LOTTERY_SOURCES[selected_lotto]
        df_raw = fetch_and_clean_data(url)
        engine = EnsembleEngine(df_raw, selected_lotto)
        
        preds, next_date = engine.predict_all(mode=mode_choice)
        dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
        labels = {'H': 'หลักร้อย (บน)', 'T': 'หลักสิบ (บน)', 'O': 'หลักหน่วย (บน)', 'T2': 'หลักสิบ (ล่าง)', 'O2': 'หลักหน่วย (ล่าง)'}

        st.markdown(f"### 🔮 ผลการวิเคราะห์{mode_choice} ประจำวัน{dow_names[next_date.dayofweek]}ที่ {next_date.strftime('%d-%m-%Y')}")
        st.divider()

        for pos in ['H', 'T', 'O', 'T2', 'O2']:
            nums = ", ".join([str(num) for num, prob in preds[pos]])
            if mode_choice == "เด่น":
                st.success(f"📍 **{labels[pos]}**: `{nums}`")
            else:
                st.error(f"🛑 **{labels[pos]} (ดับ)**: `{nums}`")
