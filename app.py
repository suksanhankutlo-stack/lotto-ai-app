import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
from datetime import datetime, timedelta
import copy
import joblib
import hashlib
import os
import glob
import time
import scipy.stats as stats
import tempfile
import warnings

# --- Machine Learning Modules ---
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. Setup Streamlit, Anti-Translate & Cache
# ==============================================================================
st.set_page_config(page_title="Ultimate Lotto Analyzer", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
        body { translate: no; }
    </style>
    <meta name="google" content="notranslate">
""", unsafe_allow_html=True)

@st.cache_resource
def setup_thai_font():
    font_path = 'thsarabunnew-webfont.ttf'
    if not os.path.exists(font_path):
        try: 
            urllib.request.urlretrieve("https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf", font_path)
        except Exception: 
            pass

    if os.path.exists(font_path):  
        fm.fontManager.addfont(font_path)  
        plt.rc('font', family='TH Sarabun New', size=14)  
    else:  
        plt.rc('font', family='Tahoma', size=12)

setup_thai_font()

CACHE_DIR = os.path.join(tempfile.gettempdir(), "lotto_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_old_cache(directory, days=7):
    if not os.path.exists(directory): return
    now = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if os.stat(filepath).st_mtime < now - days * 86400:
                try: os.remove(filepath)
                except: pass

clean_old_cache(CACHE_DIR, 7)

if 'model_cache' not in st.session_state: st.session_state.model_cache = {}
if 'bt_cache' not in st.session_state: st.session_state.bt_cache = {}

LOTTERY_URLS = {
    '1. หวยไทย': 'https://suksan18190.blogspot.com/2026/07/blog-post_07.html',
    '2. หวยธกส.': 'https://suksan18190.blogspot.com/2026/07/blog-post_12.html',
    '3. หวยออมสิน': 'https://suksan18190.blogspot.com/2026/07/blog-post_525.html',
    '4. หวยลาว': 'https://suksan18190.blogspot.com/2026/07/blog-post.html',
    '5. หวยฮานอย': 'https://suksan18190.blogspot.com/2026/07/blog-post_08.html',
    '6. หวยมาเลย์': 'https://suksan18190.blogspot.com/2026/07/blog-post_10.html',
    '7. หวยหุ้นไทยเย็น': 'https://suksan18190.blogspot.com/2026/07/blog-post_11.html',
    '8. หวยหุ้นนิเคอิบ่าย': 'https://suksan18190.blogspot.com/2026/07/blog-post_412.html',
    '9. หวยหุ้นฮั่งเส็งบ่าย': 'https://suksan18190.blogspot.com/2026/07/blog-post_229.html',
    '10. หวยหุ้นจีนบ่าย': 'https://suksan18190.blogspot.com/2026/07/blog-post_162.html'
}

def calculate_next_draw_date(last_date, lotto_name):
    if any(k in lotto_name for k in ['ไทย', 'ธกส', 'ออมสิน']):
        year, month, day = last_date.year, last_date.month, last_date.day
        if month == 12 and day >= 16 and day < 30: return datetime(year, 12, 30)
        elif month == 12 and day == 30: return datetime(year + 1, 1, 17)
        elif month == 1 and day == 17: return datetime(year, 2, 1)
        elif month == 4 and day >= 16: return datetime(year, 5, 2)
        elif month == 5 and day == 2: return datetime(year, 5, 16)
        
        if day < 15: return datetime(year, month, 16)
        else:
            next_month = 1 if month == 12 else month + 1
            next_year = year + 1 if month == 12 else year
            return datetime(next_year, next_month, 1)
    else:
        return last_date + timedelta(days=1)

# ==============================================================================
# 1. ระบบดึงข้อมูล (แยกสำหรับ เด่น และ ดับ)
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_data_hot(url):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    response = session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    response.raise_for_status()
    try: soup = BeautifulSoup(response.text, 'lxml')  
    except: soup = BeautifulSoup(response.text, 'html.parser')  
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
        if date_match: current_date = date_match.group(1).replace('/', '-')  
        num_match = num_pattern.search(line)  
        if num_match:  
            if num_match.group(1) and num_match.group(2): res3d, res2d = num_match.group(1), num_match.group(2)  
            elif num_match.group(3) and num_match.group(4): res3d, res2d = num_match.group(3)[-3:], num_match.group(4)  
            else: continue  
            extracted.append({'Date': current_date, 'Result_3D': res3d, 'Result_2D': res2d})  
    if len(extracted) < 10: raise Exception("ข้อมูลบนเว็บมีน้อยเกินไป")  
    df = pd.DataFrame(extracted)  
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  
    return df.dropna().sort_values('Date').reset_index(drop=True)  

@st.cache_data(ttl=3600)
def fetch_data_cold(url):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        response = session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        post_body = soup.find('div', class_=re.compile(r'post-body|entry-content'))
        if not post_body: return None
        text_content = post_body.get_text()
        pattern = r"\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d{2})"
        matches = re.findall(pattern, text_content)
        data = []
        for date_str, prize1, bot2 in matches:
            p1_str = str(prize1).zfill(3)
            bot2_str = str(bot2).zfill(2)
            data.append({
                'date': date_str, 'draw_num': prize1,
                'hundred': int(p1_str[-3]), 'ten': int(p1_str[-2]), 'unit': int(p1_str[-1]),
                'bot_ten': int(bot2_str[0]), 'bot_unit': int(bot2_str[1])
            })
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date').reset_index(drop=True)
    except Exception: return None

# ==============================================================================
# 2. ระบบเลขเด่น (HOT) - V.Max
# ==============================================================================
@st.cache_data
def build_features_hot(df, lags, rolls):
    df_feat = df.copy()
    new_features = {}
    df_feat['H'] = df_feat['Result_3D'].str[0].astype(int)  
    df_feat['T'] = df_feat['Result_3D'].str[1].astype(int)  
    df_feat['O'] = df_feat['Result_3D'].str[2].astype(int)  
    df_feat['T2'] = df_feat['Result_2D'].str[0].astype(int)  
    df_feat['O2'] = df_feat['Result_2D'].str[1].astype(int)  
    df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek  
    df_feat['DrawIndex'] = df_feat.index  
    new_features['DigitSum_3D'] = (df_feat['H'].shift(1) + df_feat['T'].shift(1) + df_feat['O'].shift(1)).fillna(0) % 10  
    new_features['Sum_2D'] = (df_feat['T2'].shift(1) + df_feat['O2'].shift(1)).fillna(0) % 10  
    primes = [2, 3, 5, 7]  
    for pos in ['H', 'T', 'O', 'T2', 'O2']:  
        prev = df_feat[pos].shift(1)  
        new_features[f'OddEven_{pos}'] = (prev % 2).fillna(0).astype(int)  
        new_features[f'HighLow_{pos}'] = (prev >= 5).fillna(0).astype(int)  
        new_features[f'Is_Prime_{pos}'] = prev.isin(primes).astype(int)  
        for lag in lags: new_features[f'Lag_{lag}_{pos}'] = df_feat[pos].shift(lag)  
        if f'Lag_1_{pos}' in new_features and f'Lag_2_{pos}' in new_features:  
            new_features[f'Repeat_{pos}'] = (new_features[f'Lag_1_{pos}'] == new_features[f'Lag_2_{pos}']).astype(int)  
            new_features[f'Diff_{pos}'] = (new_features[f'Lag_1_{pos}'] - new_features[f'Lag_2_{pos}']).fillna(0)  
        for w in rolls:  
            new_features[f'EMA_{w}_{pos}'] = prev.ewm(span=w, adjust=False).mean()  
            new_features[f'Roll_Med_{w}_{pos}'] = prev.rolling(w).median().fillna(-1)  
            new_features[f'Roll_Std_{w}_{pos}'] = prev.rolling(w).std().fillna(-1)  
            new_features[f'Momentum_{w}_{pos}'] = (prev - prev.rolling(w).mean()).fillna(0)  
        skips, ranks = np.zeros(len(df_feat)), np.zeros(len(df_feat))  
        last_seen = {}  
        pos_values = df_feat[pos].values  
        for i in range(len(df_feat)):  
            if i == 0:   
                skips[i], ranks[i] = 100, 5  
                last_seen[pos_values[i]] = i  
            else:  
                val_prev = pos_values[i-1]  
                if val_prev in last_seen: skips[i] = (i - 1) - last_seen[val_prev]  
                else: skips[i] = 100  
                seen_distances = {v: ((i - 1) - last_seen.get(v, -100)) for v in range(10)}  
                sorted_by_dist = sorted(seen_distances.items(), key=lambda x: x[1])  
                rank_dict = {v[0]: rank for rank, v in enumerate(sorted_by_dist)}  
                ranks[i] = rank_dict.get(val_prev, 5)  
                last_seen[val_prev] = i - 1  
        new_features[f'Skip_{pos}'] = skips  
        new_features[f'LastRank_{pos}'] = ranks  
    df_new = pd.DataFrame(new_features, index=df_feat.index)  
    df_feat = pd.concat([df_feat, df_new], axis=1)  
    return df_feat.fillna(-1)

def get_entropy_weight(probs):
    h = stats.entropy(probs + 1e-9)
    return max(0.1, (1.0 - (h / np.log(10)))**1.5)

class PositionalEquation:
    def analyze(self, df):
        latest = df.iloc[-1]
        H, T, O = latest['H'], latest['T'], latest['O']
        probs = np.zeros(10)
        for v in [(H + T) % 10, (T + O) % 10, abs(H - O) % 10, (H * T) % 10]: probs[int(v)] += 1.0
        return (probs + 0.1) / (probs + 0.1).sum()

class FrequencyEngineHot:
    def analyze(self, df, pos):
        series = df[pos].dropna()
        probs = np.zeros(10)
        freq_all = series.value_counts(normalize=True).to_dict()
        freq_10 = series.tail(10).value_counts(normalize=True).to_dict()
        for i in range(10):
            idxs = np.where(series == i)[0]
            skip = (len(series) - 1 - idxs[-1]) if len(idxs) > 0 else len(series)
            probs[i] = (freq_all.get(i, 0) * 0.3) + (freq_10.get(i, 0) * 0.5) + ((1.0 / (skip + 1)) * 0.2)
        return (probs + 0.01) / (probs + 0.01).sum()

class ConditionalSystemHot:
    def analyze(self, df, pos, next_date):
        probs = np.zeros(10)
        subset = df[(df['DayOfWeek'] == next_date.dayofweek)]
        if len(subset) == 0: subset = df
        freq = subset[pos].value_counts(normalize=True).to_dict()
        for i in range(10): probs[i] = freq.get(i, 0)
        return (probs + 0.01) / (probs + 0.01).sum()

class MarkovChainSystemHot:
    def analyze(self, df, pos):
        series = df[pos].dropna().values
        n = len(series)
        global_freq = pd.Series(series).value_counts(normalize=True).reindex(range(10), fill_value=0.1).values  
        alpha, trans_1, trans_2, trans_3 = 3.0, np.zeros((10, 10)), np.zeros((10, 10, 10)), np.zeros((10, 10, 10, 10))  
        if n < 2: return np.ones(10) / 10  
        for i in range(n-1): trans_1[int(series[i]), int(series[i+1])] += 1.0  
        for i in range(n-2): trans_2[int(series[i]), int(series[i+1]), int(series[i+2])] += 1.0  
        for i in range(n-3): trans_3[int(series[i]), int(series[i+1]), int(series[i+2]), int(series[i+3])] += 1.0  
        for i in range(10):  
            trans_1[i] = (trans_1[i] + alpha * global_freq) / (trans_1[i].sum() + alpha)  
            for j in range(10):  
                trans_2[i, j] = (trans_2[i, j] + alpha * global_freq) / (trans_2[i, j].sum() + alpha)  
                for k in range(10):  
                    trans_3[i, j, k] = (trans_3[i, j, k] + alpha * global_freq) / (trans_3[i, j, k].sum() + alpha)  
        last_1 = int(series[-1])  
        last_2 = int(series[-2]) if n >= 2 else 0  
        last_3 = int(series[-3]) if n >= 3 else 0  
        p1 = trans_1[last_1]  
        p2 = trans_2[last_2, last_1] if n >= 3 else p1  
        p3 = trans_3[last_3, last_2, last_1] if n >= 4 else p2  
        w1, w2, w3 = (0.20, 0.35, 0.45) if n >= 500 else (0.35, 0.45, 0.20) if n >= 200 else (0.60, 0.40, 0.0)
        probs = (p3 * w3) + (p2 * w2) + (p1 * w1)  
        return probs / probs.sum()

class PatternBacktestSystemHot:
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

class AISystemHot:
    def __init__(self, lottery_id, data_length):
        self.lottery_id = lottery_id
        if data_length >= 700: self.trees, self.depth = 120, 6  
        elif data_length >= 400: self.trees, self.depth = 100, 5  
        elif data_length >= 200: self.trees, self.depth = 80, 4  
        else: self.trees, self.depth = 60, 3  
        self.estimators = [  
            ('hgb', HistGradientBoostingClassifier(max_iter=self.trees, max_leaf_nodes=15, min_samples_leaf=3, random_state=42)),  
            ('xgb', XGBClassifier(n_estimators=self.trees, max_depth=max(1, self.depth-1), learning_rate=0.05, subsample=0.8, tree_method="hist", verbosity=0, random_state=42, n_jobs=-1)),  
            ('et', ExtraTreesClassifier(n_estimators=self.trees//2, max_depth=self.depth, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('rf', RandomForestClassifier(n_estimators=self.trees//2, max_depth=self.depth, class_weight='balanced', random_state=42, n_jobs=-1))  
        ]  
        self.model_name = "Turbo Calibrated AI"  

    def analyze(self, X_train, y_train, X_next, pos, data_hash, sample_weight=None):  
        model_path = os.path.join(CACHE_DIR, f"m_ai_calib_turbo_{self.lottery_id}_{pos}_{data_hash}.joblib")  
        
        le = LabelEncoder()
        y_train_enc = pd.Series(le.fit_transform(y_train), index=y_train.index)

        if not os.path.exists(model_path):  
            for old_file in glob.glob(os.path.join(CACHE_DIR, f"m_ai_calib_turbo_{self.lottery_id}_{pos}_*.joblib")):  
                try: os.remove(old_file)  
                except: pass  
            try:
                if len(X_train) >= 500:  
                    tscv = TimeSeriesSplit(n_splits=2)  
                    score_v, score_s = 0, 0  
                    for train_idx, val_idx in tscv.split(X_train):  
                        voting = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1)  
                        voting.fit(X_train.iloc[train_idx], y_train_enc.iloc[train_idx])  
                        score_v += log_loss(y_train_enc.iloc[val_idx], voting.predict_proba(X_train.iloc[val_idx]), labels=np.arange(10))  
                        
                        stacking = StackingClassifier(estimators=self.estimators, final_estimator=LogisticRegression(class_weight='balanced', max_iter=50), cv=2, n_jobs=-1)  
                        stacking.fit(X_train.iloc[train_idx], y_train_enc.iloc[train_idx])  
                        score_s += log_loss(y_train_enc.iloc[val_idx], stacking.predict_proba(X_train.iloc[val_idx]), labels=np.arange(10))  
                    best_base = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1) if score_v <= score_s else StackingClassifier(estimators=self.estimators, final_estimator=LogisticRegression(class_weight='balanced', max_iter=100), cv=2, n_jobs=-1)  
                else:  
                    best_base = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1)  
            except Exception:
                best_base = VotingClassifier(estimators=self.estimators, voting='soft', n_jobs=-1)
            
            calib_method = 'isotonic' if len(X_train) >= 200 else 'sigmoid'  
            calib_cv = 3 if len(X_train) >= 150 else 2  
            self.model = CalibratedClassifierCV(best_base, method=calib_method, cv=calib_cv)  
            try: self.model.fit(X_train, y_train_enc, sample_weight=sample_weight)  
            except: 
                try: self.model.fit(X_train, y_train_enc)
                except: 
                    self.model = clone(best_base)
                    self.model.fit(X_train, y_train_enc)
            joblib.dump(self.model, model_path)  
        else:  
            self.model = joblib.load(model_path)  

        probs_raw = self.model.predict_proba(X_next)[0]  
        res = np.zeros(10)  
        for idx, c in enumerate(self.model.classes_):
            actual_c = le.inverse_transform([int(c)])[0]
            res[int(actual_c)] = probs_raw[idx]  
            
        if res.sum() == 0: res = np.ones(10) / 10
        return res / res.sum()

class EnsembleEngineHot:
    def __init__(self, df_raw, lottery_name, target_dow=None):
        self.df_raw = df_raw
        self.target_dow = target_dow
        self.lottery_name = lottery_name
        self.lottery_id = lottery_name.split(".")[0].strip()
        n = len(df_raw)
        
        self.test_size = 15 if n >= 700 else 12 if n >= 400 else 8 if n >= 200 else 4
        if n < 100: self.test_size = min(3, max(0, n - 30))  

        self.lags, self.rolls = ([1, 2, 3] if n < 200 else [1, 2, 3, 5]), [3, 5, 10]  
        self.features = ['DayOfWeek', 'DrawIndex', 'DigitSum_3D', 'Sum_2D']  
        for pos in ['H', 'T', 'O', 'T2', 'O2']:  
            self.features.extend([f'OddEven_{pos}', f'HighLow_{pos}', f'Is_Prime_{pos}', f'Skip_{pos}', f'LastRank_{pos}'])  
            for lag in self.lags: self.features.append(f'Lag_{lag}_{pos}')  
            self.features.extend([f'Diff_{pos}', f'Repeat_{pos}'])  
            for w in self.rolls: self.features.extend([f'EMA_{w}_{pos}', f'Roll_Med_{w}_{pos}', f'Roll_Std_{w}_{pos}', f'Momentum_{w}_{pos}'])  

        hash_array = pd.util.hash_pandas_object(df_raw[['Result_3D', 'Result_2D']], index=False).values  
        self.data_hash = f"{hashlib.md5(hash_array).hexdigest()}_{self.test_size}_turbo_vmax"  

        self.pos_sys, self.freq_sys = PositionalEquation(), FrequencyEngineHot()  
        self.cond_sys, self.markov_sys = ConditionalSystemHot(), MarkovChainSystemHot()  
        self.ptn_sys = PatternBacktestSystemHot()  
        self.ai_sys = AISystemHot(self.lottery_id, n)  
        self.base_weights = {'AI': 0.40, 'Freq': 0.15, 'Markov': 0.15, 'Cal': 0.10, 'BT': 0.10, 'Eq': 0.10}  

    def _process_single_position(self, pos, df_hist, X_all, next_x, next_date):  
        bt_size, n = self.test_size, len(df_hist)
        cache_key = os.path.join(CACHE_DIR, f"bt_turbo_{self.lottery_id}_{pos}_{self.data_hash}.joblib")  
        train_len = len(X_all) - bt_size if bt_size > 0 else len(X_all)  
        X_train_full, y_train_full = X_all.iloc[:train_len], df_hist[pos].iloc[:train_len]  

        valid_features = [f for f in X_all.columns if f in X_train_full.columns]  
        mi_scores = mutual_info_classif(X_train_full, y_train_full, random_state=42)  
        mi_series = pd.Series(mi_scores, index=valid_features)  
        
        mi_thresh = 0.010 if n >= 700 else 0.008 if n >= 400 else 0.005 if n >= 200 else 0.002
        target_feats = min(len(valid_features), max(30, int(n * 0.15)))  
        pre_selected = mi_series[mi_series > mi_thresh].sort_values(ascending=False).head(target_feats * 2).index  
        if len(pre_selected) < 10: pre_selected = valid_features[:target_feats]  
          
        corr_matrix = X_all[pre_selected].corr().abs()  
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  
        to_drop = set()  
        for col in upper_tri.columns:  
            high_corr = upper_tri.index[upper_tri[col] > max(0.75, 0.95 - (n / 5000.0))].tolist()  
            for r in high_corr: to_drop.add(r if mi_series[col] > mi_series[r] else col)  
          
        selected_feats = [f for f in pre_selected if f not in to_drop][:target_feats]  
        self.final_feat_count = len(selected_feats)  
        X_all_fs, next_x_fs = X_all[selected_feats], next_x[selected_feats]  
        
        decay_factor = 2.5 if n >= 700 else 2.0 if n >= 400 else 1.6 if n >= 200 else 1.2
        full_sample_weights = np.exp(np.linspace(-decay_factor, 0, len(X_all_fs)))  

        if os.path.exists(cache_key):  
            norm_weights, bt_msg = joblib.load(cache_key)  
        elif n < bt_size + 30 or bt_size <= 0:  
            norm_weights, bt_msg = self.base_weights, "(ข้อมูลน้อย ข้าม Backtest)"  
        else:  
            scores = {k: 0.0 for k in self.base_weights.keys()}  
            lite_trees = max(20, self.ai_sys.trees // 3)  
            lite_estimators = [  
                ('hgb', HistGradientBoostingClassifier(max_iter=lite_trees, max_leaf_nodes=15, min_samples_leaf=3, random_state=42)),  
                ('xgb', XGBClassifier(n_estimators=lite_trees, max_depth=max(1, self.ai_sys.depth-1), learning_rate=0.05, subsample=0.8, tree_method="hist", verbosity=0, random_state=42, n_jobs=-1)),  
                ('et', ExtraTreesClassifier(n_estimators=lite_trees, max_depth=self.ai_sys.depth, class_weight='balanced', random_state=42, n_jobs=-1))  
            ]  
            try: bt_ai_base = StackingClassifier(estimators=lite_estimators, final_estimator=LogisticRegression(class_weight='balanced', max_iter=50), cv=2, n_jobs=-1) if n >= 500 else VotingClassifier(estimators=lite_estimators, voting='soft', n_jobs=-1)
            except: bt_ai_base = VotingClassifier(estimators=lite_estimators, voting='soft', n_jobs=-1)

            for i in range(bt_size):  
                bt_ai_model = clone(bt_ai_base)
                curr_train_len = len(X_all_fs) - bt_size + i  
                X_train_step, y_train_step = X_all_fs.iloc[:curr_train_len], df_hist[pos].iloc[:curr_train_len]  
                X_test_step, actual_val = X_all_fs.iloc[[curr_train_len]], df_hist[pos].iloc[curr_train_len]  
                
                le = LabelEncoder()
                y_train_step_enc = pd.Series(le.fit_transform(y_train_step), index=y_train_step.index)

                try: bt_ai_model.fit(X_train_step, y_train_step_enc)  
                except: 
                    bt_ai_model = VotingClassifier(estimators=lite_estimators, voting='soft', n_jobs=-1)
                    bt_ai_model.fit(X_train_step, y_train_step_enc)
                
                probs_ai = bt_ai_model.predict_proba(X_test_step)[0]  
                ai_res = np.zeros(10)  
                for idx, c in enumerate(bt_ai_model.classes_):
                    actual_c = le.inverse_transform([int(c)])[0]
                    ai_res[int(actual_c)] = probs_ai[idx]  
                
                if ai_res.sum() == 0: ai_res = np.ones(10)/10
                else: ai_res /= ai_res.sum()

                curr_df = df_hist.iloc[:curr_train_len]  
                target_date = df_hist.iloc[curr_train_len]['Date']  
                sys_probs = {  
                    'AI': ai_res, 'Freq': self.freq_sys.analyze(curr_df, pos), 'Cal': self.cond_sys.analyze(curr_df, pos, target_date),  
                    'Markov': self.markov_sys.analyze(curr_df, pos), 'BT': self.ptn_sys.analyze(curr_df, pos), 'Eq': self.pos_sys.analyze(curr_df)  
                }  
                step_weight = np.exp((i - bt_size + 1) * 0.15)   
                for sys_name, p in sys_probs.items():  
                    ranked = np.argsort(p)[::-1]  
                    metric_score = ((1 if actual_val == ranked[0] else 0)*0.4 + (1 if actual_val in ranked[:3] else 0)*0.3 + (1 if actual_val in ranked[:5] else 0)*0.1) + (1.0 / (-np.log(p[actual_val] + 1e-9) + 1.0))*0.1 + (1.0 - (np.sum((p - np.eye(10)[actual_val])**2)/2.0))*0.1  
                    scores[sys_name] += (metric_score * step_weight)  

            total_score = sum(scores.values())  
            norm_weights = {k: v/total_score for k, v in scores.items()}  
            msg = f"(Turbo-BT: AI {norm_weights['AI']*100:.1f}% | Bayesian Markov {norm_weights['Markov']*100:.1f}%)"  
            joblib.dump((norm_weights, msg), cache_key)  
            bt_msg = msg  

        p_ai = self.ai_sys.analyze(X_all_fs, df_hist[pos], next_x_fs, pos, self.data_hash, sample_weight=full_sample_weights)  
        p_fq = self.freq_sys.analyze(df_hist, pos)  
        p_cal = self.cond_sys.analyze(df_hist, pos, next_date)  
        p_mk = self.markov_sys.analyze(df_hist, pos)  
        p_bt = self.ptn_sys.analyze(df_hist, pos)  
        p_eq = self.pos_sys.analyze(df_hist)  

        W = {k: norm_weights[k] * get_entropy_weight(p) for k, p in zip(norm_weights.keys(), [p_ai, p_fq, p_cal, p_mk, p_bt, p_eq])}  
        total_w = sum(W.values()) or 1.0  
        W = {k: v/total_w for k, v in W.items()}  

        final_score = (W['AI']*p_ai + W['Freq']*p_fq + W['Cal']*p_cal + W['Markov']*p_mk + W['BT']*p_bt + W['Eq']*p_eq)  
        final_score /= final_score.sum()  

        return pos, {  
            'AI': sorted([(i, p_ai[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5],  
            'Calendar': sorted([(i, p_cal[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5],  
            'Markov': sorted([(i, p_mk[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5],  
            'Final': sorted([(i, final_score[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5],  
            'Probs_For_Graph': final_score, 'BT_Msg': bt_msg + " [Turbo Applied]", 'Feat_Count': self.final_feat_count  
        }  

    def predict_all(self, status_container):  
        last_date = self.df_raw['Date'].iloc[-1]  
        if self.target_dow is not None:  
            days_ahead = self.target_dow - last_date.dayofweek  
            if days_ahead <= 0: days_ahead += 7  
            next_date = last_date + timedelta(days=days_ahead)  
        else:  
            next_date = calculate_next_draw_date(last_date, self.lottery_name)

        dummy = pd.DataFrame([{'Date': next_date, 'Result_3D': '000', 'Result_2D': '00'}])  
        df_ext = pd.concat([self.df_raw, dummy], ignore_index=True)  
        df_ext = build_features_hot(df_ext, self.lags, self.rolls)  
        
        next_x = df_ext.iloc[[-1]][self.features]  
        df_hist = df_ext.iloc[:-1]  
        X_all = df_hist[self.features]  
        results = []  
          
        for pos in ['H', 'T', 'O', 'T2', 'O2']:  
            status_container.write(f"กำลังวิเคราะห์โมเดลตำแหน่ง {pos}...")
            results.append(self._process_single_position(pos, df_hist, X_all, next_x, next_date))  
              
        return {pos: data for pos, data in results}, next_date

# ==============================================================================
# 3. ระบบเลขดับ (COLD) - PRO V4
# ==============================================================================
@st.cache_data
def build_features_adaptive_cold(df, col, lags, rolls):
    df_feat = df.copy()
    n = len(df)
    df_feat['prev_val'] = df_feat[col].shift(1)
    df_feat['mirror'] = (df_feat['prev_val'] + 5) % 10
    df_feat['is_even'] = (df_feat['prev_val'] % 2 == 0).astype(int)
    df_feat['is_high'] = (df_feat['prev_val'] >= 5).astype(int)
    df_feat['mod3'] = (df_feat['prev_val'] % 3).fillna(0).astype(int)
    df_feat['weekday'] = df_feat['date'].dt.weekday
    df_feat['sin_num'] = np.sin(2 * np.pi * df_feat['prev_val'] / 10).fillna(0)
    df_feat['cos_num'] = np.cos(2 * np.pi * df_feat['prev_val'] / 10).fillna(0)
    df_feat['sin_weekday'] = np.sin(2 * np.pi * df_feat['weekday'] / 7).fillna(0)
    df_feat['cos_weekday'] = np.cos(2 * np.pi * df_feat['weekday'] / 7).fillna(0)
    for lag in lags: df_feat[f'lag_{lag}'] = df_feat[col].shift(lag)
    if 'lag_1' in df_feat.columns and 'lag_2' in df_feat.columns:
        df_feat['repeat_2'] = (df_feat['lag_1'] == df_feat['lag_2']).astype(int)
        if 'lag_3' in df_feat.columns:
            df_feat['repeat_3'] = ((df_feat['lag_1'] == df_feat['lag_2']) & (df_feat['lag_2'] == df_feat['lag_3'])).astype(int)
    for w in rolls:
        df_feat[f'rolling_mean_{w}'] = df_feat[col].shift(1).rolling(w).mean()
        df_feat[f'rolling_std_{w}'] = df_feat[col].shift(1).rolling(w).std()

    history = df_feat[col].values
    hc_windows = list(rolls)
    if n >= 500 and 50 not in hc_windows: hc_windows.append(50)

    stats_cols = {f'{typ}{w}_{d}': np.zeros(n) for typ in ['hot', 'cold'] for w in hc_windows for d in range(10) if not (typ=='cold' and w>=50)}
    skip_cols = {f'skip_{d}': np.full(n, 100) for d in range(10)}
    last_seen = {d: -1 for d in range(10)}

    for i in range(1, n):
        for d in range(10): skip_cols[f'skip_{d}'][i] = i - last_seen[d] if last_seen[d] != -1 else 100
        for w in hc_windows:
            window_slice = history[max(0, i-w):i]
            for d in range(10):
                hot_count = np.sum(window_slice == d)
                stats_cols[f'hot{w}_{d}'][i] = hot_count
                if w < 50: stats_cols[f'cold{w}_{d}'][i] = len(window_slice) - hot_count
        last_seen[history[i]] = i

    for key, val in skip_cols.items(): df_feat[key] = val
    for key, val in stats_cols.items(): df_feat[key] = val
    return df_feat.fillna(-1)

class OptimizedEliminationSystemV4:
    def __init__(self, df, target_col, lotto_name):
        self.df = df.copy()
        self.target_col = target_col
        self.lotto_name = lotto_name
        n = len(self.df)

        if n >= 700:
            self.mode_name, self.trees, self.test_size, self.depth = "Mode 4 (700+ งวด)", 100, 20, 6
            self.lags, self.rolls, self.ai_weights = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20], (1.0, 1.0, 1.0, 1.0)
        elif n >= 400:
            self.mode_name, self.trees, self.test_size, self.depth = "Mode 3 (400-699 งวด)", 100, 20, 5
            self.lags, self.rolls, self.ai_weights = [1, 2, 3, 5, 8, 13], [3, 5, 10, 20], (1.0, 0.9, 0.8, 1.0)
        elif n >= 200:
            self.mode_name, self.trees, self.test_size, self.depth = "Mode 2 (200-399 งวด)", 80, 15, 4
            self.lags, self.rolls, self.ai_weights = [1, 2, 3, 5, 8], [3, 5, 10, 20], (1.0, 0.8, 0.6, 0.5)
        else:
            self.mode_name, self.trees, self.test_size, self.depth = "Mode 1 (100-199 งวด)", 60, 10, 3
            self.lags, self.rolls, self.ai_weights = [1, 2, 3, 5], [3, 5, 10], (1.0, 0.8, 0.5, 0.15)
        if n < 100: self.test_size = min(5, max(0, n - 30))

        self.models = {
            'rf': RandomForestClassifier(n_estimators=self.trees, random_state=42, max_depth=self.depth, n_jobs=1),
            'et': ExtraTreesClassifier(n_estimators=self.trees, random_state=42, max_depth=self.depth, n_jobs=1),
            'hgb': HistGradientBoostingClassifier(random_state=42, max_iter=50),
            'xgb': XGBClassifier(n_estimators=50, max_depth=self.depth, tree_method="hist", verbosity=0, random_state=42, n_jobs=1)
        }
        self.model_weights_dict = {'rf': self.ai_weights[0], 'et': self.ai_weights[1], 'hgb': self.ai_weights[2], 'xgb': self.ai_weights[3]}
        self.df_feat = build_features_adaptive_cold(self.df, self.target_col, tuple(self.lags), tuple(self.rolls))

    def precompute_markov_adaptive(self, df_hist):
        seq = df_hist[self.target_col].values
        n = len(seq)
        if n < 5: return np.ones(10)/10.0
        L1, L2, L3 = seq[-1], seq[-2], seq[-3] if n >= 6 else -1
        mc1, tot1 = np.zeros(10), 0
        for i in range(1, len(seq)-1):
            if seq[i] == L1: mc1[seq[i+1]] += 1; tot1 += 1
        prob_o1 = mc1 / tot1 if tot1 > 0 else np.ones(10)/10.0
        if n < 200: return prob_o1
        mc2, tot2 = np.zeros(10), 0
        for i in range(2, len(seq)-1):
            if seq[i-1] == L2 and seq[i] == L1: mc2[seq[i+1]] += 1; tot2 += 1
        prob_o2 = mc2 / tot2 if tot2 > 0 else prob_o1
        if n < 500: return (0.6 * prob_o2) + (0.4 * prob_o1)
        mc3, tot3 = np.zeros(10), 0
        for i in range(3, len(seq)-1):
            if seq[i-2] == L3 and seq[i-1] == L2 and seq[i] == L1: mc3[seq[i+1]] += 1; tot3 += 1
        prob_o3 = mc3 / tot3 if tot3 > 0 else prob_o2
        return (0.5 * prob_o3) + (0.3 * prob_o2) + (0.2 * prob_o1)

    def calculate_freq_skip(self, df_hist, digit):
        freq = (df_hist[self.target_col] == digit).sum() / max(len(df_hist), 1)
        matches = df_hist[df_hist[self.target_col] == digit]
        skip = len(df_hist) - matches.index[-1] - 1 if len(matches) > 0 else 100
        return (0.5 * min(freq * 10, 1.0)) + (0.5 * max(1.0 - (skip / 30), 0.0))

    def run_backtest(self, X_train, y_train, df_hist_cut, test_size):
        cache_key = f"bt_{self.lotto_name}_{self.target_col}_{len(df_hist_cut)}_{test_size}_ultimate_fs"
        if cache_key in st.session_state.bt_cache: return st.session_state.bt_cache[cache_key]

        bt_train_X, bt_train_y = X_train.iloc[:-test_size], y_train.iloc[:-test_size]
        bt_test_X, bt_test_y = X_train.iloc[-test_size:], y_train.iloc[-test_size:].values

        le = LabelEncoder()
        bt_train_y_enc = le.fit_transform(bt_train_y)

        ai_fails, stat_fails, day_fails = 0, 0, 0
        trained_models = {name: clone(model).fit(bt_train_X, bt_train_y_enc) for name, model in self.models.items()}
        ai_preds = np.zeros((test_size, 10))
        for name, m in trained_models.items():
            preds = m.predict_proba(bt_test_X)
            full_preds = np.zeros((test_size, 10))
            for idx, c in enumerate(m.classes_): 
                actual_c = le.inverse_transform([int(c)])[0]
                full_preds[:, int(actual_c)] = preds[:, idx]
            ai_preds += full_preds * self.model_weights_dict[name]
        ai_preds /= sum(self.model_weights_dict.values())

        for i in range(test_size):
            if bt_test_y[i] in np.argsort(ai_preds[i])[:7]: ai_fails += 1
            curr_hist = df_hist_cut.iloc[:-(test_size - i)]
            mk = self.precompute_markov_adaptive(curr_hist)
            st_probs = np.array([(0.5 * self.calculate_freq_skip(curr_hist, d)) + (0.5 * mk[d]) for d in range(10)])
            if bt_test_y[i] in np.argsort(st_probs)[:7]: stat_fails += 1
            day_df = curr_hist[curr_hist['date'].dt.weekday == df_hist_cut.iloc[-(test_size - i)]['date'].weekday()]
            day_probs = np.zeros(10)
            if len(day_df) > 0:
                counts = day_df[self.target_col].value_counts(normalize=True)
                for d in range(10): day_probs[d] = counts.get(d, 0.0)
            else: day_probs = np.ones(10)/10.0
            if bt_test_y[i] in np.argsort(day_probs)[:7]: day_fails += 1

        result = (ai_fails, stat_fails, day_fails)
        st.session_state.bt_cache[cache_key] = result
        return result

    def analyze(self, target_dow):
        df_hist = self.df
        data_size = len(df_hist)
        if data_size < 30: return None

        feature_cols = [c for c in self.df_feat.columns if c not in ['date', 'draw_num', 'hundred', 'ten', 'unit', 'bot_ten', 'bot_unit', self.target_col]]
        X, y = self.df_feat[feature_cols], self.df_feat[self.target_col]
        train_X, test_X, train_y, df_hist_cut = X.iloc[:-1], X.iloc[-1:], y.iloc[:-1], df_hist.iloc[:-1]

        le = LabelEncoder()
        train_y_enc = pd.Series(le.fit_transform(train_y), index=train_y.index)

        selector = ExtraTreesClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=1).fit(train_X, train_y_enc)
        top_n = min(40 if data_size >= 400 else 30 if data_size >= 200 else 20, len(feature_cols))
        selected_features = [feature_cols[i] for i in np.argsort(selector.feature_importances_)[::-1][:top_n]]
        self.selected_feat_count = len(selected_features)
        train_X, test_X = train_X[selected_features], test_X[selected_features]

        w_ai, w_stat, w_day = (0.50, 0.35, 0.15) if data_size >= 500 else (0.40, 0.40, 0.20) if data_size >= 200 else (0.30, 0.50, 0.20)
        backtest_msg = ""
        if self.test_size > 0 and data_size > self.test_size + 30:
            ai_f, st_f, day_f = self.run_backtest(train_X, train_y, df_hist_cut, self.test_size)
            ai_score, st_score, day_score = max(0.1, 1.0 - (ai_f/self.test_size))**2, max(0.1, 1.0 - (st_f/self.test_size))**2, max(0.1, 1.0 - (day_f/self.test_size))**2
            total_adj = (w_ai*ai_score) + (w_stat*st_score) + (w_day*day_score)
            w_ai, w_stat, w_day = (w_ai*ai_score)/total_adj, (w_stat*st_score)/total_adj, (w_day*day_score)/total_adj
            backtest_msg = f" (BT-Score: AI {int((1-ai_f/self.test_size)*100)}% | Stat {int((1-st_f/self.test_size)*100)}% | Day {int((1-day_f/self.test_size)*100)}%)"

        cache_key = f"{self.lotto_name}_{self.target_col}_{df_hist['date'].iloc[-1].strftime('%Y-%m-%d')}_ultimate_fs"
        if cache_key not in st.session_state.model_cache:
            st.session_state.model_cache[cache_key] = {name: clone(model).fit(train_X, train_y_enc) for name, model in self.models.items()}
        trained_models = st.session_state.model_cache[cache_key]

        ai_probs = np.zeros(10)
        for name, model in trained_models.items():
            preds = model.predict_proba(test_X)[0]
            m_probs = np.zeros(10)
            for idx, c in enumerate(model.classes_): 
                actual_c = le.inverse_transform([int(c)])[0]
                m_probs[int(actual_c)] = preds[idx]
            ai_probs += m_probs * self.model_weights_dict[name]
        ai_probs /= (sum(self.model_weights_dict.values()) or 1.0)
        ai_probs /= (ai_probs.sum() + 1e-9)

        mk = self.precompute_markov_adaptive(df_hist_cut)
        stat_probs = np.array([(0.5 * self.calculate_freq_skip(df_hist_cut, d)) + (0.5 * mk[d]) for d in range(10)])
        stat_probs /= (stat_probs.sum() + 1e-9)

        day_df = df_hist_cut[df_hist_cut['date'].dt.weekday == target_dow]
        day_probs = np.zeros(10)
        if len(day_df) > 0:
            counts = day_df[self.target_col].value_counts(normalize=True)
            for d in range(10): day_probs[d] = counts.get(d, 0.0)
        else: day_probs = np.ones(10)/10.0

        final_probs = (w_ai * ai_probs) + (w_stat * stat_probs) + (w_day * day_probs)
        final_probs /= (final_probs.sum() + 1e-9)

        return {'ai': ai_probs, 'stat': stat_probs, 'day': day_probs, 'final': final_probs, 'w_ai': w_ai, 'w_stat': w_stat, 'w_day': w_day, 'bt_msg': backtest_msg}

# ==============================================================================
# 4. Dashboard (UI ของ Streamlit)
# ==============================================================================
st.title("🎯 ระบบวิเคราะห์หวยครบวงจร (เด่น V.Max & ดับ PRO V4)")

col1, col2 = st.columns(2)
with col1: selected_lotto = st.selectbox("🎯 เลือกหวย:", list(LOTTERY_URLS.keys()))
with col2:
    day_options = {'อัตโนมัติ (คำนวณจากงวดล่าสุด)': None, 'วันจันทร์': 0, 'วันอังคาร': 1, 'วันพุธ': 2, 'วันพฤหัสบดี': 3, 'วันศุกร์': 4, 'วันเสาร์': 5, 'วันอาทิตย์': 6}
    selected_day_name = st.selectbox("📅 ออกวัน:", list(day_options.keys()))
    target_dow_input = day_options[selected_day_name]

# แบ่งปุ่มกดออกเป็น 2 โหมด และปรับให้เป็นสีแดงเหมือนกันทั้งคู่ (type="primary")
btn_col1, btn_col2 = st.columns(2)
with btn_col1: btn_hot = st.button("🚀 วิเคราะห์เลขเด่น (V.Max)", type="primary", use_container_width=True)
with btn_col2: btn_cold = st.button("🛑 วิเคราะห์เลขดับ (PRO V4)", type="primary", use_container_width=True)

dow_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]  

# -----------------------------------------------------------------------------
# RUN โหมดเลขเด่น (HOT)
# -----------------------------------------------------------------------------
if btn_hot:
    url = LOTTERY_URLS[selected_lotto]
    try:
        with st.status("🚀 โหมดเลขเด่น: กำลังดึงข้อมูลและเตรียมวิเคราะห์...", expanded=True) as status:
            df_raw = fetch_data_hot(url)
            status.write("ดึงข้อมูลสำเร็จ! กำลังสร้างฟีเจอร์...")
            engine = EnsembleEngineHot(df_raw, selected_lotto, target_dow=target_dow_input)
            preds, next_date = engine.predict_all(status)
            status.update(label="✨ วิเคราะห์เสร็จสิ้นสมบูรณ์!", state="complete", expanded=False)
        
        labels = {'H': 'หลักร้อย (บน)', 'T': 'หลักสิบ (บน)', 'O': 'หลักหน่วย (บน)', 'T2': 'หลักสิบ (ล่าง)', 'O2': 'หลักหน่วย (ล่าง)'}  
        probs_top = (preds['H']['Probs_For_Graph'] + preds['T']['Probs_For_Graph'] + preds['O']['Probs_For_Graph']) / 3  
        probs_bot = (preds['T2']['Probs_For_Graph'] + preds['O2']['Probs_For_Graph']) / 2  

        def get_top5(probs): return sorted([(i, probs[i]) for i in range(10)], key=lambda x: x[1], reverse=True)[:5]  
        top5_top, top5_bot = get_top5(probs_top), get_top5(probs_bot)  

        st.subheader("🔥 สรุปฟันธง เลขเด่นมาแรง (V.Max Quantum)")
        st.info(f"**🚀 เด่นบนรวม (ร้อย-สิบ-หน่วย) : {' , '.join([str(x[0]) for x in top5_top])}**")
        st.info(f"**⬇️ เด่นล่างรวม (สิบ-หน่วย) : {' , '.join([str(x[0]) for x in top5_bot])}**")
        st.write(f"🔮 ผลวิเคราะห์เลขเด่น ประจำวัน{dow_names[next_date.dayofweek]}ที่ {next_date.strftime('%d-%m-%Y')} (อ้างอิง {len(df_raw)} งวด)")

        for pos in ['H', 'T', 'O', 'T2', 'O2']:  
            with st.expander(f"📍 เจาะลึกเลขเด่น: {labels[pos]}"):
                st.caption(f"คัดฟีเจอร์เด่นสุด {preds[pos]['Feat_Count']} ตัว | {preds[pos]['BT_Msg']}")
                st.markdown(f"- 🧠 **เลขเด่น Quantum AI:** {', '.join([str(num) for num, p in preds[pos]['AI']])}")  
                st.markdown(f"- 🔗 **เลขเด่น มาร์คอฟแบบเบย์:** {', '.join([str(num) for num, p in preds[pos]['Markov']])}")  
                st.markdown(f"- 📅 **เลขเด่น กำลังวัน:** {', '.join([str(num) for num, p in preds[pos]['Calendar']])}")  
                st.markdown(f"- 🌟 **เด่นสรุปรวม 5 ตัว:** {', '.join([str(num) for num, p in preds[pos]['Final']])}")  

        st.subheader("📊 กราฟโอกาสความน่าจะเป็น (เลขเด่น)")
        fig = plt.figure(figsize=(12, 8))  
        colors_list = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  
        for idx, pos in enumerate(['H', 'T', 'O', 'T2', 'O2']):  
            ax = plt.subplot(2, 3, idx + 1)  
            top_5_items = preds[pos]['Final']  
            ax.bar([str(x[0]) for x in top_5_items], [x[1]*100 for x in top_5_items], color=colors_list)  
            ax.set_title(labels[pos])  
            ax.set_ylabel('โอกาส (%)')  
        plt.tight_layout()  
        st.pyplot(fig)  
        plt.close(fig)

    except requests.exceptions.RequestException as e: st.error(f"❌ Network Error: {str(e)}")
    except Exception as e: st.error(f"❌ Error: {str(e)}")

# -----------------------------------------------------------------------------
# RUN โหมดเลขดับ (COLD)
# -----------------------------------------------------------------------------
elif btn_cold:
    url = LOTTERY_URLS[selected_lotto]
    try:
        with st.status("🛑 โหมดเลขดับ: กำลังดึงข้อมูลและเตรียมวิเคราะห์...", expanded=True) as status:
            df = fetch_data_cold(url)
            if df is None or df.empty:
                status.update(label="❌ ไม่สามารถดึงข้อมูลได้ โปรดตรวจสอบการเชื่อมต่อ", state="error")
                st.stop()
            status.write("ข้อมูลพร้อมใช้งาน! เริ่มกระบวนการ Feature Selection...")
            
            sys_status = OptimizedEliminationSystemV4(df, 'hundred', selected_lotto)
            _ = sys_status.analyze(0)  
            
            last_date = df['date'].iloc[-1]
            if target_dow_input is not None:
                days_ahead = target_dow_input - last_date.dayofweek
                if days_ahead <= 0: days_ahead += 7
                target_date = last_date + timedelta(days=days_ahead)
                target_dow = target_dow_input
            else:
                target_date = calculate_next_draw_date(last_date, selected_lotto)
                target_dow = target_date.weekday()

            store_final_probs = {}
            positions = {'💯 3 ตัวบน (ร้อย)': 'hundred', '🔟 3 ตัวบน (สิบ)': 'ten', '1️⃣ 3 ตัวบน (หน่วย)': 'unit', '🔽 2 ตัวล่าง (สิบ)': 'bot_ten', '⬇️ 2 ตัวล่าง (หน่วย)': 'bot_unit'}
            results_output = {}
            for pos_th, col_en in positions.items():
                status.write(f"กำลังสกัดเลขดับ: {pos_th} ...")
                system = OptimizedEliminationSystemV4(df, col_en, selected_lotto)
                res = system.analyze(target_dow)
                if res:
                    store_final_probs[col_en] = res['final']
                    results_output[pos_th] = res
            status.update(label="✨ ประมวลผลเลขดับเสร็จสิ้นสมบูรณ์!", state="complete", expanded=False)

        def get_dead_nums(probs_array, k=7): return [(idx, probs_array[idx]) for idx in np.argsort(probs_array)[:k]]
        def format_dead(dead_list): return " - ".join([str(num) for num, prob in dead_list])

        st.subheader("🧊 สรุปภาพรวมเลขดับ (PRO V4)")
        if all(k in store_final_probs for k in ['hundred', 'ten', 'unit']):
            top_probs = (store_final_probs['hundred'] + store_final_probs['ten'] + store_final_probs['unit']) / 3.0
            st.info(f"🚫 **ดับบนรวม (ร้อย-สิบ-หน่วย) : {format_dead(get_dead_nums(top_probs, 7))}**")
        if all(k in store_final_probs for k in ['bot_ten', 'bot_unit']):
            bot_probs = (store_final_probs['bot_ten'] + store_final_probs['bot_unit']) / 2.0
            st.info(f"🚫 **ดับล่างรวม (สิบ-หน่วย) : {format_dead(get_dead_nums(bot_probs, 7))}**")
            
        st.write(f"🔮 ผลวิเคราะห์เลขดับ ประจำวัน{dow_names[target_dow]}ที่ {target_date.strftime('%d-%m-%Y')} (อ้างอิง {len(df)} งวด)")

        for pos_th, res in results_output.items():
            with st.expander(f"📍 เจาะลึกเลขดับ: {pos_th}"):
                w_ai, w_st, w_dy = int(res['w_ai']*100), int(res['w_stat']*100), int(res['w_day']*100)
                st.caption(f"น้ำหนักสุทธิ: AI {w_ai}% | Stat {w_st}% | Day {w_dy}% {res['bt_msg']}")
                st.write(f"- 🤖 **ดับ AI:** {format_dead(get_dead_nums(res['ai']))}")
                st.write(f"- 📅 **ดับกำลังวัน:** {format_dead(get_dead_nums(res['day']))}")
                st.write(f"- 📊 **ดับสถิติ:** {format_dead(get_dead_nums(res['stat']))}")
                st.markdown(f"- 🌟 **ดับสรุปรวม 7 ตัว:** **{format_dead(get_dead_nums(res['final']))}**")

        # --- เพิ่มกราฟสำหรับเลขดับ (7 ตัวที่โอกาสออกน้อยสุด) ---
        st.subheader("📊 กราฟโอกาสความน่าจะเป็น (เลขดับ 7 อันดับ)")
        fig = plt.figure(figsize=(12, 8))  
        # เลือกใช้สีโทนเย็น (Cold Colors) สำหรับแสดงเลขดับ
        colors_list_cold = ['#1f77b4', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']  
        
        # จัดชื่อหัวกราฟให้สั้นกระชับเหมือนเลขเด่น
        clean_labels = {
            '💯 3 ตัวบน (ร้อย)': 'หลักร้อย (บน)',
            '🔟 3 ตัวบน (สิบ)': 'หลักสิบ (บน)',
            '1️⃣ 3 ตัวบน (หน่วย)': 'หลักหน่วย (บน)',
            '🔽 2 ตัวล่าง (สิบ)': 'หลักสิบ (ล่าง)',
            '⬇️ 2 ตัวล่าง (หน่วย)': 'หลักหน่วย (ล่าง)'
        }
        
        for idx, (pos_th, res) in enumerate(results_output.items()):
            ax = plt.subplot(2, 3, idx + 1)  
            dead_7_items = get_dead_nums(res['final'], 7)  
            # แสดงกราฟแท่งโดยดึงค่าความน่าจะเป็นที่ต่ำสุด 7 อันดับแรก
            ax.bar([str(x[0]) for x in dead_7_items], [x[1]*100 for x in dead_7_items], color=colors_list_cold)  
            ax.set_title(clean_labels.get(pos_th, pos_th))  
            ax.set_ylabel('โอกาส (%)')  
            
        plt.tight_layout()  
        st.pyplot(fig)  
        plt.close(fig)

    except requests.exceptions.RequestException as e: st.error(f"❌ Network Error: {str(e)}")
    except Exception as e: st.error(f"❌ Error: {str(e)}")

else:
    st.info("👈 **กรุณากดปุ่มด้านบนเพื่อเลือกโหมดวิเคราะห์ (เน้นเลขเด่น หรือ หาเลขดับ)**")
