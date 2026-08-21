# ============================================================
# 🚀 LOTTO AI ULTIMATE V.MAX 5-TOP TURBO
# ============================================================
# LEAKAGE SAFE
# NO PERSISTENT MODEL
# RF + EXTRA TREES + HGB
# WALK-FORWARD BACKTEST
# TOP-1 / TOP-3 / TOP-5
# LOGLOSS
# DYNAMIC WEIGHT + SHRINKAGE
# FAST FEATURE ENGINEERING
# MOBILE OPTIMIZED
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

.main-title {
    text-align:center;
    font-size:29px;
    font-weight:900;
    color:#D32F2F;
    margin-top:5px;
}

.sub-title {
    text-align:center;
    color:#666;
    font-size:14px;
    margin-bottom:18px;
}

.hot-card {
    padding:18px;
    border-radius:16px;
    border:2px solid #ff4b4b;
    margin:10px 0;
    background:linear-gradient(to bottom right,#ffffff,#fff5f5);
    box-shadow:0 4px 8px rgba(255,75,75,.10);
}

.number-highlight {
    font-size:36px;
    font-weight:900;
    color:#D32F2F;
    letter-spacing:2px;
}

.dot-sep {
    color:#FFCDD2;
    font-size:26px;
    margin:0 8px;
}

.badge-ai {
    background:#E3F2FD;
    color:#1565C0;
    padding:4px 10px;
    border-radius:15px;
    font-weight:800;
    border:1px solid #BBDEFB;
}

.badge-stat {
    background:#E8F5E9;
    color:#2E7D32;
    padding:4px 10px;
    border-radius:15px;
    font-weight:800;
    border:1px solid #C8E6C9;
}

.badge-cal {
    background:#FFF3E0;
    color:#E65100;
    padding:4px 10px;
    border-radius:15px;
    font-weight:800;
    border:1px solid #FFE0B2;
}

.position-title {
    font-size:20px;
    font-weight:800;
    margin-top:20px;
    color:#333;
    border-bottom:2px solid #eee;
    padding-bottom:6px;
}

.info-row {
    margin:8px 0;
    font-size:14px;
}

.stat-box {
    padding:10px;
    border-radius:12px;
    background:#f8f9fa;
    border:1px solid #e5e5e5;
    text-align:center;
    margin-bottom:8px;
}

.small-muted {
    font-size:12px;
    color:#888;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. DATA FETCH
# ============================================================

@st.cache_data(
    ttl=180,
    show_spinner=False
)
def fetch_and_clean_data(url):

    try:

        headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        main = soup.find(
            "div",
            class_=re.compile(
                r"post-body|entry-content|post-content|content"
            )
        )

        if main is None:
            main = soup

        lines = main.get_text(
            separator="\n"
        ).split("\n")

        date_pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2}"
            r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        )

        num_pattern = re.compile(
            r"\b(\d{3})\b.*?\b(\d{2})\b"
            r"|\b(\d{5,6})\b.*?\b(\d{2})\b"
        )

        current_date = pd.Timestamp(
            datetime.now()
        )

        rows = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            dm = date_pattern.search(line)

            if dm:

                try:

                    d = pd.to_datetime(
                        dm.group(1),
                        errors="coerce"
                    )

                    if not pd.isna(d):
                        current_date = d

                except Exception:
                    pass

            nm = num_pattern.search(line)

            if not nm:
                continue

            if nm.group(1):

                r3 = nm.group(1)
                r2 = nm.group(2)

            elif nm.group(3):

                r3 = nm.group(3)[-3:]
                r2 = nm.group(4)

            else:
                continue

            rows.append({
                "Date": current_date,
                "Result_3D": str(r3).zfill(3),
                "Result_2D": str(r2).zfill(2)
            })

        if len(rows) < 10:
            raise ValueError(
                "ข้อมูลน้อยเกินไป"
            )

        df = pd.DataFrame(rows)

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = (
            df
            .dropna()
            .drop_duplicates()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except Exception as e:

        st.error(
            f"❌ ดึงข้อมูลไม่ได้: {e}"
        )

        return pd.DataFrame()

# ============================================================
# 4. SAFE FEATURE ENGINEERING
# ============================================================

@st.cache_data(
    show_spinner=False
)
def build_features(
    df,
    lags=(1, 2, 3, 5),
    rolls=(3, 5, 10)
):

    x = df.copy()

    r3 = x["Result_3D"].astype(str)
    r2 = x["Result_2D"].astype(str)

    # --------------------------------------------------------
    # Current result columns
    # --------------------------------------------------------

    x["H"] = r3.str[0].astype(np.int8)
    x["T"] = r3.str[1].astype(np.int8)
    x["O"] = r3.str[2].astype(np.int8)

    x["T2"] = r2.str[0].astype(np.int8)
    x["O2"] = r2.str[1].astype(np.int8)

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    x["DOW"] = (
        x["Date"]
        .dt.dayofweek
        .astype(np.int8)
    )

    x["Month"] = (
        x["Date"]
        .dt.month
        .astype(np.int8)
    )

    x["Day"] = (
        x["Date"]
        .dt.day
        .astype(np.int8)
    )

    x["DayOfYear"] = (
        x["Date"]
        .dt.dayofyear
        .astype(np.int16)
    )

    x["Gap"] = (
        x["Date"]
        .diff()
        .dt.days
        .fillna(7)
        .clip(0, 60)
        .astype(np.int16)
    )

    x["DOW_SIN"] = np.sin(
        2 * np.pi * x["DOW"] / 7
    )

    x["DOW_COS"] = np.cos(
        2 * np.pi * x["DOW"] / 7
    )

    x["MONTH_SIN"] = np.sin(
        2 * np.pi * x["Month"] / 12
    )

    x["MONTH_COS"] = np.cos(
        2 * np.pi * x["Month"] / 12
    )

    # --------------------------------------------------------
    # Previous draw features
    # --------------------------------------------------------

    ph = x["H"].shift(1)
    pt = x["T"].shift(1)
    po = x["O"].shift(1)

    x["PrevSum"] = (
        ph + pt + po
    )

    x["PrevOdd"] = (
        (ph % 2) +
        (pt % 2) +
        (po % 2)
    )

    x["PrevHigh"] = (
        (ph >= 5).astype(np.int8) +
        (pt >= 5).astype(np.int8) +
        (po >= 5).astype(np.int8)
    )

    x["DistHT"] = (
        ph - pt
    ).abs()

    x["DistTO"] = (
        pt - po
    ).abs()

    # --------------------------------------------------------
    # Position features
    # IMPORTANT:
    # Everything is shifted / based on previous data.
    # --------------------------------------------------------

    positions = [
        "H",
        "T",
        "O",
        "T2",
        "O2"
    ]

    for pos in positions:

        s = x[pos]

        prev = s.shift(1)

        # Previous parity
        x[f"Odd_{pos}"] = (
            prev % 2
        )

        # Previous high/low
        x[f"High_{pos}"] = (
            prev >= 5
        ).astype(np.int8)

        # Previous prime
        x[f"Prime_{pos}"] = (
            prev.isin(
                [2, 3, 5, 7]
            )
        ).astype(np.int8)

        # ----------------------------------------------------
        # Lag features
        # ----------------------------------------------------

        for lag in lags:

            x[f"L{lag}_{pos}"] = (
                s.shift(lag)
            )

        # ----------------------------------------------------
        # Rolling features
        # IMPORTANT:
        # shift(1) before rolling
        # ----------------------------------------------------

        for w in rolls:

            x[f"RM{w}_{pos}"] = (
                s.shift(1)
                .rolling(
                    w,
                    min_periods=1
                )
                .mean()
            )

        # ----------------------------------------------------
        # SAFE SKIP
        #
        # OLD:
        # skip[i] = gap using s[i]
        #
        # NEW:
        # calculate recurrence gap from PREVIOUS value only.
        # Therefore current target never enters the feature.
        # ----------------------------------------------------

        arr = s.to_numpy()

        skip = np.full(
            len(arr),
            -1,
            dtype=np.float32
        )

        last_seen = np.full(
            10,
            -1,
            dtype=np.int32
        )

        for i in range(len(arr)):

            if i == 0:
                continue

            prev_value = int(
                arr[i - 1]
            )

            if (
                0 <= prev_value <= 9
            ):

                if last_seen[prev_value] >= 0:

                    skip[i] = (
                        (i - 1)
                        -
                        last_seen[prev_value]
                    )

                else:

                    skip[i] = i

                last_seen[prev_value] = (
                    i - 1
                )

        x[f"Skip_{pos}"] = skip

        # ----------------------------------------------------
        # Previous repeat indicator
        # ----------------------------------------------------

        x[f"RepeatPrev_{pos}"] = (
            prev == s.shift(2)
        ).astype(np.int8)

    return (
        x
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(-1)
    )

# ============================================================
# 5. FREQUENCY ENGINE
# ============================================================

class FrequencyEngine:

    def analyze(
        self,
        df,
        pos
    ):

        s = df[pos].astype(int)

        if len(s) == 0:
            return np.ones(10) / 10

        r15 = (
            s.tail(15)
            .value_counts(
                normalize=True
            )
        )

        r30 = (
            s.tail(30)
            .value_counts(
                normalize=True
            )
        )

        all_f = (
            s.value_counts(
                normalize=True
            )
        )

        score = np.array([
            (
                r15.get(d, 0) * 0.55
                +
                r30.get(d, 0) * 0.30
                +
                all_f.get(d, 0) * 0.15
            )
            for d in range(10)
        ])

        score += 0.02

        return (
            score /
            score.sum()
        )

# ============================================================
# 6. CALENDAR ENGINE
# ============================================================

class CalendarEngine:

    def analyze(
        self,
        df,
        pos,
        next_date
    ):

        subset = df[
            df["DOW"]
            ==
            next_date.dayofweek
        ]

        if len(subset) == 0:
            return np.ones(10) / 10

        all_freq = (
            subset[pos]
            .astype(int)
            .value_counts(
                normalize=True
            )
        )

        recent = (
            subset
            .tail(min(25, len(subset)))
        )

        recent_freq = (
            recent[pos]
            .astype(int)
            .value_counts(
                normalize=True
            )
        )

        # Adaptive weight
        if len(subset) >= 25:
            w_recent = 0.70
        elif len(subset) >= 15:
            w_recent = 0.55
        else:
            w_recent = 0.35

        w_all = 1.0 - w_recent

        score = np.array([
            (
                all_freq.get(d, 0)
                * w_all
                +
                recent_freq.get(d, 0)
                * w_recent
            )
            for d in range(10)
        ])

        # Bayesian-style smoothing
        score += 0.02

        return (
            score /
            score.sum()
        )

# ============================================================
# 7. TRANSITION ENGINE
# ============================================================

class TransitionEngine:

    def analyze(
        self,
        df,
        pos
    ):

        if len(df) < 8:
            return np.ones(10) / 10

        current = int(
            df[pos].iloc[-1]
        )

        prev = df[pos].shift(1)

        subset = df[
            prev == current
        ]

        if len(subset) < 2:
            return np.ones(10) / 10

        freq = (
            subset[pos]
            .astype(int)
            .value_counts(
                normalize=True
            )
        )

        score = np.array([
            freq.get(d, 0)
            for d in range(10)
        ])

        score += 0.02

        return (
            score /
            score.sum()
        )

# ============================================================
# 8. PATTERN ENGINE
# ============================================================

class PatternEngine:

    def analyze(
        self,
        df,
        pos
    ):

        if len(df) < 10:
            return np.ones(10) / 10

        a = int(
            df[pos].iloc[-1]
        )

        b = int(
            df[pos].iloc[-2]
        )

        p1 = df[pos].shift(1)
        p2 = df[pos].shift(2)

        subset = df[
            (p1 == a)
            &
            (p2 == b)
        ]

        if len(subset) < 2:

            subset = df[
                p1 == a
            ]

        if len(subset) < 1:
            return np.ones(10) / 10

        freq = (
            subset[pos]
            .astype(int)
            .value_counts(
                normalize=True
            )
        )

        score = np.array([
            freq.get(d, 0)
            for d in range(10)
        ])

        score += 0.02

        return (
            score /
            score.sum()
        )

# ============================================================
# 9. EQUATION ENGINE
# ============================================================

class EquationEngine:

    def analyze(
        self,
        df
    ):

        if len(df) == 0:
            return np.ones(10) / 10

        row = df.iloc[-1]

        h = int(row["H"])
        t = int(row["T"])
        o = int(row["O"])

        vals = [
            (h + t) % 10,
            (t + o) % 10,
            abs(h - o) % 10,
            (h * t) % 10,
            (h + t + o) % 10,
            (h * 2 + o) % 10
        ]

        score = np.ones(
            10,
            dtype=np.float64
        ) * 0.05

        for v in vals:
            score[v] += 1.0

        return (
            score /
            score.sum()
        )

# ============================================================
# 10. AI ENSEMBLE
# ============================================================

class FastAI:

    def __init__(
        self,
        trees=55
    ):

        self.trees = trees

        self.model_weights = {
            "RF": 0.35,
            "ET": 0.35,
            "HGB": 0.30
        }

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    def create_models(
        self,
        backtest=False
    ):

        if backtest:

            rf_trees = 18
            et_trees = 18
            hgb_iter = 25

        else:

            rf_trees = self.trees
            et_trees = self.trees
            hgb_iter = 60

        models = {

            "RF": RandomForestClassifier(
                n_estimators=rf_trees,
                max_depth=6,
                min_samples_leaf=3,
                max_samples=0.85,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            ),

            "ET": ExtraTreesClassifier(
                n_estimators=et_trees,
                max_depth=6,
                min_samples_leaf=3,
                bootstrap=True,
                max_samples=0.85,
                max_features="sqrt",
                n_jobs=-1,
                random_state=43
            ),

            "HGB": HistGradientBoostingClassifier(
                max_iter=hgb_iter,
                learning_rate=0.08,
                max_leaf_nodes=15,
                min_samples_leaf=3,
                l2_regularization=1.2,
                random_state=44
            )
        }

        return models

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict(
        self,
        X,
        y,
        X_next
    ):

        result = np.zeros(
            10,
            dtype=np.float64
        )

        total_weight = 0.0

        models = self.create_models(
            backtest=False
        )

        for name, model in models.items():

            weight = self.model_weights[name]

            try:

                model.fit(
                    X,
                    y
                )

                probs = (
                    model
                    .predict_proba(
                        X_next
                    )[0]
                )

                for cls, p in zip(
                    model.classes_,
                    probs
                ):

                    result[
                        int(cls)
                    ] += (
                        float(p)
                        *
                        weight
                    )

                total_weight += weight

            except Exception:
                continue

        if total_weight <= 0:
            return np.ones(10) / 10

        result /= total_weight

        result += 1e-9

        return (
            result /
            result.sum()
        )

# ============================================================
# 11. ADAPTIVE CONFIG
# ============================================================

def get_config(n):

    if n >= 700:

        return {
            "trees": 55,
            "bt": 20,
            "min_train": 60
        }

    elif n >= 400:

        return {
            "trees": 50,
            "bt": 18,
            "min_train": 55
        }

    elif n >= 200:

        return {
            "trees": 45,
            "bt": 15,
            "min_train": 45
        }

    elif n >= 100:

        return {
            "trees": 40,
            "bt": 12,
            "min_train": 40
        }

    else:

        return {
            "trees": 35,
            "bt": 10,
            "min_train": 30
        }

# ============================================================
# 12. ENSEMBLE ENGINE
# ============================================================

class EnsembleEngine:

    def __init__(
        self,
        df,
        lottery_name,
        target_dow=None
    ):

        self.df = df.copy()

        self.lottery_name = lottery_name

        self.target_dow = target_dow

        n = len(df)

        cfg = get_config(n)

        self.trees = cfg["trees"]

        self.bt = cfg["bt"]

        self.min_train = cfg["min_train"]

        self.lags = (
            1, 2, 3, 5
        )

        self.rolls = (
            3, 5, 10
        )

        # ----------------------------------------------------
        # Feature list
        # ----------------------------------------------------

        self.features = [

            "DOW",
            "Month",
            "Day",
            "DayOfYear",
            "Gap",

            "DOW_SIN",
            "DOW_COS",

            "MONTH_SIN",
            "MONTH_COS",

            "PrevSum",
            "PrevOdd",
            "PrevHigh",

            "DistHT",
            "DistTO"
        ]

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            self.features.extend([
                f"Odd_{pos}",
                f"High_{pos}",
                f"Prime_{pos}",
                f"Skip_{pos}",
                f"RepeatPrev_{pos}"
            ])

            for lag in self.lags:

                self.features.append(
                    f"L{lag}_{pos}"
                )

            for w in self.rolls:

                self.features.append(
                    f"RM{w}_{pos}"
                )

        # ----------------------------------------------------
        # Engines
        # ----------------------------------------------------

        self.freq = FrequencyEngine()

        self.calendar = CalendarEngine()

        self.transition = TransitionEngine()

        self.pattern = PatternEngine()

        self.equation = EquationEngine()

        self.ai = FastAI(
            self.trees
        )

        # ----------------------------------------------------
        # Base weights
        # Equation kept small.
        # ----------------------------------------------------

        self.base_weights = {

            "AI": 0.50,

            "Freq": 0.16,

            "ST": 0.11,

            "Cal": 0.11,

            "Pattern": 0.08,

            "Eq": 0.04
        }

    # ========================================================
    # SAFE SCORE
    # ========================================================

    @staticmethod
    def top_hit(
        prob,
        actual,
        n
    ):

        return (
            actual
            in
            np.argsort(
                prob
            )[::-1][:n]
        )

    # ========================================================
    # BACKTEST
    # ========================================================

    def backtest(
        self,
        pos,
        X,
        df_hist
    ):

        n = len(X)

        if n < self.min_train + 5:

            return (
                self.base_weights.copy(),
                "ข้อมูลน้อยสำหรับ Walk-Forward",
                [],
                {
                    "top1": 0,
                    "top3": 0,
                    "top5": 0,
                    "logloss": None
                }
            )

        start = max(
            self.min_train,
            n - self.bt
        )

        scores = {
            "AI": 0.0,
            "Freq": 0.0,
            "ST": 0.0,
            "Cal": 0.0,
            "Pattern": 0.0
        }

        total_decay = 0.0

        history = []

        logloss_values = []

        top1_count = 0
        top3_count = 0
        top5_count = 0

        # ----------------------------------------------------
        # Walk Forward
        # ----------------------------------------------------

        for step, idx in enumerate(
            range(start, n)
        ):

            # More recent = more important
            decay = (
                1.06 ** step
            )

            total_decay += decay

            Xtr = X.iloc[:idx]

            ytr = (
                df_hist[pos]
                .iloc[:idx]
                .astype(int)
            )

            xt = X.iloc[[idx]]

            actual = int(
                df_hist[pos]
                .iloc[idx]
            )

            target_date = (
                df_hist["Date"]
                .iloc[idx]
            )

            # ------------------------------------------------
            # AI Backtest
            # ------------------------------------------------

            ai_prob = np.ones(10) / 10

            try:

                models = (
                    self.ai.create_models(
                        backtest=True
                    )
                )

                tmp = np.zeros(
                    10,
                    dtype=np.float64
                )

                total_model_weight = 0.0

                for name, model in models.items():

                    mw = (
                        self.ai
                        .model_weights[name]
                    )

                    model.fit(
                        Xtr,
                        ytr
                    )

                    p = (
                        model
                        .predict_proba(
                            xt
                        )[0]
                    )

                    for cls, val in zip(
                        model.classes_,
                        p
                    ):

                        tmp[
                            int(cls)
                        ] += (
                            float(val)
                            *
                            mw
                        )

                    total_model_weight += mw

                if total_model_weight > 0:

                    ai_prob = (
                        tmp /
                        total_model_weight
                    )

            except Exception:
                pass

            # ------------------------------------------------
            # Other engines
            # ------------------------------------------------

            hist = (
                df_hist
                .iloc[:idx]
                .copy()
            )

            fq = self.freq.analyze(
                hist,
                pos
            )

            cal = self.calendar.analyze(
                hist,
                pos,
                target_date
            )

            stp = self.transition.analyze(
                hist,
                pos
            )

            ptn = self.pattern.analyze(
                hist,
                pos
            )

            # ------------------------------------------------
            # Component scores
            # ------------------------------------------------

            if self.top_hit(
                ai_prob,
                actual,
                5
            ):
                scores["AI"] += decay

            if self.top_hit(
                fq,
                actual,
                5
            ):
                scores["Freq"] += decay

            if self.top_hit(
                cal,
                actual,
                5
            ):
                scores["Cal"] += decay

            if self.top_hit(
                stp,
                actual,
                5
            ):
                scores["ST"] += decay

            if self.top_hit(
                ptn,
                actual,
                5
            ):
                scores["Pattern"] += decay

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            ai_rank = (
                np.argsort(
                    ai_prob
                )[::-1]
            )

            if actual == ai_rank[0]:
                top1_count += 1

            if actual in ai_rank[:3]:
                top3_count += 1

            if actual in ai_rank[:5]:
                top5_count += 1

            try:

                ai_ll = log_loss(
                    [actual],
                    [ai_prob],
                    labels=list(range(10))
                )

                logloss_values.append(
                    ai_ll
                )

            except Exception:
                pass

            # ------------------------------------------------
            # Combined proxy
            # ------------------------------------------------

            combined = (

                self.base_weights["AI"]
                * ai_prob

                +

                self.base_weights["Freq"]
                * fq

                +

                self.base_weights["Cal"]
                * cal

                +

                self.base_weights["ST"]
                * stp

                +

                self.base_weights["Pattern"]
                * ptn
            )

            combined += 1e-9

            combined /= combined.sum()

            top5 = (
                np.argsort(
                    combined
                )[::-1][:5]
                .tolist()
            )

            history.append({

                "date":
                    target_date.strftime(
                        "%d/%m/%Y"
                    ),

                "actual":
                    actual,

                "top_5_ordered":
                    top5,

                "is_success":
                    actual in top5
            })

        # ====================================================
        # Dynamic weights
        # ====================================================

        if total_decay <= 0:

            return (
                self.base_weights.copy(),
                "Backtest error",
                [],
                {}
            )

        accuracy = {

            k:
            scores[k]
            /
            total_decay

            for k in scores
        }

        # ----------------------------------------------------
        # Shrinkage
        #
        # Prevent 10-15 observations from moving
        # weights too aggressively.
        # ----------------------------------------------------

        shrink = 0.60

        weighted = {}

        for k in accuracy:

            observed = accuracy[k]

            stable_score = (
                shrink * 0.50
                +
                (1.0 - shrink)
                * observed
            )

            weighted[k] = (
                self.base_weights[k]
                *
                (
                    0.40
                    +
                    0.60
                    * stable_score
                )
            )

        # Equation gets a conservative fixed contribution
        weighted["Eq"] = (
            self.base_weights["Eq"]
            * 0.40
        )

        total = sum(
            weighted.values()
        )

        if total <= 0:

            weights_pct = (
                self.base_weights.copy()
            )

        else:

            weights_pct = {
                k:
                v / total
                for k, v
                in weighted.items()
            }

        # ----------------------------------------------------
        # AI maximum weight
        # ----------------------------------------------------

        if weights_pct["AI"] > 0.58:

            diff = (
                weights_pct["AI"]
                -
                0.58
            )

            weights_pct["AI"] = 0.58

            others = [
                k
                for k
                in weights_pct
                if k != "AI"
            ]

            other_sum = sum(
                weights_pct[k]
                for k in others
            )

            if other_sum > 0:

                for k in others:

                    weights_pct[k] += (
                        diff
                        *
                        (
                            weights_pct[k]
                            /
                            other_sum
                        )
                    )

        # ----------------------------------------------------
        # Backtest message
        # ----------------------------------------------------

        ll = (
            np.mean(
                logloss_values
            )
            if logloss_values
            else None
        )

        metrics = {

            "top1":
                top1_count,

            "top3":
                top3_count,

            "top5":
                top5_count,

            "total":
                max(
                    0,
                    n - start
                ),

            "logloss":
                ll
        }

        bt_msg = (
            f"WF {metrics['total']} งวด | "
            f"AI Top-1 "
            f"{top1_count}/{metrics['total']} | "
            f"Top-3 "
            f"{top3_count}/{metrics['total']} | "
            f"Top-5 "
            f"{top5_count}/{metrics['total']}"
        )

        return (
            weights_pct,
            bt_msg,
            history[-10:],
            metrics
        )

    # ========================================================
    # PROCESS POSITION
    # ========================================================

    def process_position(
        self,
        pos,
        hist,
        X,
        X_next,
        next_date
    ):

        (
            weights,
            bt_msg,
            history,
            metrics
        ) = self.backtest(
            pos,
            X,
            hist
        )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        ai = self.ai.predict(
            X,
            hist[pos].astype(int),
            X_next
        )

        # ----------------------------------------------------
        # Statistical engines
        # ----------------------------------------------------

        fq = self.freq.analyze(
            hist,
            pos
        )

        cal = self.calendar.analyze(
            hist,
            pos,
            next_date
        )

        stp = self.transition.analyze(
            hist,
            pos
        )

        ptn = self.pattern.analyze(
            hist,
            pos
        )

        eq = self.equation.analyze(
            hist
        )

        # ----------------------------------------------------
        # Final ensemble
        # ----------------------------------------------------

        final = (

            weights["AI"]
            * ai

            +

            weights["Freq"]
            * fq

            +

            weights["Cal"]
            * cal

            +

            weights["ST"]
            * stp

            +

            weights["Pattern"]
            * ptn

            +

            weights["Eq"]
            * eq
        )

        final += 1e-9

        final /= final.sum()

        # ----------------------------------------------------
        # Top N
        # ----------------------------------------------------

        def top_n(
            p,
            n
        ):

            idx = (
                np.argsort(
                    p
                )[::-1][:n]
            )

            return [
                (
                    int(i),
                    float(p[i])
                )
                for i in idx
            ]

        return {

            "Final":
                top_n(final, 5),

            "AI":
                top_n(ai, 3),

            "Freq":
                top_n(fq, 3),

            "Calendar":
                top_n(cal, 3),

            "Prob":
                final,

            "Weights":
                weights,

            "BT":
                bt_msg,

            "Metrics":
                metrics,

            "History":
                history
        }

    # ========================================================
    # NEXT DATE
    # ========================================================

    def get_next_date(
        self
    ):

        last_date = (
            self.df["Date"]
            .iloc[-1]
        )

        # ----------------------------------------------------
        # User selected day
        # ----------------------------------------------------

        if self.target_dow is not None:

            days = (
                self.target_dow
                -
                last_date.dayofweek
            ) % 7

            if days == 0:
                days = 7

            return (
                last_date
                +
                timedelta(
                    days=days
                )
            )

        # ----------------------------------------------------
        # Automatic:
        # estimate using latest interval
        # ----------------------------------------------------

        if len(self.df) >= 3:

            gaps = (
                self.df["Date"]
                .diff()
                .dt.days
                .dropna()
                .tail(5)
            )

            if len(gaps) > 0:

                median_gap = int(
                    round(
                        gaps.median()
                    )
                )

                median_gap = max(
                    1,
                    min(
                        median_gap,
                        14
                    )
                )

                return (
                    last_date
                    +
                    timedelta(
                        days=median_gap
                    )
                )

        return (
            last_date
            +
            timedelta(
                days=7
            )
        )

    # ========================================================
    # PREDICT ALL
    # ========================================================

    def predict_all(self):

        next_date = (
            self.get_next_date()
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT append fake 000 / 00 result.
        #
        # We build features from historical data only,
        # then construct next feature row separately.
        # ----------------------------------------------------

        hist = (
            build_features(
                self.df,
                self.lags,
                self.rolls
            )
        )

        X = (
            hist[
                self.features
            ]
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Construct next-date feature row
        # from known history only.
        # ----------------------------------------------------

        next_row = {}

        last = hist.iloc[-1]

        prev_h = int(
            hist["H"].iloc[-1]
        )

        prev_t = int(
            hist["T"].iloc[-1]
        )

        prev_o = int(
            hist["O"].iloc[-1]
        )

        next_row["DOW"] = (
            next_date.dayofweek
        )

        next_row["Month"] = (
            next_date.month
        )

        next_row["Day"] = (
            next_date.day
        )

        next_row["DayOfYear"] = (
            next_date.dayofyear
        )

        next_row["Gap"] = max(
            0,
            min(
                60,
                int(
                    (
                        next_date
                        -
                        self.df["Date"].iloc[-1]
                    ).days
                )
            )
        )

        next_row["DOW_SIN"] = np.sin(
            2 * np.pi
            * next_row["DOW"]
            / 7
        )

        next_row["DOW_COS"] = np.cos(
            2 * np.pi
            * next_row["DOW"]
            / 7
        )

        next_row["MONTH_SIN"] = np.sin(
            2 * np.pi
            * next_row["Month"]
            / 12
        )

        next_row["MONTH_COS"] = np.cos(
            2 * np.pi
            * next_row["Month"]
            / 12
        )

        next_row["PrevSum"] = (
            prev_h
            +
            prev_t
            +
            prev_o
        )

        next_row["PrevOdd"] = (
            (prev_h % 2)
            +
            (prev_t % 2)
            +
            (prev_o % 2)
        )

        next_row["PrevHigh"] = (
            int(prev_h >= 5)
            +
            int(prev_t >= 5)
            +
            int(prev_o >= 5)
        )

        next_row["DistHT"] = abs(
            prev_h - prev_t
        )

        next_row["DistTO"] = abs(
            prev_t - prev_o
        )

        # ----------------------------------------------------
        # Position features for next draw
        # ----------------------------------------------------

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            s = (
                hist[pos]
                .astype(int)
            )

            prev = int(
                s.iloc[-1]
            )

            next_row[
                f"Odd_{pos}"
            ] = prev % 2

            next_row[
                f"High_{pos}"
            ] = int(
                prev >= 5
            )

            next_row[
                f"Prime_{pos}"
            ] = int(
                prev in [2, 3, 5, 7]
            )

            # ------------------------------------------------
            # Lag values
            # ------------------------------------------------

            for lag in self.lags:

                if len(s) >= lag:

                    value = int(
                        s.iloc[-lag]
                    )

                else:

                    value = -1

                next_row[
                    f"L{lag}_{pos}"
                ] = value

            # ------------------------------------------------
            # Rolling
            # ------------------------------------------------

            for w in self.rolls:

                values = (
                    s.tail(w)
                )

                next_row[
                    f"RM{w}_{pos}"
                ] = (
                    values.mean()
                    if len(values)
                    else -1
                )

            # ------------------------------------------------
            # SAFE SKIP FOR NEXT DRAW
            #
            # Number of draws since previous occurrence
            # of the LAST OBSERVED value.
            # No target result is used.
            # ------------------------------------------------

            indices = np.where(
                s.to_numpy() == prev
            )[0]

            if len(indices) >= 2:

                gap = (
                    indices[-1]
                    -
                    indices[-2]
                )

            else:

                gap = len(s)

            next_row[
                f"Skip_{pos}"
            ] = float(gap)

            # Previous repeat
            if len(s) >= 2:

                next_row[
                    f"RepeatPrev_{pos}"
                ] = int(
                    s.iloc[-1]
                    ==
                    s.iloc[-2]
                )

            else:

                next_row[
                    f"RepeatPrev_{pos}"
                ] = 0

        X_next = pd.DataFrame(
            [next_row]
        )[
            self.features
        ].astype(np.float32)

        # ----------------------------------------------------
        # Predict each position
        # ----------------------------------------------------

        results = {}

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            results[pos] = (
                self.process_position(
                    pos,
                    hist,
                    X,
                    X_next,
                    next_date
                )
            )

        return (
            results,
            next_date
        )

# ============================================================
# 13. HTML HELPERS
# ============================================================

def html_top5(
    items
):

    parts = []

    for n, p in items:

        parts.append(
            f'<span class="number-highlight">'
            f'{n}'
            f'</span>'
        )

    return (
        '<span class="dot-sep">•</span>'
        .join(parts)
    )


def html_badge(
    items,
    badge_class
):

    parts = [
        str(n)
        for n, p in items
    ]

    return (
        f'<span class="{badge_class}">'
        +
        " &nbsp;•&nbsp; "
        .join(parts)
        +
        '</span>'
    )


def nums_prob(
    items
):

    return " | ".join(
        f"{n} ({p:.1%})"
        for n, p in items
    )


def combine_top_n(
    preds,
    positions,
    n=5
):

    score = (
        sum(
            preds[pos]["Prob"]
            for pos in positions
        )
        /
        len(positions)
    )

    idx = (
        np.argsort(
            score
        )[::-1][:n]
    )

    return [
        (
            int(i),
            float(score[i])
        )
        for i in idx
    ]

# ============================================================
# 14. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚀 LOTTO AI V.MAX'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    '5-TOP TURBO • Leakage Safe • Walk-Forward<br>'
    '<b>RF + ExtraTrees + HGB | AI + สถิติ + วัน + Pattern</b>'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# 15. INPUT
# ============================================================

c1, c2 = st.columns(2)

selected_lotto = c1.selectbox(
    "🎯 เลือกหวย",
    list(
        LOTTERY_SOURCES.keys()
    )
)

day_options = {
    "อัตโนมัติ": None,
    "วันจันทร์": 0,
    "วันอังคาร": 1,
    "วันพุธ": 2,
    "วันพฤหัสบดี": 3,
    "วันศุกร์": 4,
    "วันเสาร์": 5,
    "วันอาทิตย์": 6
}

day_label = c2.selectbox(
    "📅 วันออกรางวัล",
    list(
        day_options.keys()
    )
)

# ============================================================
# 16. RUN
# ============================================================

if st.button(
    "🚀 วิเคราะห์เลขเด่นด้วย AI TURBO",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "⚡ กำลังดึงข้อมูล + Walk-Forward + AI Ensemble..."
    ):

        df = fetch_and_clean_data(
            LOTTERY_SOURCES[
                selected_lotto
            ]
        )

        if df.empty:
            st.stop()

        # ----------------------------------------------------
        # Data information
        # ----------------------------------------------------

        st.success(
            f"โหลดข้อมูลสำเร็จ "
            f"{len(df)} งวด"
        )

        engine = EnsembleEngine(
            df,
            selected_lotto,
            day_options[
                day_label
            ]
        )

        preds, next_date = (
            engine.predict_all()
        )

        labels = {

            "H":
                "หลักร้อย 3 ตัวบน",

            "T":
                "หลักสิบ 3 ตัวบน",

            "O":
                "หลักหน่วย 3 ตัวบน",

            "T2":
                "หลักสิบ 2 ตัวล่าง",

            "O2":
                "หลักหน่วย 2 ตัวล่าง"
        }

        days = [
            "จันทร์",
            "อังคาร",
            "พุธ",
            "พฤหัสบดี",
            "ศุกร์",
            "เสาร์",
            "อาทิตย์"
        ]

        # ====================================================
        # TARGET
        # ====================================================

        st.divider()

        st.info(
            f"📅 งวดเป้าหมาย: "
            f"วัน{days[next_date.dayofweek]} "
            f"{next_date.strftime('%d-%m-%Y')} "
            f"| ข้อมูล {len(df)} งวด"
        )

        # ====================================================
        # POSITION RESULTS
        # ====================================================

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            res = preds[pos]

            st.markdown(
                f'<div class="position-title">'
                f'📍 {labels[pos]}'
                f'</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # HOT TOP 5
            # ------------------------------------------------

            st.markdown(
f"""
<div class="hot-card">
    <div style="font-weight:700; color:#444; margin-bottom:8px;">
        🔥 HOT TOP-5
    </div>
    <div style="text-align:center; margin:10px 0;">
        {html_top5(res["Final"])}
    </div>
    <div style="font-size:13px; color:#888; text-align:center; margin-top:8px;">
        {nums_prob(res["Final"])}
    </div>
</div>
""",
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # COMPONENT TOP 3
            # ------------------------------------------------

            st.markdown(
f"""
<div class="info-row">
    🤖 <b>AI TOP-3:</b> &nbsp; {html_badge(res["AI"], "badge-ai")}
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown(
f"""
<div class="info-row">
    📊 <b>สถิติ TOP-3:</b> &nbsp; {html_badge(res["Freq"], "badge-stat")}
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown(
f"""
<div class="info-row">
    📅 <b>กำลังวัน TOP-3:</b> &nbsp; {html_badge(res["Calendar"], "badge-cal")}
</div>
""",
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # BACKTEST
            # ------------------------------------------------

            st.markdown(
f"""
<div class="small-muted">
    📈 {res["BT"]}
</div>
""",
                unsafe_allow_html=True
            )

            metrics = res[
                "Metrics"
            ]

            if metrics:

                total_bt = metrics.get(
                    "total",
                    0
                )

                if total_bt > 0:

                    top1_pct = (
                        metrics["top1"]
                        /
                        total_bt
                    )

                    top3_pct = (
                        metrics["top3"]
                        /
                        total_bt
                    )

                    top5_pct = (
                        metrics["top5"]
                        /
                        total_bt
                    )

                    ll = metrics.get(
                        "logloss"
                    )

                    ll_text = (
                        f"{ll:.3f}"
                        if ll is not None
                        else "-"
                    )

                    st.markdown(
f"""
<div style="display:flex; gap:6px; margin-top:8px;">
    <div class="stat-box" style="flex:1;">
        <b>Top-1</b><br>{top1_pct:.0%}
    </div>
    <div class="stat-box" style="flex:1;">
        <b>Top-3</b><br>{top3_pct:.0%}
    </div>
    <div class="stat-box" style="flex:1;">
        <b>Top-5</b><br>{top5_pct:.0%}
    </div>
    <div class="stat-box" style="flex:1;">
        <b>LogLoss</b><br>{ll_text}
    </div>
</div>
""",
                        unsafe_allow_html=True
                    )

            # ------------------------------------------------
            # WEIGHTS
            # ------------------------------------------------

            W = res[
                "Weights"
            ]

            st.markdown(
f"""
<div class="small-muted">
    ⚖️ น้ำหนัก: AI {W["AI"]:.0%} | สถิติ {W["Freq"]:.0%} | วัน {W["Cal"]:.0%} | ก้าวเดิน {W["ST"]:.0%} | Pattern {W["Pattern"]:.0%} | Eq {W["Eq"]:.0%}
</div>
""",
                unsafe_allow_html=True
            )

            # =================================================
            # HISTORY 10
            # =================================================

            with st.expander(
                f"🕰️ Backtest ย้อนหลัง 10 งวด — {labels[pos]}",
                expanded=False
            ):

                history = res.get(
                    "History",
                    []
                )

                if history:

                    recent_hist = (
                        history[::-1]
                    )

                    # ------------------------------------------------
                    # Summary
                    # ------------------------------------------------

                    wins = sum(
                        1
                        for h in recent_hist
                        if h["is_success"]
                    )

                    total_hist = len(
                        recent_hist
                    )

                    rate = (
                        wins /
                        total_hist
                        if total_hist
                        else 0
                    )

                    st.markdown(
f"""
<div class="stat-box">
    🏆 TOP-5 เข้า <b>{wins}/{total_hist}</b> ({rate:.0%})
</div>
""",
                        unsafe_allow_html=True
                    )

                    # ------------------------------------------------
                    # Table
                    # ------------------------------------------------

                    html_table = """
<div style="overflow-x:auto;">
<table style="width:100%; text-align:center; border-collapse:collapse; font-family:sans-serif; font-size:13px;">
<tr style="background:#f1f3f4; color:#333;">
    <th style="padding:9px; border-bottom:2px solid #ccc;">วันที่</th>
    <th style="padding:9px; border-bottom:2px solid #ccc;">TOP-5</th>
    <th style="padding:9px; border-bottom:2px solid #ccc;">จริง</th>
    <th style="padding:9px; border-bottom:2px solid #ccc;">ผล</th>
</tr>
"""

                    for h in recent_hist:

                        bg = "#F1F8E9" if h["is_success"] else "#FFEBEE"
                        icon = "✅ WIN" if h["is_success"] else "❌ หลุด"

                        parts = []
                        for n in h["top_5_ordered"]:
                            if n == h["actual"]:
                                parts.append(f'<span style="color:#D32F2F; font-weight:900; font-size:16px;">{n}</span>')
                            else:
                                parts.append(str(n))

                        top5_str = " - ".join(parts)

                        html_table += f"""
<tr style="background:{bg}; border-bottom:1px solid #ddd;">
    <td style="padding:9px;">{h["date"]}</td>
    <td style="padding:9px; font-weight:700;">{top5_str}</td>
    <td style="padding:9px; font-size:16px; font-weight:900;">{h["actual"]}</td>
    <td style="padding:9px; font-weight:800;">{icon}</td>
</tr>
"""

                    html_table += """
</table>
</div>
"""
                    st.markdown(
                        html_table,
                        unsafe_allow_html=True
                    )

                else:

                    st.info(
                        "ข้อมูลไม่เพียงพอสำหรับ Backtest"
                    )

            st.write("")

        # ====================================================
        # OVERALL
        # ====================================================

        hot_top = combine_top_n(
            preds,
            ["H", "T", "O"]
        )

        hot_bottom = combine_top_n(
            preds,
            ["T2", "O2"]
        )

        st.divider()

        st.subheader(
            "🔥 สรุปเลขเด่นภาพรวม"
        )

        # ----------------------------------------------------
        # TOP
        # ----------------------------------------------------

        st.markdown(
f"""
<div class="hot-card">
    <div style="font-weight:700; color:#444; margin-bottom:8px;">
        🔥 HOT 5-TOP รูด/วิ่ง — บน
    </div>
    <div style="text-align:center; margin:10px 0;">
        {html_top5(hot_top)}
    </div>
    <div style="font-size:13px; color:#888; text-align:center; margin-top:8px;">
        {nums_prob(hot_top)}
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # BOTTOM
        # ----------------------------------------------------

        st.markdown(
f"""
<div class="hot-card">
    <div style="font-weight:700; color:#444; margin-bottom:8px;">
        🔥 HOT 5-TOP รูด/วิ่ง — ล่าง
    </div>
    <div style="text-align:center; margin:10px 0;">
        {html_top5(hot_bottom)}
    </div>
    <div style="font-size:13px; color:#888; text-align:center; margin-top:8px;">
        {nums_prob(hot_bottom)}
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        # ====================================================
        # FINAL NOTICE
        # ====================================================

        st.success(
            "✅ วิเคราะห์เสร็จสิ้น "
            "• Leakage Safe "
            "• Walk-Forward "
            "• Dynamic Weight "
            "• AI Ensemble "
            "• Top-1/3/5"
        )

        st.caption(
            "หมายเหตุ: ผลลัพธ์เป็นการจัดอันดับความน่าจะเป็นจากข้อมูลย้อนหลัง "
            "ไม่สามารถรับประกันผลรางวัลจริงได้"
        )
