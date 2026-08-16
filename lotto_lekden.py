# ============================================================
# 🚀 LOTTO AI ULTIMATE ENSEMBLE V.MAX 3-TOP
# ============================================================
# SEQUENTIAL DRAW-TO-DRAW
# STRICT WALK-FORWARD
# TOP-3 OPTIMIZED
# HOT TOP-3
# DEAD TOP-7
# TIME-DECAY BACKTEST
# DYNAMIC ENSEMBLE WEIGHT
# LEAKAGE SAFE
# MOBILE FRIENDLY
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

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

# XGBoost = optional
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================
# 0. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Lotto AI Ultimate V.Max 3-Top",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 1. LOTTERY SOURCES
# ============================================================

LOTTERY_SOURCES = {
    "1. หวยไทย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",

    "2. หวยธกส.":
        "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",

    "3. หวยออมสิน":
        "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",

    "4. หวยลาว":
        "https://suksan18190.blogspot.com/2026/07/blog-post.html",

    "5. หวยฮานอย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",

    "6. หวยมาเลย์":
        "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",

    "7. หวยหุ้นไทยเย็น":
        "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",

    "8. หวยหุ้นนิเคอิบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",

    "9. หวยหุ้นฮั่งเส็งบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",

    "10. หวยหุ้นจีนบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
}


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 27px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 4px;
}

.sub-title {
    text-align: center;
    color: #777;
    font-size: 14px;
    margin-bottom: 15px;
}

.position-card {
    padding: 14px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 12px;
}

.hot-box {
    padding: 14px;
    border-radius: 14px;
    border: 2px solid #ff9800;
    margin-bottom: 10px;
}

.dead-box {
    padding: 14px;
    border-radius: 14px;
    border: 2px solid #777;
    margin-bottom: 10px;
}

.big-number {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 3px;
}

.small-label {
    font-size: 13px;
    color: #777;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. FETCH DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_clean_data(url):

    try:

        headers = {
            "User-Agent":
            "Mozilla/5.0 (Linux; Android 10) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120 Mobile Safari/537.36"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        main_content = soup.find(
            "div",
            class_=re.compile(
                r"post-body|entry-content|post-content|content"
            )
        )

        if main_content is None:
            main_content = soup

        text_lines = (
            main_content
            .get_text(separator="\n")
            .split("\n")
        )

        extracted = []

        date_pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2}|"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        )

        num_pattern = re.compile(
            r"\b(\d{3})\b.*?\b(\d{2})\b|"
            r"\b(\d{5,6})\b.*?\b(\d{2})\b"
        )

        current_date = pd.Timestamp(datetime.now())

        for line in text_lines:

            line = line.strip()

            if not line:
                continue

            date_match = date_pattern.search(line)

            if date_match:

                try:
                    parsed_date = pd.to_datetime(
                        date_match.group(1),
                        errors="coerce"
                    )

                    if not pd.isna(parsed_date):
                        current_date = parsed_date

                except Exception:
                    pass

            num_match = num_pattern.search(line)

            if not num_match:
                continue

            if num_match.group(1) and num_match.group(2):

                res3d = num_match.group(1)
                res2d = num_match.group(2)

            elif num_match.group(3) and num_match.group(4):

                res3d = num_match.group(3)[-3:]
                res2d = num_match.group(4)

            else:
                continue

            extracted.append({
                "Date": current_date,
                "Result_3D": str(res3d).zfill(3),
                "Result_2D": str(res2d).zfill(2)
            })

        if len(extracted) < 10:
            raise ValueError(
                "ข้อมูลที่ดึงมาไม่เพียงพอ"
            )

        df = pd.DataFrame(extracted)

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "Date",
                "Result_3D",
                "Result_2D"
            ]
        )

        df = df.drop_duplicates(
            subset=[
                "Date",
                "Result_3D",
                "Result_2D"
            ]
        )

        df = df.sort_values(
            "Date"
        ).reset_index(drop=True)

        return df

    except Exception as e:

        st.error(
            f"❌ ไม่สามารถดึงข้อมูลได้: {e}"
        )

        return pd.DataFrame()


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def build_features(df, lags, rolls):

    df_feat = df.copy()

    # --------------------------------------------------------
    # Digit extraction
    # --------------------------------------------------------

    df_feat["H"] = (
        df_feat["Result_3D"]
        .astype(str)
        .str[0]
        .astype(int)
    )

    df_feat["T"] = (
        df_feat["Result_3D"]
        .astype(str)
        .str[1]
        .astype(int)
    )

    df_feat["O"] = (
        df_feat["Result_3D"]
        .astype(str)
        .str[2]
        .astype(int)
    )

    df_feat["T2"] = (
        df_feat["Result_2D"]
        .astype(str)
        .str[0]
        .astype(int)
    )

    df_feat["O2"] = (
        df_feat["Result_2D"]
        .astype(str)
        .str[1]
        .astype(int)
    )

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    df_feat["DayOfWeek"] = (
        df_feat["Date"].dt.dayofweek
    )

    df_feat["Month"] = (
        df_feat["Date"].dt.month
    )

    df_feat["Day"] = (
        df_feat["Date"].dt.day
    )

    df_feat["DayOfYear"] = (
        df_feat["Date"].dt.dayofyear
    )

    df_feat["Gap"] = (
        df_feat["Date"]
        .diff()
        .dt.days
        .fillna(7)
        .clip(lower=0)
        .astype(int)
    )

    df_feat["DOW_SIN"] = np.sin(
        2 * np.pi *
        df_feat["DayOfWeek"] / 7
    )

    df_feat["DOW_COS"] = np.cos(
        2 * np.pi *
        df_feat["DayOfWeek"] / 7
    )

    df_feat["MONTH_SIN"] = np.sin(
        2 * np.pi *
        df_feat["Month"] / 12
    )

    df_feat["MONTH_COS"] = np.cos(
        2 * np.pi *
        df_feat["Month"] / 12
    )

    # --------------------------------------------------------
    # Previous draw only
    # --------------------------------------------------------

    df_feat["PrevSum3"] = (
        df_feat["H"].shift(1)
        + df_feat["T"].shift(1)
        + df_feat["O"].shift(1)
    )

    df_feat["PrevRange3"] = (
        df_feat[
            ["H", "T", "O"]
        ]
        .shift(1)
        .max(axis=1)
        -
        df_feat[
            ["H", "T", "O"]
        ]
        .shift(1)
        .min(axis=1)
    )

    df_feat["PrevOdd3"] = (
        (
            df_feat["H"].shift(1)
            % 2
        )
        +
        (
            df_feat["T"].shift(1)
            % 2
        )
        +
        (
            df_feat["O"].shift(1)
            % 2
        )
    )

    df_feat["PrevHigh3"] = (
        (
            df_feat["H"].shift(1) >= 5
        ).astype(int)
        +
        (
            df_feat["T"].shift(1) >= 5
        ).astype(int)
        +
        (
            df_feat["O"].shift(1) >= 5
        ).astype(int)
    )

    # --------------------------------------------------------
    # Cross digit
    # --------------------------------------------------------

    df_feat["Dist_HT"] = (
        df_feat["H"].shift(1)
        -
        df_feat["T"].shift(1)
    ).abs()

    df_feat["Dist_TO"] = (
        df_feat["T"].shift(1)
        -
        df_feat["O"].shift(1)
    ).abs()

    df_feat["Dist_HO"] = (
        df_feat["H"].shift(1)
        -
        df_feat["O"].shift(1)
    ).abs()

    positions = [
        "H",
        "T",
        "O",
        "T2",
        "O2"
    ]

    prime_digits = [2, 3, 5, 7]

    # --------------------------------------------------------
    # Position features
    # --------------------------------------------------------

    for pos in positions:

        prev = df_feat[pos].shift(1)

        # Odd / Even
        df_feat[
            f"OddEven_{pos}"
        ] = (
            prev % 2
        ).fillna(0)

        # High / Low
        df_feat[
            f"HighLow_{pos}"
        ] = (
            prev >= 5
        ).fillna(0).astype(int)

        # Prime
        df_feat[
            f"IsPrime_{pos}"
        ] = (
            prev.isin(prime_digits)
        ).astype(int)

        # Mirror
        df_feat[
            f"Mirror_{pos}"
        ] = (
            (prev + 5) % 10
        ).fillna(0)

        # ----------------------------------------------------
        # Lags
        # ----------------------------------------------------

        for lag in lags:

            df_feat[
                f"Lag_{lag}_{pos}"
            ] = df_feat[pos].shift(lag)

        # ----------------------------------------------------
        # Rolling
        # ----------------------------------------------------

        shifted = df_feat[pos].shift(1)

        for w in rolls:

            df_feat[
                f"Roll_{w}_Mean_{pos}"
            ] = (
                shifted
                .rolling(
                    w,
                    min_periods=1
                )
                .mean()
            )

            df_feat[
                f"Roll_{w}_Std_{pos}"
            ] = (
                shifted
                .rolling(
                    w,
                    min_periods=1
                )
                .std()
            )

        # ----------------------------------------------------
        # Repeat
        # ----------------------------------------------------

        if (
            f"Lag_1_{pos}" in df_feat.columns
            and
            f"Lag_2_{pos}" in df_feat.columns
        ):

            df_feat[
                f"Repeat_{pos}"
            ] = (
                df_feat[
                    f"Lag_1_{pos}"
                ]
                ==
                df_feat[
                    f"Lag_2_{pos}"
                ]
            ).astype(int)

        else:

            df_feat[
                f"Repeat_{pos}"
            ] = 0

        # ----------------------------------------------------
        # Hot20
        # ----------------------------------------------------

        for d in range(10):

            df_feat[
                f"Hot20_{pos}_{d}"
            ] = (
                shifted
                .eq(d)
                .rolling(
                    20,
                    min_periods=1
                )
                .sum()
            )

        # ----------------------------------------------------
        # Skip
        # ----------------------------------------------------

        values = df_feat[pos].values

        skips = np.zeros(
            len(values),
            dtype=float
        )

        last_seen = {}

        for i, value in enumerate(values):

            if i == 0:

                skips[i] = 100

            else:

                if value in last_seen:
                    skips[i] = (
                        i -
                        last_seen[value]
                    )
                else:
                    skips[i] = i

            last_seen[value] = i

        df_feat[
            f"Skip_{pos}"
        ] = skips

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df_feat = df_feat.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df_feat = df_feat.fillna(-1)

    return df_feat


# ============================================================
# 5. POSITIONAL EQUATION
# ============================================================

class PositionalEquation:

    def analyze(self, df):

        latest = df.iloc[-1]

        H = int(latest["H"])
        T = int(latest["T"])
        O = int(latest["O"])

        probs = np.zeros(10)

        values = [

            (H + T) % 10,

            (T + O) % 10,

            abs(H - O) % 10,

            (H * T) % 10,

            (H + T + O) % 10,

            (H * 2 + O) % 10,

            (T * 2 + H) % 10,

            (H + O * 2) % 10

        ]

        for v in values:
            probs[int(v)] += 1

        probs += 0.05

        return (
            probs /
            probs.sum()
        )


# ============================================================
# 6. FREQUENCY ENGINE
# ============================================================

class FrequencyEngine:

    def analyze(self, df, pos):

        series = (
            df[pos]
            .dropna()
            .astype(int)
        )

        if len(series) == 0:
            return np.ones(10) / 10

        probs = np.zeros(10)

        freq_all = (
            series
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        freq_10 = (
            series
            .tail(10)
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        freq_20 = (
            series
            .tail(20)
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        for i in range(10):

            idxs = np.where(
                series.values == i
            )[0]

            if len(idxs):

                skip = (
                    len(series)
                    - 1
                    - idxs[-1]
                )

            else:

                skip = len(series)

            # Frequency
            f_all = freq_all.get(
                i,
                0
            )

            f10 = freq_10.get(
                i,
                0
            )

            f20 = freq_20.get(
                i,
                0
            )

            # Recent emphasis
            recent_score = (
                f10 * 0.50
                +
                f20 * 0.30
                +
                f_all * 0.20
            )

            # Small skip component
            skip_score = (
                1.0 /
                (skip + 1)
            )

            probs[i] = (
                recent_score * 0.85
                +
                skip_score * 0.15
            )

        probs += 0.01

        return (
            probs /
            probs.sum()
        )


# ============================================================
# 7. CONDITIONAL / CALENDAR
# ============================================================

class ConditionalSystem:

    def analyze(
        self,
        df,
        pos,
        next_date
    ):

        probs = np.zeros(10)

        subset = df[
            df["DayOfWeek"]
            ==
            next_date.dayofweek
        ]

        if len(subset) < 5:
            subset = df

        # Recent same weekday gets extra importance
        recent_subset = subset.tail(20)

        freq_all = (
            subset[pos]
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        freq_recent = (
            recent_subset[pos]
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        for i in range(10):

            probs[i] = (
                freq_all.get(i, 0)
                * 0.40
                +
                freq_recent.get(i, 0)
                * 0.60
            )

        probs += 0.01

        return (
            probs /
            probs.sum()
        )


# ============================================================
# 8. STATE TRANSITION
# ============================================================

class StateTransitionSystem:

    def analyze(self, df, pos):

        probs = np.zeros(10)

        if len(df) < 5:
            return np.ones(10) / 10

        last_value = int(
            df[pos].iloc[-1]
        )

        subset = df[
            df[f"Lag_1_{pos}"]
            ==
            last_value
        ]

        if len(subset) < 2:

            return np.ones(10) / 10

        freq = (
            subset[pos]
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        for i in range(10):
            probs[i] = freq.get(
                i,
                0
            )

        probs += 0.01

        return (
            probs /
            probs.sum()
        )


# ============================================================
# 9. PATTERN SYSTEM
# ============================================================

class PatternBacktestSystem:

    def analyze(self, df, pos):

        probs = np.zeros(10)

        if len(df) < 5:
            return np.ones(10) / 10

        l1 = int(
            df[pos].iloc[-1]
        )

        l2 = int(
            df[pos].iloc[-2]
        )

        subset = df[
            (
                df[f"Lag_1_{pos}"]
                == l1
            )
            &
            (
                df[f"Lag_2_{pos}"]
                == l2
            )
        ]

        if len(subset) < 2:

            subset = df[
                df[f"Lag_1_{pos}"]
                == l1
            ]

        if len(subset) == 0:

            return np.ones(10) / 10

        freq = (
            subset[pos]
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        for i in range(10):
            probs[i] = freq.get(
                i,
                0
            )

        probs += 0.01

        return (
            probs /
            probs.sum()
        )


# ============================================================
# 10. AI SYSTEM
# ============================================================

class AISystem:

    def __init__(
        self,
        trees,
        rf_w,
        et_w,
        hgb_w,
        xgb_w
    ):

        self.trees = trees

        self.rf_w = rf_w
        self.et_w = et_w
        self.hgb_w = hgb_w
        self.xgb_w = xgb_w

    def _make_models(self):

        models = []
        weights = []

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        models.append(
            RandomForestClassifier(
                n_estimators=self.trees,
                max_depth=7,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
        )

        weights.append(
            self.rf_w
        )

        # ----------------------------------------------------
        # Extra Trees
        # ----------------------------------------------------

        models.append(
            ExtraTreesClassifier(
                n_estimators=self.trees,
                max_depth=8,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=43
            )
        )

        weights.append(
            self.et_w
        )

        # ----------------------------------------------------
        # HistGradientBoosting
        # ----------------------------------------------------

        models.append(
            HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.045,
                max_leaf_nodes=15,
                min_samples_leaf=5,
                l2_regularization=0.15,
                random_state=44
            )
        )

        weights.append(
            self.hgb_w
        )

        # ----------------------------------------------------
        # XGBoost optional
        # ----------------------------------------------------

        if (
            XGB_AVAILABLE
            and
            self.xgb_w > 0
        ):

            models.append(
                XGBClassifier(
                    n_estimators=70,
                    max_depth=4,
                    learning_rate=0.045,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=2,
                    reg_lambda=1.0,
                    tree_method="hist",
                    eval_metric="mlogloss",
                    verbosity=0,
                    random_state=45,
                    n_jobs=-1
                )
            )

            weights.append(
                self.xgb_w
            )

        return models, weights

    def analyze(
        self,
        X_train,
        y_train,
        X_next
    ):

        models, weights = (
            self._make_models()
        )

        final = np.zeros(10)

        used_weight = 0.0

        for model, weight in zip(
            models,
            weights
        ):

            if weight <= 0:
                continue

            try:

                model.fit(
                    X_train,
                    y_train
                )

                probs = (
                    model
                    .predict_proba(
                        X_next
                    )[0]
                )

                temp = np.zeros(10)

                for c, p in zip(
                    model.classes_,
                    probs
                ):

                    temp[int(c)] = p

                final += (
                    temp *
                    weight
                )

                used_weight += weight

            except Exception:
                continue

        if used_weight <= 0:
            return np.ones(10) / 10

        final /= used_weight

        total = final.sum()

        if total <= 0:
            return np.ones(10) / 10

        return final / total


# ============================================================
# 11. ENSEMBLE ENGINE
# ============================================================

class EnsembleEngine:

    def __init__(
        self,
        df_raw,
        lottery_name,
        target_dow=None
    ):

        self.df_raw = df_raw.copy()

        self.target_dow = target_dow

        self.lottery_name = (
            lottery_name
        )

        n = len(df_raw)

        # ----------------------------------------------------
        # Adaptive Config
        # ----------------------------------------------------

        if n >= 700:

            self.mode_name = (
                "MODE 4 | 700+ | MAX"
            )

            self.trees = 110
            self.test_size = 30
            self.early_stop = 14

            self.lags = [
                1, 2, 3, 5, 8, 13
            ]

            self.rolls = [
                3, 5, 10, 20
            ]

            self.ai_weights = (
                1.00,
                1.10,
                0.90,
                0.80
            )

        elif n >= 400:

            self.mode_name = (
                "MODE 3 | 400-699 | MAX"
            )

            self.trees = 100
            self.test_size = 25
            self.early_stop = 13

            self.lags = [
                1, 2, 3, 5, 8, 13
            ]

            self.rolls = [
                3, 5, 10, 20
            ]

            self.ai_weights = (
                1.00,
                1.10,
                0.85,
                0.65
            )

        elif n >= 200:

            self.mode_name = (
                "MODE 2 | 200-399 | TURBO"
            )

            self.trees = 85
            self.test_size = 20
            self.early_stop = 11

            self.lags = [
                1, 2, 3, 5, 8
            ]

            self.rolls = [
                3, 5, 10, 20
            ]

            self.ai_weights = (
                1.00,
                1.05,
                0.80,
                0.40
            )

        else:

            self.mode_name = (
                "MODE 1 | 100-199 | FAST"
            )

            self.trees = 65
            self.test_size = 15
            self.early_stop = 9

            self.lags = [
                1, 2, 3, 5
            ]

            self.rolls = [
                3, 5, 10
            ]

            self.ai_weights = (
                1.00,
                1.00,
                0.70,
                0.20
            )

        if n < 100:

            self.test_size = min(
                8,
                max(0, n - 30)
            )

        # ----------------------------------------------------
        # Feature list
        # ----------------------------------------------------

        self.features = [

            "DayOfWeek",
            "Month",
            "Day",
            "DayOfYear",
            "Gap",

            "DOW_SIN",
            "DOW_COS",
            "MONTH_SIN",
            "MONTH_COS",

            "PrevSum3",
            "PrevRange3",
            "PrevOdd3",
            "PrevHigh3",

            "Dist_HT",
            "Dist_TO",
            "Dist_HO"
        ]

        positions = [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]

        for pos in positions:

            self.features.extend([

                f"OddEven_{pos}",
                f"HighLow_{pos}",
                f"IsPrime_{pos}",
                f"Mirror_{pos}",
                f"Skip_{pos}",
                f"Repeat_{pos}"
            ])

            for lag in self.lags:

                self.features.append(
                    f"Lag_{lag}_{pos}"
                )

            for w in self.rolls:

                self.features.extend([
                    f"Roll_{w}_Mean_{pos}",
                    f"Roll_{w}_Std_{pos}"
                ])

            for d in range(10):

                self.features.append(
                    f"Hot20_{pos}_{d}"
                )

        # ----------------------------------------------------
        # Systems
        # ----------------------------------------------------

        self.pos_sys = (
            PositionalEquation()
        )

        self.freq_sys = (
            FrequencyEngine()
        )

        self.cond_sys = (
            ConditionalSystem()
        )

        self.st_sys = (
            StateTransitionSystem()
        )

        self.ptn_sys = (
            PatternBacktestSystem()
        )

        self.ai_sys = AISystem(
            self.trees,
            *self.ai_weights
        )

        # ----------------------------------------------------
        # Base weights
        # ----------------------------------------------------

        self.base_weights = {

            "AI": 0.40,

            "Freq": 0.18,

            "ST": 0.12,

            "Cal": 0.10,

            "BT": 0.12,

            "Eq": 0.08
        }

    # ========================================================
    # BACKTEST
    # ========================================================

    def _process_single_position(
        self,
        pos,
        df_hist,
        X_all,
        next_x,
        next_date
    ):

        bt_size = self.test_size

        # ----------------------------------------------------
        # Default weights
        # ----------------------------------------------------

        if (
            len(df_hist)
            < bt_size + 35
            or
            bt_size <= 0
        ):

            norm_weights = (
                self.base_weights.copy()
            )

            bt_msg = (
                "ข้อมูลน้อย → ใช้น้ำหนักพื้นฐาน"
            )

        else:

            ai_hits = 0.0
            fq_hits = 0.0
            cal_hits = 0.0
            st_hits = 0.0
            ptn_hits = 0.0

            steps_run = 0
            total_decay = 0.0

            # ------------------------------------------------
            # Strict Walk Forward
            # ------------------------------------------------

            start_idx = (
                len(X_all)
                - bt_size
            )

            for i in range(bt_size):

                curr_train_len = (
                    start_idx + i
                )

                if curr_train_len < 35:
                    continue

                # --------------------------------------------
                # Time decay
                # --------------------------------------------

                decay_weight = (
                    1.08 ** i
                )

                total_decay += (
                    decay_weight
                )

                X_train_step = (
                    X_all.iloc[
                        :curr_train_len
                    ]
                )

                y_train_step = (
                    df_hist[pos]
                    .iloc[
                        :curr_train_len
                    ]
                )

                X_test_step = (
                    X_all.iloc[
                        [curr_train_len]
                    ]
                )

                actual_val = int(
                    df_hist[pos]
                    .iloc[
                        curr_train_len
                    ]
                )

                # --------------------------------------------
                # Fast AI proxy
                # --------------------------------------------

                proxy = ExtraTreesClassifier(
                    n_estimators=25,
                    max_depth=6,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=100 + i
                )

                try:

                    proxy.fit(
                        X_train_step,
                        y_train_step
                    )

                    probs = (
                        proxy
                        .predict_proba(
                            X_test_step
                        )[0]
                    )

                    ai_res = np.zeros(10)

                    for c, p in zip(
                        proxy.classes_,
                        probs
                    ):

                        ai_res[int(c)] = p

                    # TOP-3 backtest
                    ai_top3 = np.argsort(
                        ai_res
                    )[::-1][:3]

                    if actual_val in ai_top3:
                        ai_hits += (
                            decay_weight
                        )

                except Exception:
                    pass

                # --------------------------------------------
                # Historical systems
                # --------------------------------------------

                curr_df = (
                    df_hist.iloc[
                        :curr_train_len
                    ]
                )

                target_date = (
                    df_hist[
                        "Date"
                    ].iloc[
                        curr_train_len
                    ]
                )

                fq = self.freq_sys.analyze(
                    curr_df,
                    pos
                )

                cal = self.cond_sys.analyze(
                    curr_df,
                    pos,
                    target_date
                )

                stp = self.st_sys.analyze(
                    curr_df,
                    pos
                )

                ptn = self.ptn_sys.analyze(
                    curr_df,
                    pos
                )

                # TOP-3
                if actual_val in np.argsort(
                    fq
                )[::-1][:3]:

                    fq_hits += (
                        decay_weight
                    )

                if actual_val in np.argsort(
                    cal
                )[::-1][:3]:

                    cal_hits += (
                        decay_weight
                    )

                if actual_val in np.argsort(
                    stp
                )[::-1][:3]:

                    st_hits += (
                        decay_weight
                    )

                if actual_val in np.argsort(
                    ptn
                )[::-1][:3]:

                    ptn_hits += (
                        decay_weight
                    )

                steps_run += 1

                if (
                    steps_run
                    >=
                    self.early_stop
                ):
                    break

            # ------------------------------------------------
            # Dynamic weight
            # ------------------------------------------------

            if (
                steps_run <= 0
                or
                total_decay <= 0
            ):

                norm_weights = (
                    self.base_weights.copy()
                )

                bt_msg = (
                    "Backtest ไม่สามารถทำงานได้"
                )

            else:

                scores = {

                    "AI":
                        ai_hits /
                        total_decay,

                    "Freq":
                        fq_hits /
                        total_decay,

                    "Cal":
                        cal_hits /
                        total_decay,

                    "ST":
                        st_hits /
                        total_decay,

                    "BT":
                        ptn_hits /
                        total_decay
                }

                # ------------------------------------------------
                # Squared performance weighting
                # ------------------------------------------------

                weighted = {}

                for key in scores:

                    # Prevent zeroing
                    score = max(
                        0.08,
                        scores[key]
                    )

                    # stronger separation
                    weighted[key] = (
                        self.base_weights[key]
                        *
                        (
                            0.30
                            +
                            0.70 * score
                        )
                        ** 2
                    )

                # Equation receives fixed small weight
                weighted["Eq"] = (
                    self.base_weights["Eq"]
                    * 0.35
                )

                total = sum(
                    weighted.values()
                )

                norm_weights = {

                    key:
                    value / total

                    for key, value
                    in weighted.items()
                }

                bt_msg = (
                    f"Backtest TOP-3 "
                    f"{steps_run} งวด | "
                    f"AI {scores['AI']:.0%} | "
                    f"Freq {scores['Freq']:.0%} | "
                    f"ST {scores['ST']:.0%} | "
                    f"Cal {scores['Cal']:.0%} | "
                    f"Pattern {scores['BT']:.0%}"
                )

        # ====================================================
        # CURRENT PREDICTION
        # ====================================================

        p_ai = self.ai_sys.analyze(
            X_all,
            df_hist[pos],
            next_x
        )

        p_fq = self.freq_sys.analyze(
            df_hist,
            pos
        )

        p_cal = self.cond_sys.analyze(
            df_hist,
            pos,
            next_date
        )

        p_st = self.st_sys.analyze(
            df_hist,
            pos
        )

        p_bt = self.ptn_sys.analyze(
            df_hist,
            pos
        )

        p_eq = self.pos_sys.analyze(
            df_hist
        )

        W = norm_weights

        # ----------------------------------------------------
        # Final Ensemble
        # ----------------------------------------------------

        final_score = (

            W["AI"] * p_ai

            +

            W["Freq"] * p_fq

            +

            W["Cal"] * p_cal

            +

            W["ST"] * p_st

            +

            W["BT"] * p_bt

            +

            W["Eq"] * p_eq
        )

        total = (
            final_score.sum()
        )

        if total <= 0:

            final_score = (
                np.ones(10) / 10
            )

        else:

            final_score = (
                final_score /
                total
            )

        # ====================================================
        # TOP-3
        # ====================================================

        def get_top3(probs):

            idx = np.argsort(
                probs
            )[::-1][:3]

            return [
                (
                    int(i),
                    float(probs[i])
                )
                for i in idx
            ]

        # ====================================================
        # DEAD TOP-7
        # ====================================================

        def get_dead7(probs):

            idx = np.argsort(
                probs
            )[:7]

            return [
                (
                    int(i),
                    float(probs[i])
                )
                for i in idx
            ]

        return (

            pos,

            {

                "AI":
                    get_top3(p_ai),

                "Calendar":
                    get_top3(p_cal),

                "Frequency":
                    get_top3(p_fq),

                "Final":
                    get_top3(final_score),

                "Dead":
                    get_dead7(final_score),

                "Probs_For_Graph":
                    final_score,

                "Weights":
                    norm_weights,

                "BT_Msg":
                    bt_msg
            }
        )

    # ========================================================
    # PREDICT ALL
    # ========================================================

    def predict_all(self):

        last_date = (
            self.df_raw[
                "Date"
            ].iloc[-1]
        )

        # ----------------------------------------------------
        # Target date
        # ----------------------------------------------------

        if self.target_dow is not None:

            days_ahead = (
                self.target_dow
                -
                last_date.dayofweek
            )

            if days_ahead <= 0:
                days_ahead += 7

            next_date = (
                last_date
                +
                timedelta(
                    days=days_ahead
                )
            )

        else:

            if len(
                self.df_raw
            ) <= 1:

                gap_days = 7

            else:

                gap_days = max(
                    1,
                    (
                        self.df_raw[
                            "Date"
                        ].iloc[-1]
                        -
                        self.df_raw[
                            "Date"
                        ].iloc[-2]
                    ).days
                )

            next_date = (
                last_date
                +
                timedelta(
                    days=gap_days
                )
            )

        # ----------------------------------------------------
        # Dummy next row
        # ----------------------------------------------------

        dummy = pd.DataFrame([
            {
                "Date": next_date,
                "Result_3D": "000",
                "Result_2D": "00"
            }
        ])

        df_ext = pd.concat(
            [
                self.df_raw,
                dummy
            ],
            ignore_index=True
        )

        df_ext = build_features(
            df_ext,
            self.lags,
            self.rolls
        )

        # IMPORTANT:
        # Dummy result is removed before training.
        df_hist = (
            df_ext.iloc[:-1]
            .copy()
        )

        X_all = (
            df_hist[
                self.features
            ]
            .copy()
        )

        next_x = (
            df_ext.iloc[
                [-1]
            ][
                self.features
            ]
            .copy()
        )

        results = []

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            results.append(
                self._process_single_position(
                    pos,
                    df_hist,
                    X_all,
                    next_x,
                    next_date
                )
            )

        return (
            {
                pos: data
                for pos, data
                in results
            },
            next_date
        )


# ============================================================
# 12. HELPER
# ============================================================

def format_nums(items):

    return " • ".join(
        str(num)
        for num, prob
        in items
    )


def format_nums_prob(items):

    return " | ".join(
        f"{num} ({prob:.1%})"
        for num, prob
        in items
    )


def get_hot3_from_positions(
    preds,
    positions
):

    score = np.zeros(10)

    for pos in positions:

        score += (
            preds[pos][
                "Probs_For_Graph"
            ]
        )

    score /= len(positions)

    idx = np.argsort(
        score
    )[::-1][:3]

    return [
        (
            int(i),
            float(score[i])
        )
        for i in idx
    ]


def get_dead7_from_positions(
    preds,
    positions
):

    score = np.zeros(10)

    for pos in positions:

        score += (
            preds[pos][
                "Probs_For_Graph"
            ]
        )

    score /= len(positions)

    idx = np.argsort(
        score
    )[:7]

    return [
        (
            int(i),
            float(score[i])
        )
        for i in idx
    ]


# ============================================================
# 13. UI HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚀 LOTTO AI ULTIMATE V.MAX'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Sequential Draw-to-Draw • TOP-3 Optimized '
    '• Time-Decay • Dynamic Ensemble'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 14. CONTROL
# ============================================================

col1, col2 = st.columns(2)

with col1:

    selected_lotto = st.selectbox(
        "🎯 เลือกหวย",
        list(
            LOTTERY_SOURCES.keys()
        ),
        index=0
    )

with col2:

    day_options = {

        "อัตโนมัติ":
            None,

        "วันจันทร์":
            0,

        "วันอังคาร":
            1,

        "วันพุธ":
            2,

        "วันพฤหัสบดี":
            3,

        "วันศุกร์":
            4,

        "วันเสาร์":
            5,

        "วันอาทิตย์":
            6
    }

    selected_day_label = (
        st.selectbox(
            "📅 วันออกรางวัล",
            list(
                day_options.keys()
            )
        )
    )

    target_dow = (
        day_options[
            selected_day_label
        ]
    )


# ============================================================
# 15. RUN
# ============================================================

if st.button(
    "🚀 วิเคราะห์ TOP-3",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "⏳ กำลังดึงข้อมูล + "
        "Walk-Forward Backtest + "
        "AI Ensemble..."
    ):

        url = (
            LOTTERY_SOURCES[
                selected_lotto
            ]
        )

        df_raw = (
            fetch_and_clean_data(
                url
            )
        )

        if df_raw.empty:
            st.stop()

        # ----------------------------------------------------
        # Engine
        # ----------------------------------------------------

        engine = EnsembleEngine(
            df_raw,
            selected_lotto,
            target_dow=target_dow
        )

        # ----------------------------------------------------
        # System status
        # ----------------------------------------------------

        st.success(
            f"✅ โหลดข้อมูลสำเร็จ "
            f"{len(df_raw):,} งวด"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "📚 ข้อมูล",
                f"{len(df_raw):,}"
            )

        with c2:
            st.metric(
                "🤖 AI Trees",
                engine.trees
            )

        with c3:
            st.metric(
                "🎯 Output",
                "TOP-3"
            )

        st.caption(
            f"⚙️ {engine.mode_name}"
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        preds, next_date = (
            engine.predict_all()
        )

        dow_names = [
            "จันทร์",
            "อังคาร",
            "พุธ",
            "พฤหัสบดี",
            "ศุกร์",
            "เสาร์",
            "อาทิตย์"
        ]

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

        st.divider()

        st.subheader(
            "🔮 ผลการวิเคราะห์"
        )

        st.info(
            f"📅 งวดเป้าหมาย: "
            f"วัน{dow_names[next_date.dayofweek]} "
            f"{next_date.strftime('%d-%m-%Y')}"
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

            final_top3 = (
                preds[pos]["Final"]
            )

            ai_top3 = (
                preds[pos]["AI"]
            )

            cal_top3 = (
                preds[pos]["Calendar"]
            )

            stat_top3 = (
                preds[pos]["Frequency"]
            )

            dead7 = (
                preds[pos]["Dead"]
            )

            W = (
                preds[pos]["Weights"]
            )

            st.markdown(
                f"### 📍 {labels[pos]}"
            )

            # ------------------------------------------------
            # Final TOP 3
            # ------------------------------------------------

            st.success(
                f"🔥 **HOT TOP-3:**  "
                f"`{format_nums(final_top3)}`"
            )

            st.caption(
                "ความน่าจะเป็น Ensemble: "
                +
                format_nums_prob(
                    final_top3
                )
            )

            # ------------------------------------------------
            # AI
            # ------------------------------------------------

            st.markdown(
                f"🤖 **AI TOP-3:** "
                f"`{format_nums(ai_top3)}`"
            )

            # ------------------------------------------------
            # Other systems
            # ------------------------------------------------

            st.markdown(
                f"📊 **สถิติ TOP-3:** "
                f"`{format_nums(stat_top3)}`"
            )

            st.markdown(
                f"📅 **กำลังวัน TOP-3:** "
                f"`{format_nums(cal_top3)}`"
            )

            # ------------------------------------------------
            # Dead
            # ------------------------------------------------

            st.warning(
                f"❄️ **DEAD TOP-7:** "
                f"`{format_nums(dead7)}`"
            )

            # ------------------------------------------------
            # Backtest
            # ------------------------------------------------

            st.caption(
                "📈 "
                +
                preds[pos]["BT_Msg"]
            )

            st.caption(
                "⚖️ Weight → "
                f"AI {W['AI']:.0%} | "
                f"Freq {W['Freq']:.0%} | "
                f"ST {W['ST']:.0%} | "
                f"Cal {W['Cal']:.0%} | "
                f"BT {W['BT']:.0%} | "
                f"Eq {W['Eq']:.0%}"
            )

            st.divider()

        # ====================================================
        # HOT TOP 3 SUMMARY
        # ====================================================

        hot_top3 = (
            get_hot3_from_positions(
                preds,
                [
                    "H",
                    "T",
                    "O"
                ]
            )
        )

        hot_bottom3 = (
            get_hot3_from_positions(
                preds,
                [
                    "T2",
                    "O2"
                ]
            )
        )

        dead_top7 = (
            get_dead7_from_positions(
                preds,
                [
                    "H",
                    "T",
                    "O"
                ]
            )
        )

        dead_bottom7 = (
            get_dead7_from_positions(
                preds,
                [
                    "T2",
                    "O2"
                ]
            )
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        st.subheader(
            "🔥 สรุปฟันธง"
        )

        st.markdown(
            '<div class="hot-box">'
            '<div class="small-label">'
            '🔥 HOT TOP-3 เด่นบนรวม'
            '</div>'
            f'<div class="big-number">'
            f'{format_nums(hot_top3)}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="hot-box">'
            '<div class="small-label">'
            '🔥 HOT TOP-3 เด่นล่างรวม'
            '</div>'
            f'<div class="big-number">'
            f'{format_nums(hot_bottom3)}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="dead-box">'
            '<div class="small-label">'
            '❄️ DEAD TOP-7 บน'
            '</div>'
            f'<div class="big-number">'
            f'{format_nums(dead_top7)}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="dead-box">'
            '<div class="small-label">'
            '❄️ DEAD TOP-7 ล่าง'
            '</div>'
            f'<div class="big-number">'
            f'{format_nums(dead_bottom7)}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # GRAPH
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Probability TOP-3"
        )

        positions = [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 7)
        )

        axes = axes.flatten()

        for idx, pos in enumerate(
            positions
        ):

            ax = axes[idx]

            top3 = (
                preds[pos]["Final"]
            )

            numbers = [
                str(x[0])
                for x in top3
            ]

            probabilities = [
                x[1] * 100
                for x in top3
            ]

            ax.bar(
                numbers,
                probabilities
            )

            ax.set_title(
                labels[pos],
                fontsize=10
            )

            ax.set_ylabel(
                "%"
            )

            max_prob = (
                max(
                    probabilities
                )
                if probabilities
                else 1
            )

            ax.set_ylim(
                0,
                max_prob * 1.30
            )

            for i, value in enumerate(
                probabilities
            ):

                ax.text(
                    i,
                    value,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

        fig.delaxes(
            axes[5]
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # ====================================================
        # DATA INFO
        # ====================================================

        st.divider()

        st.caption(
            "🛡️ ระบบใช้ Strict Walk-Forward "
            "Backtest และไม่ใช้ผลของงวดเป้าหมาย "
            "ในการฝึกโมเดล"
        )

        st.caption(
            "🔄 เมื่อกดวิเคราะห์ ระบบจะคำนวณ "
            "ใหม่จากข้อมูลล่าสุด"
        )

        if not XGB_AVAILABLE:

            st.warning(
                "ℹ️ ไม่พบ XGBoost — ระบบใช้ "
                "RandomForest + ExtraTrees + "
                "HistGradientBoosting แทน"
                        )
