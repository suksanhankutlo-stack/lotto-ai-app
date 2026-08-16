# ============================================================
# 🚀 LOTTO AI ULTIMATE V.MAX 3-TOP TURBO
# ============================================================
# TOP-3 ONLY
# NO DEAD NUMBER
# STRICT WALK-FORWARD
# LEAKAGE SAFE
# TIME-DECAY
# DYNAMIC WEIGHT
# TURBO AI
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

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================
# 0. STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Lotto AI V.MAX 3-TOP TURBO",
    page_icon="🚀",
    layout="centered"
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
# 2. UI STYLE
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align:center;
    font-size:27px;
    font-weight:800;
}

.sub-title {
    text-align:center;
    color:#777;
    font-size:13px;
}

.hot-card {
    padding:14px;
    border-radius:14px;
    border:2px solid #ff9800;
    margin:8px 0;
}

.hot-number {
    font-size:28px;
    font-weight:800;
    letter-spacing:3px;
}

.position {
    font-size:18px;
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. FETCH DATA
# ============================================================

@st.cache_data(ttl=180, show_spinner=False)
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

        lines = (
            main
            .get_text(separator="\n")
            .split("\n")
        )

        date_pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2}|"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        )

        num_pattern = re.compile(
            r"\b(\d{3})\b.*?\b(\d{2})\b|"
            r"\b(\d{5,6})\b.*?\b(\d{2})\b"
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

        df = (
            df.sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except Exception as e:

        st.error(
            f"❌ ดึงข้อมูลไม่ได้: {e}"
        )

        return pd.DataFrame()


# ============================================================
# 4. FAST FEATURE ENGINEERING
# ============================================================

def build_features(
    df,
    lags,
    rolls
):

    x = df.copy()

    # --------------------------------------------------------
    # DIGITS
    # --------------------------------------------------------

    r3 = x["Result_3D"].astype(str)
    r2 = x["Result_2D"].astype(str)

    x["H"] = r3.str[0].astype(np.int8)
    x["T"] = r3.str[1].astype(np.int8)
    x["O"] = r3.str[2].astype(np.int8)

    x["T2"] = r2.str[0].astype(np.int8)
    x["O2"] = r2.str[1].astype(np.int8)

    # --------------------------------------------------------
    # CALENDAR
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

    # --------------------------------------------------------
    # PREVIOUS DRAW
    # --------------------------------------------------------

    ph = x["H"].shift(1)
    pt = x["T"].shift(1)
    po = x["O"].shift(1)

    x["PrevSum"] = (
        ph + pt + po
    )

    x["PrevRange"] = (
        pd.concat(
            [ph, pt, po],
            axis=1
        ).max(axis=1)
        -
        pd.concat(
            [ph, pt, po],
            axis=1
        ).min(axis=1)
    )

    x["PrevOdd"] = (
        (ph % 2)
        +
        (pt % 2)
        +
        (po % 2)
    )

    x["PrevHigh"] = (
        (ph >= 5).astype(int)
        +
        (pt >= 5).astype(int)
        +
        (po >= 5).astype(int)
    )

    x["DistHT"] = (
        ph - pt
    ).abs()

    x["DistTO"] = (
        pt - po
    ).abs()

    # --------------------------------------------------------
    # POSITION FEATURES
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

        x[f"Odd_{pos}"] = (
            prev % 2
        )

        x[f"High_{pos}"] = (
            prev >= 5
        ).astype(np.int8)

        x[f"Prime_{pos}"] = (
            prev.isin(
                [2, 3, 5, 7]
            )
        ).astype(np.int8)

        x[f"Mirror_{pos}"] = (
            (prev + 5) % 10
        )

        # ----------------------------------------------------
        # LAGS
        # ----------------------------------------------------

        for lag in lags:

            x[
                f"L{lag}_{pos}"
            ] = s.shift(lag)

        # ----------------------------------------------------
        # ROLLING
        # ----------------------------------------------------

        for w in rolls:

            shifted = s.shift(1)

            x[
                f"RM{w}_{pos}"
            ] = (
                shifted
                .rolling(
                    w,
                    min_periods=1
                )
                .mean()
            )

        # ----------------------------------------------------
        # REPEAT
        # ----------------------------------------------------

        x[
            f"Repeat_{pos}"
        ] = (
            s.shift(1)
            ==
            s.shift(2)
        ).astype(np.int8)

        # ----------------------------------------------------
        # RECENT HOT COUNT
        # ----------------------------------------------------

        for d in range(10):

            x[
                f"Hot_{pos}_{d}"
            ] = (
                s.shift(1)
                .eq(d)
                .rolling(
                    15,
                    min_periods=1
                )
                .sum()
            )

        # ----------------------------------------------------
        # SKIP
        # ----------------------------------------------------

        arr = s.to_numpy()

        skip = np.zeros(
            len(arr),
            dtype=np.float32
        )

        last = np.full(
            10,
            -1,
            dtype=np.int32
        )

        for i, val in enumerate(arr):

            v = int(val)

            if last[v] < 0:

                skip[i] = i

            else:

                skip[i] = (
                    i - last[v]
                )

            last[v] = i

        x[
            f"Skip_{pos}"
        ] = skip

    x = x.replace(
        [np.inf, -np.inf],
        np.nan
    )

    x = x.fillna(-1)

    return x


# ============================================================
# 5. FAST FREQUENCY
# ============================================================

class FrequencyEngine:

    def analyze(
        self,
        df,
        pos
    ):

        s = (
            df[pos]
            .astype(int)
        )

        if len(s) == 0:
            return np.ones(10) / 10

        recent10 = (
            s.tail(10)
            .value_counts(
                normalize=True
            )
        )

        recent20 = (
            s.tail(20)
            .value_counts(
                normalize=True
            )
        )

        allfreq = (
            s.value_counts(
                normalize=True
            )
        )

        score = np.zeros(10)

        for d in range(10):

            score[d] = (

                recent10.get(
                    d, 0
                ) * 0.50

                +

                recent20.get(
                    d, 0
                ) * 0.30

                +

                allfreq.get(
                    d, 0
                ) * 0.20
            )

        score += 0.01

        return (
            score /
            score.sum()
        )


# ============================================================
# 6. CALENDAR
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

        if len(subset) < 5:
            subset = df

        recent = subset.tail(20)

        a = (
            subset[pos]
            .value_counts(
                normalize=True
            )
        )

        b = (
            recent[pos]
            .value_counts(
                normalize=True
            )
        )

        score = np.array([
            a.get(d, 0) * 0.4
            +
            b.get(d, 0) * 0.6
            for d in range(10)
        ])

        score += 0.01

        return (
            score /
            score.sum()
        )


# ============================================================
# 7. TRANSITION
# ============================================================

class TransitionEngine:

    def analyze(
        self,
        df,
        pos
    ):

        if len(df) < 6:
            return np.ones(10) / 10

        last = int(
            df[pos].iloc[-1]
        )

        prev = df[pos].shift(1)

        mask = (
            prev == last
        )

        subset = df[
            mask
        ]

        if len(subset) < 2:
            return np.ones(10) / 10

        freq = (
            subset[pos]
            .value_counts(
                normalize=True
            )
        )

        score = np.array([
            freq.get(d, 0)
            for d in range(10)
        ])

        score += 0.01

        return (
            score /
            score.sum()
        )


# ============================================================
# 8. PATTERN
# ============================================================

class PatternEngine:

    def analyze(
        self,
        df,
        pos
    ):

        if len(df) < 7:
            return np.ones(10) / 10

        a = int(
            df[pos].iloc[-1]
        )

        b = int(
            df[pos].iloc[-2]
        )

        s1 = df[pos].shift(1)
        s2 = df[pos].shift(2)

        subset = df[
            (s1 == a)
            &
            (s2 == b)
        ]

        if len(subset) < 2:

            subset = df[
                s1 == a
            ]

        if len(subset) < 1:
            return np.ones(10) / 10

        freq = (
            subset[pos]
            .value_counts(
                normalize=True
            )
        )

        score = np.array([
            freq.get(d, 0)
            for d in range(10)
        ])

        score += 0.01

        return (
            score /
            score.sum()
        )


# ============================================================
# 9. EQUATION
# ============================================================

class EquationEngine:

    def analyze(self, df):

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
            10
        ) * 0.05

        for v in vals:
            score[v] += 1

        return (
            score /
            score.sum()
        )


# ============================================================
# 10. FAST AI
# ============================================================

class FastAI:

    def __init__(
        self,
        trees,
        weights
    ):

        self.trees = trees
        self.weights = weights

    def predict(
        self,
        X,
        y,
        X_next
    ):

        rf_w, et_w, hgb_w = (
            self.weights
        )

        result = np.zeros(10)

        total_w = 0

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        if rf_w > 0:

            model = RandomForestClassifier(

                n_estimators=self.trees,

                max_depth=6,

                min_samples_leaf=2,

                max_features="sqrt",

                class_weight="balanced",

                n_jobs=-1,

                random_state=42
            )

            model.fit(
                X,
                y
            )

            p = model.predict_proba(
                X_next
            )[0]

            for c, prob in zip(
                model.classes_,
                p
            ):

                result[
                    int(c)
                ] += (
                    prob * rf_w
                )

            total_w += rf_w

        # ----------------------------------------------------
        # EXTRA TREES
        # ----------------------------------------------------

        if et_w > 0:

            model = ExtraTreesClassifier(

                n_estimators=self.trees,

                max_depth=7,

                min_samples_leaf=2,

                max_features="sqrt",

                class_weight="balanced",

                n_jobs=-1,

                random_state=43
            )

            model.fit(
                X,
                y
            )

            p = model.predict_proba(
                X_next
            )[0]

            for c, prob in zip(
                model.classes_,
                p
            ):

                result[
                    int(c)
                ] += (
                    prob * et_w
                )

            total_w += et_w

        # ----------------------------------------------------
        # HGB
        # ----------------------------------------------------

        if hgb_w > 0:

            model = HistGradientBoostingClassifier(

                max_iter=55,

                learning_rate=0.05,

                max_leaf_nodes=12,

                min_samples_leaf=5,

                l2_regularization=0.15,

                random_state=44
            )

            model.fit(
                X,
                y
            )

            p = model.predict_proba(
                X_next
            )[0]

            for c, prob in zip(
                model.classes_,
                p
            ):

                result[
                    int(c)
                ] += (
                    prob * hgb_w
                )

            total_w += hgb_w

        if total_w <= 0:
            return np.ones(10) / 10

        result /= total_w

        if result.sum() <= 0:
            return np.ones(10) / 10

        return (
            result /
            result.sum()
        )


# ============================================================
# 11. ENSEMBLE ENGINE
# ============================================================

class EnsembleEngine:

    def __init__(
        self,
        df,
        lottery_name,
        target_dow=None
    ):

        self.df = df.copy()

        self.lottery_name = (
            lottery_name
        )

        self.target_dow = (
            target_dow
        )

        n = len(df)

        # ----------------------------------------------------
        # TURBO CONFIG
        # ----------------------------------------------------

        if n >= 700:

            self.mode = (
                "700+ TURBO MAX"
            )

            self.trees = 65
            self.bt = 10
            self.lags = [
                1, 2, 3, 5, 8
            ]

            self.rolls = [
                3, 5, 10
            ]

        elif n >= 400:

            self.mode = (
                "400-699 TURBO"
            )

            self.trees = 55
            self.bt = 9

            self.lags = [
                1, 2, 3, 5, 8
            ]

            self.rolls = [
                3, 5, 10
            ]

        elif n >= 200:

            self.mode = (
                "200-399 FAST"
            )

            self.trees = 45
            self.bt = 8

            self.lags = [
                1, 2, 3, 5
            ]

            self.rolls = [
                3, 5, 10
            ]

        else:

            self.mode = (
                "100-199 FAST"
            )

            self.trees = 35
            self.bt = 7

            self.lags = [
                1, 2, 3
            ]

            self.rolls = [
                3, 5
            ]

        # ----------------------------------------------------
        # FEATURE LIST
        # ----------------------------------------------------

        self.features = [

            "DOW",
            "Month",
            "Day",
            "Gap",

            "DOW_SIN",
            "DOW_COS",

            "PrevSum",
            "PrevRange",
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
                f"Mirror_{pos}",
                f"Repeat_{pos}",
                f"Skip_{pos}"
            ])

            for lag in self.lags:

                self.features.append(
                    f"L{lag}_{pos}"
                )

            for w in self.rolls:

                self.features.append(
                    f"RM{w}_{pos}"
                )

            for d in range(10):

                self.features.append(
                    f"Hot_{pos}_{d}"
                )

        # ----------------------------------------------------
        # SYSTEMS
        # ----------------------------------------------------

        self.freq = (
            FrequencyEngine()
        )

        self.calendar = (
            CalendarEngine()
        )

        self.transition = (
            TransitionEngine()
        )

        self.pattern = (
            PatternEngine()
        )

        self.equation = (
            EquationEngine()
        )

        self.ai = FastAI(
            self.trees,
            (
                0.35,
                0.45,
                0.20
            )
        )

        self.base_weights = {

            "AI": 0.46,

            "Freq": 0.18,

            "ST": 0.12,

            "Cal": 0.10,

            "BT": 0.08,

            "Eq": 0.06
        }

    # ========================================================
    # FAST BACKTEST
    # ========================================================

    def backtest(
        self,
        pos,
        X,
        df_hist
    ):

        n = len(X)

        if n < 45:

            return (
                self.base_weights.copy(),
                "Backtest ข้อมูลน้อย"
            )

        start = max(
            35,
            n - self.bt
        )

        scores = {
            "AI": 0.0,
            "Freq": 0.0,
            "ST": 0.0,
            "Cal": 0.0,
            "BT": 0.0
        }

        total_decay = 0.0

        # ----------------------------------------------------
        # Only lightweight ExtraTrees proxy
        # ----------------------------------------------------

        for step, idx in enumerate(
            range(start, n)
        ):

            decay = (
                1.10 ** step
            )

            total_decay += decay

            Xtr = X.iloc[:idx]

            ytr = (
                df_hist[pos]
                .iloc[:idx]
            )

            xt = X.iloc[
                [idx]
            ]

            actual = int(
                df_hist[pos]
                .iloc[idx]
            )

            # -----------------------------------------------
            # AI Proxy
            # -----------------------------------------------

            try:

                proxy = ExtraTreesClassifier(

                    n_estimators=12,

                    max_depth=5,

                    min_samples_leaf=2,

                    max_features="sqrt",

                    n_jobs=-1,

                    random_state=200 + step
                )

                proxy.fit(
                    Xtr,
                    ytr
                )

                p = proxy.predict_proba(
                    xt
                )[0]

                tmp = np.zeros(10)

                for c, prob in zip(
                    proxy.classes_,
                    p
                ):

                    tmp[int(c)] = prob

                if actual in np.argsort(
                    tmp
                )[::-1][:3]:

                    scores["AI"] += decay

            except Exception:
                pass

            # -----------------------------------------------
            # Other systems
            # -----------------------------------------------

            hist = (
                df_hist.iloc[:idx]
                .copy()
            )

            target_date = (
                df_hist["Date"]
                .iloc[idx]
            )

            f = self.freq.analyze(
                hist,
                pos
            )

            c = self.calendar.analyze(
                hist,
                pos,
                target_date
            )

            s = self.transition.analyze(
                hist,
                pos
            )

            b = self.pattern.analyze(
                hist,
                pos
            )

            if actual in np.argsort(
                f
            )[::-1][:3]:

                scores["Freq"] += decay

            if actual in np.argsort(
                c
            )[::-1][:3]:

                scores["Cal"] += decay

            if actual in np.argsort(
                s
            )[::-1][:3]:

                scores["ST"] += decay

            if actual in np.argsort(
                b
            )[::-1][:3]:

                scores["BT"] += decay

        if total_decay <= 0:

            return (
                self.base_weights.copy(),
                "Backtest error"
            )

        accuracy = {
            k:
            v / total_decay
            for k, v in scores.items()
        }

        # ----------------------------------------------------
        # Dynamic weight
        # ----------------------------------------------------

        weighted = {}

        for k in accuracy:

            acc = max(
                0.10,
                accuracy[k]
            )

            weighted[k] = (
                self.base_weights[k]
                *
                (
                    0.35
                    +
                    0.65 * acc
                )
                ** 2
            )

        weighted["Eq"] = (
            self.base_weights["Eq"]
            * 0.35
        )

        total = sum(
            weighted.values()
        )

        weights = {
            k: v / total
            for k, v
            in weighted.items()
        }

        msg = (
            f"WF Top-3 {self.bt} งวด | "
            f"AI {accuracy['AI']:.0%} | "
            f"Freq {accuracy['Freq']:.0%} | "
            f"ST {accuracy['ST']:.0%} | "
            f"Cal {accuracy['Cal']:.0%}"
        )

        return (
            weights,
            msg
        )

    # ========================================================
    # POSITION
    # ========================================================

    def process_position(
        self,
        pos,
        hist,
        X,
        X_next,
        next_date
    ):

        weights, bt_msg = (
            self.backtest(
                pos,
                X,
                hist
            )
        )

        # ----------------------------------------------------
        # Current prediction
        # ----------------------------------------------------

        ai = self.ai.predict(
            X,
            hist[pos],
            X_next
        )

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

        final = (

            weights["AI"] * ai

            +

            weights["Freq"] * fq

            +

            weights["Cal"] * cal

            +

            weights["ST"] * stp

            +

            weights["BT"] * ptn

            +

            weights["Eq"] * eq
        )

        final = (
            final /
            final.sum()
        )

        top3_idx = np.argsort(
            final
        )[::-1][:3]

        top3 = [
            (
                int(i),
                float(final[i])
            )
            for i in top3_idx
        ]

        return {

            "Final": top3,

            "AI": self.top3(ai),

            "Freq": self.top3(fq),

            "Calendar": self.top3(cal),

            "Prob": final,

            "Weights": weights,

            "BT": bt_msg
        }

    # ========================================================
    # TOP 3
    # ========================================================

    @staticmethod
    def top3(p):

        idx = np.argsort(
            p
        )[::-1][:3]

        return [
            (
                int(i),
                float(p[i])
            )
            for i in idx
        ]

    # ========================================================
    # PREDICT
    # ========================================================

    def predict_all(self):

        last_date = (
            self.df[
                "Date"
            ].iloc[-1]
        )

        if self.target_dow is not None:

            days = (
                self.target_dow
                -
                last_date.dayofweek
            )

            if days <= 0:
                days += 7

        else:

            if len(self.df) >= 2:

                days = max(
                    1,
                    (
                        self.df[
                            "Date"
                        ].iloc[-1]
                        -
                        self.df[
                            "Date"
                        ].iloc[-2]
                    ).days
                )

            else:

                days = 7

        next_date = (
            last_date
            +
            timedelta(
                days=days
            )
        )

        dummy = pd.DataFrame([{

            "Date": next_date,

            "Result_3D": "000",

            "Result_2D": "00"
        }])

        ext = pd.concat(
            [
                self.df,
                dummy
            ],
            ignore_index=True
        )

        ext = build_features(
            ext,
            self.lags,
            self.rolls
        )

        # ----------------------------------------------------
        # STRICT:
        # remove dummy result
        # ----------------------------------------------------

        hist = (
            ext.iloc[:-1]
            .copy()
        )

        X = (
            hist[
                self.features
            ]
            .astype(np.float32)
        )

        X_next = (
            ext.iloc[
                [-1]
            ][
                self.features
            ]
            .astype(np.float32)
        )

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
# 12. FORMAT
# ============================================================

def nums(items):

    return " • ".join(
        str(n)
        for n, p in items
    )


def nums_prob(items):

    return " | ".join(
        f"{n} ({p:.1%})"
        for n, p in items
    )


def combine_top3(
    preds,
    positions
):

    score = np.zeros(10)

    for pos in positions:

        score += (
            preds[pos]["Prob"]
        )

    score /= len(
        positions
    )

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


# ============================================================
# 13. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚀 LOTTO AI V.MAX 3-TOP TURBO'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Strict Walk-Forward • Time-Decay • '
    'Dynamic Ensemble • TOP-3 Only'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 14. SELECT
# ============================================================

c1, c2 = st.columns(2)

with c1:

    selected_lotto = st.selectbox(
        "🎯 เลือกหวย",
        list(
            LOTTERY_SOURCES.keys()
        )
    )

with c2:

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

    day_label = st.selectbox(
        "📅 วันออกรางวัล",
        list(
            day_options.keys()
        )
    )

    target_dow = (
        day_options[day_label]
    )


# ============================================================
# 15. RUN
# ============================================================

if st.button(
    "🚀 วิเคราะห์เลขเด่น TOP-3",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "⚡ Turbo AI กำลังคำนวณ..."
    ):

        df = fetch_and_clean_data(
            LOTTERY_SOURCES[
                selected_lotto
            ]
        )

        if df.empty:
            st.stop()

        engine = EnsembleEngine(
            df,
            selected_lotto,
            target_dow
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "📚 งวด",
                f"{len(df):,}"
            )

        with b:
            st.metric(
                "🌲 Trees",
                engine.trees
            )

        with c:
            st.metric(
                "🎯 Output",
                "TOP-3"
            )

        st.caption(
            f"⚡ {engine.mode}"
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        preds, next_date = (
            engine.predict_all()
        )

        days = [
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
            "🔮 ผลวิเคราะห์"
        )

        st.info(
            f"📅 งวดเป้าหมาย "
            f"วัน{days[next_date.dayofweek]} "
            f"{next_date.strftime('%d-%m-%Y')}"
        )

        # ====================================================
        # POSITION OUTPUT
        # ====================================================

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            result = preds[pos]

            st.markdown(
                f'<div class="position">'
                f'📍 {labels[pos]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="hot-card">'
                '<div>🔥 HOT TOP-3</div>'
                f'<div class="hot-number">'
                f'{nums(result["Final"])}'
                f'</div>'
                f'<div>{nums_prob(result["Final"])}'
                f'</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"🤖 AI: `{nums(result['AI'])}`"
            )

            st.markdown(
                f"📊 สถิติ: `{nums(result['Freq'])}`"
            )

            st.markdown(
                f"📅 วัน: `{nums(result['Calendar'])}`"
            )

            st.caption(
                "📈 "
                +
                result["BT"]
            )

            W = result["Weights"]

            st.caption(
                f"⚖️ AI {W['AI']:.0%} | "
                f"Freq {W['Freq']:.0%} | "
                f"ST {W['ST']:.0%} | "
                f"Cal {W['Cal']:.0%} | "
                f"BT {W['BT']:.0%}"
            )

            st.divider()

        # ====================================================
        # GLOBAL HOT TOP 3
        # ====================================================

        hot_top = combine_top3(
            preds,
            [
                "H",
                "T",
                "O"
            ]
        )

        hot_bottom = combine_top3(
            preds,
            [
                "T2",
                "O2"
            ]
        )

        st.subheader(
            "🔥 สรุปเลขเด่น"
        )

        st.markdown(
            '<div class="hot-card">'
            '<div>🔥 HOT TOP-3 บนรวม</div>'
            f'<div class="hot-number">'
            f'{nums(hot_top)}'
            f'</div>'
            f'<div>{nums_prob(hot_top)}'
            f'</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="hot-card">'
            '<div>🔥 HOT TOP-3 ล่างรวม</div>'
            f'<div class="hot-number">'
            f'{nums(hot_bottom)}'
            f'</div>'
            f'<div>{nums_prob(hot_bottom)}'
            f'</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # GRAPH
        # ====================================================

        st.subheader(
            "📊 ความน่าจะเป็น TOP-3"
        )

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(9, 6)
        )

        axes = axes.flatten()

        for i, pos in enumerate([
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]):

            ax = axes[i]

            top3 = (
                preds[pos]["Final"]
            )

            x = [
                str(n)
                for n, p
                in top3
            ]

            y = [
                p * 100
                for n, p
                in top3
            ]

            ax.bar(
                x,
                y
            )

            ax.set_title(
                labels[pos],
                fontsize=9
            )

            ax.set_ylabel(
                "%"
            )

            for j, val in enumerate(y):

                ax.text(
                    j,
                    val,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

            ax.set_ylim(
                0,
                max(y) * 1.3
                if y
                else 1
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
        # FOOTER
        # ====================================================

        st.success(
            "✅ วิเคราะห์เสร็จแล้ว • "
            "แสดงเฉพาะ HOT TOP-3"
        )

        st.caption(
            "🛡️ Strict Walk-Forward: "
            "ไม่ใช้ผลของงวดเป้าหมายในการฝึก"
        )

        st.caption(
            "⚡ Turbo Mode: "
            "ลดรอบ Backtest และลด Trees "
            "เพื่อให้ประมวลผลเร็วขึ้น"
            )
