# ==============================================================================
# 🛑 LOTTO AI PRO V5.0 TURBO MAX
# STRICT CAUSAL • LEAKAGE SAFE • FAST WALK-FORWARD
# CANDIDATE ELIMINATION TOP-7
# RANDOM FOREST + EXTRA TREES + HIST GRADIENT BOOSTING
# ==============================================================================

import streamlit as st
import requests
import warnings
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np

from datetime import timedelta

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. STREAMLIT SETUP
# ==============================================================================

st.set_page_config(
    page_title="ระบบวิเคราะห์เลขดับ PRO V5.0 TURBO MAX",
    page_icon="🛑",
    layout="centered"
)

st.markdown("""
<style>

.main-title {
    text-align:center;
    font-size:31px;
    font-weight:900;
    color:#B71C1C;
    margin-bottom:5px;
}

.sub-title {
    text-align:center;
    color:#555;
    font-size:14px;
    margin-bottom:20px;
}

.dead-card {
    background:linear-gradient(135deg,#ffffff,#fff5f5);
    border-left:6px solid #D32F2F;
    padding:18px;
    border-radius:12px;
    margin-bottom:18px;
    box-shadow:0 3px 9px rgba(0,0,0,.08);
}

.position-title {
    font-size:19px;
    font-weight:800;
    color:#333;
    border-bottom:2px solid #eee;
    padding-bottom:7px;
    margin-bottom:10px;
}

.dead-number-highlight {
    font-size:30px;
    font-weight:900;
    color:#D32F2F;
    letter-spacing:3px;
    text-align:center;
    margin:10px 0;
}

.info-row {
    margin:7px 0;
    font-size:13px;
}

.badge {
    padding:3px 9px;
    border-radius:15px;
    font-size:13px;
    font-weight:700;
}

.badge-ai {
    background:#E3F2FD;
    color:#1565C0;
}

.badge-stat {
    background:#E8F5E9;
    color:#2E7D32;
}

.badge-day {
    background:#FFF3E0;
    color:#E65100;
}

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
    "หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
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

        response = requests.get(
            url,
            headers={
                "User-Agent":
                "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36"
            },
            timeout=8
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.content,
            "html.parser"
        )

        post_body = (
            soup.find(
                "div",
                class_=re.compile(
                    r"post-body|entry-content|post-content|content"
                )
            )
            or soup
        )

        text = post_body.get_text(separator="\n")

        pattern = re.compile(
            r"\*\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d+)\s*\|\s*(\d{2})"
        )

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
                "bot_unit": int(bot[1])
            })

        df = pd.DataFrame(data)

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = (
            df.dropna(subset=["date"])
              .drop_duplicates(
                  subset=["date"],
                  keep="last"
              )
              .sort_values("date")
              .reset_index(drop=True)
        )

        return df

    except Exception as e:

        st.error(f"❌ ดึงข้อมูลไม่สำเร็จ: {e}")
        return None


# ==============================================================================
# 3. ADAPTIVE CONFIG
# ==============================================================================

def get_adaptive_config(n):

    if n >= 700:

        return {
            "mode": "TURBO MAX 700+",
            "trees": 65,
            "max_depth": 7,
            "leaf": 2,
            "bt_steps": 8,
            "min_train": 45,
            "lags": [1, 2, 3, 5, 8],
            "rolls": [3, 5, 10]
        }

    elif n >= 400:

        return {
            "mode": "TURBO MAX 400-699",
            "trees": 55,
            "max_depth": 7,
            "leaf": 2,
            "bt_steps": 7,
            "min_train": 40,
            "lags": [1, 2, 3, 5, 8],
            "rolls": [3, 5, 10]
        }

    elif n >= 200:

        return {
            "mode": "TURBO 200-399",
            "trees": 45,
            "max_depth": 6,
            "leaf": 2,
            "bt_steps": 6,
            "min_train": 35,
            "lags": [1, 2, 3, 5],
            "rolls": [3, 5, 10]
        }

    else:

        return {
            "mode": "TURBO 30-199",
            "trees": 35,
            "max_depth": 5,
            "leaf": 2,
            "bt_steps": 5,
            "min_train": 30,
            "lags": [1, 2, 3],
            "rolls": [3, 5]
        }


# ==============================================================================
# 4. FAST CAUSAL FEATURE ENGINEERING
# ==============================================================================

@st.cache_data(show_spinner=False)
def build_features_cached(
    df,
    target_col,
    lags,
    rolls
):

    x = df.copy()

    target = pd.to_numeric(
        x[target_col],
        errors="coerce"
    )

    # --------------------------------------------------------------------------
    # Previous value ONLY
    # --------------------------------------------------------------------------

    prev = target.shift(1)

    x["prev_val"] = prev

    x["prev_even"] = (
        (prev % 2 == 0)
        .astype(np.int8)
    )

    x["prev_high"] = (
        (prev >= 5)
        .astype(np.int8)
    )

    # Mirror of PREVIOUS value
    x["mirror"] = (
        (prev + 5) % 10
    )

    # --------------------------------------------------------------------------
    # Time features
    # --------------------------------------------------------------------------

    dt = x["date"].dt

    weekday = dt.weekday.astype(float)

    x["weekday_sin"] = np.sin(
        2 * np.pi * weekday / 7
    )

    x["weekday_cos"] = np.cos(
        2 * np.pi * weekday / 7
    )

    month = dt.month.astype(float)

    x["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    x["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    # --------------------------------------------------------------------------
    # Lag features
    # --------------------------------------------------------------------------

    for lag in lags:

        x[f"lag_{lag}"] = target.shift(lag)

    # --------------------------------------------------------------------------
    # Rolling features
    # IMPORTANT:
    # shift(1) means current result is NEVER included
    # --------------------------------------------------------------------------

    shifted = target.shift(1)

    for w in rolls:

        x[f"roll_mean_{w}"] = (
            shifted
            .rolling(w, min_periods=1)
            .mean()
        )

        x[f"roll_std_{w}"] = (
            shifted
            .rolling(w, min_periods=1)
            .std()
            .fillna(0)
        )

        x[f"roll_min_{w}"] = (
            shifted
            .rolling(w, min_periods=1)
            .min()
        )

        x[f"roll_max_{w}"] = (
            shifted
            .rolling(w, min_periods=1)
            .max()
        )

    # --------------------------------------------------------------------------
    # STRICT CAUSAL SKIP
    #
    # Old version could see target[i].
    # This version calculates skip from target[i-1] backwards only.
    # --------------------------------------------------------------------------

    prev_arr = target.shift(1).to_numpy()

    n = len(prev_arr)

    idx = np.arange(n)

    for d in range(10):

        hit = np.where(
            np.isfinite(prev_arr) &
            (prev_arr == d),
            idx,
            -1
        )

        last_seen = np.maximum.accumulate(hit)

        skip = np.where(
            last_seen >= 0,
            idx - last_seen,
            30
        )

        x[f"skip_{d}"] = np.clip(
            skip,
            0,
            30
        )

    # --------------------------------------------------------------------------
    # Previous digit relationships
    # --------------------------------------------------------------------------

    x["prev_diff_1"] = (
        prev - target.shift(2)
    ).abs()

    x["prev_repeat"] = (
        prev == target.shift(2)
    ).astype(np.int8)

    # --------------------------------------------------------------------------
    # Remove impossible values
    # --------------------------------------------------------------------------

    x = x.replace(
        [np.inf, -np.inf],
        np.nan
    )

    x = x.fillna(-1)

    return x


# ==============================================================================
# 5. PROBABILITY NORMALIZER
# ==============================================================================

def normalize_probs(p):

    p = np.asarray(
        p,
        dtype=float
    )

    p[~np.isfinite(p)] = 0

    total = p.sum()

    if total <= 0:
        return np.ones(10) / 10

    return p / total


def model_probs(model, X):

    try:

        raw = model.predict_proba(X)[0]

        result = np.zeros(10)

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

class FastStatSystem:

    @staticmethod
    def markov(seq):

        seq = np.asarray(
            seq,
            dtype=int
        )

        if len(seq) < 8:
            return np.ones(10) / 10

        last = int(seq[-1])

        mask = seq[:-1] == last

        next_values = seq[1:][mask]

        if len(next_values) < 2:

            return np.ones(10) / 10

        counts = np.bincount(
            next_values,
            minlength=10
        ).astype(float)

        # Laplace smoothing
        counts += 0.5

        return normalize_probs(counts)

    @staticmethod
    def frequency_skip(seq):

        seq = np.asarray(
            seq,
            dtype=int
        )

        n = len(seq)

        if n == 0:
            return np.ones(10) / 10

        recent = (
            seq[-30:]
            if n >= 30
            else seq
        )

        result = np.zeros(10)

        for d in range(10):

            freq_all = np.mean(
                seq == d
            )

            freq_recent = np.mean(
                recent == d
            )

            pos = np.where(
                seq == d
            )[0]

            if len(pos):

                skip = n - pos[-1] - 1

            else:

                skip = 30

            freq_score = (
                0.40 * freq_all +
                0.60 * freq_recent
            )

            skip_score = np.clip(
                1.0 - skip / 30.0,
                0,
                1
            )

            result[d] = (
                0.60 * freq_score +
                0.40 * skip_score
            )

        return normalize_probs(result)

    @staticmethod
    def day_probability(
        df,
        target_col,
        target_dow
    ):

        mask = (
            df["date"].dt.weekday ==
            target_dow
        )

        values = df.loc[
            mask,
            target_col
        ].astype(int).to_numpy()

        if len(values) < 3:

            return np.ones(10) / 10

        counts = np.bincount(
            values,
            minlength=10
        ).astype(float)

        # Mild smoothing
        counts += 1.0

        return normalize_probs(counts)


# ==============================================================================
# 7. AI ENGINE
# ==============================================================================

class TurboAI:

    def __init__(
        self,
        df,
        target_col
    ):

        self.df = df
        self.target_col = target_col
        self.n = len(df)

        self.cfg = get_adaptive_config(
            self.n
        )

        self.trees = self.cfg["trees"]
        self.depth = self.cfg["max_depth"]
        self.leaf = self.cfg["leaf"]

        self.lags = self.cfg["lags"]
        self.rolls = self.cfg["rolls"]

        self.models = {
            "ET": ExtraTreesClassifier(
                n_estimators=self.trees,
                max_depth=self.depth,
                min_samples_leaf=self.leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ),

            "RF": RandomForestClassifier(
                n_estimators=self.trees,
                max_depth=self.depth,
                min_samples_leaf=self.leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=43,
                n_jobs=-1
            ),

            "HGB": HistGradientBoostingClassifier(
                max_iter=45,
                max_depth=min(
                    5,
                    self.depth
                ),
                learning_rate=0.07,
                min_samples_leaf=max(
                    3,
                    self.leaf
                ),
                l2_regularization=0.2,
                random_state=44
            )
        }

    # --------------------------------------------------------------------------
    # AI probability
    # --------------------------------------------------------------------------

    def train_predict(
        self,
        X_train,
        y_train,
        X_predict,
        weights=None
    ):

        if weights is None:

            weights = {
                "ET": 0.40,
                "RF": 0.35,
                "HGB": 0.25
            }

        result = np.zeros(10)
        total = 0.0

        for name, base in self.models.items():

            w = float(
                weights.get(name, 0)
            )

            if w <= 0:
                continue

            try:

                params = base.get_params()

                model = type(base)(
                    **params
                )

                model.fit(
                    X_train,
                    y_train
                )

                p = model_probs(
                    model,
                    X_predict
                )

                result += p * w
                total += w

            except Exception:
                continue

        if total <= 0:

            return np.ones(10) / 10

        return normalize_probs(
            result / total
        )

    # --------------------------------------------------------------------------
    # Fast proxy backtest
    # --------------------------------------------------------------------------

    def walk_forward(
        self,
        X,
        y,
        df
    ):

        n = len(X)

        min_train = self.cfg["min_train"]
        steps = self.cfg["bt_steps"]

        if n <= min_train + 2:

            return {
                "ai": 0.5,
                "stat": 0.5,
                "day": 0.5,
                "steps": 0
            }

        start = max(
            min_train,
            n - steps
        )

        indices = np.arange(
            start,
            n
        )

        # Small model for backtest
        proxy = ExtraTreesClassifier(
            n_estimators=18,
            max_depth=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=99,
            n_jobs=-1
        )

        ai_hits = 0
        stat_hits = 0
        day_hits = 0
        count = 0

        values = y.to_numpy(
            dtype=int
        )

        for i in indices:

            X_train = X.iloc[:i]
            y_train = y.iloc[:i]

            X_test = X.iloc[[i]]

            actual = int(
                y.iloc[i]
            )

            # --------------------------------------------------------------
            # AI
            # --------------------------------------------------------------

            try:

                proxy.fit(
                    X_train,
                    y_train
                )

                p_ai = model_probs(
                    proxy,
                    X_test
                )

                ai_top7 = np.argsort(
                    p_ai
                )[:7]

                if actual in ai_top7:
                    ai_hits += 1

            except Exception:

                pass

            # --------------------------------------------------------------
            # Statistical
            # --------------------------------------------------------------

            hist = values[:i]

            p_markov = (
                FastStatSystem.markov(
                    hist
                )
            )

            p_freq = (
                FastStatSystem.frequency_skip(
                    hist
                )
            )

            p_stat = normalize_probs(
                0.55 * p_markov +
                0.45 * p_freq
            )

            if actual in np.argsort(
                p_stat
            )[:7]:

                stat_hits += 1

            # --------------------------------------------------------------
            # Day
            # --------------------------------------------------------------

            target_day = int(
                df.iloc[i]["date"].weekday()
            )

            p_day = (
                FastStatSystem.day_probability(
                    df.iloc[:i],
                    self.target_col,
                    target_day
                )
            )

            if actual in np.argsort(
                p_day
            )[:7]:

                day_hits += 1

            count += 1

        if count == 0:

            return {
                "ai": 0.5,
                "stat": 0.5,
                "day": 0.5,
                "steps": 0
            }

        return {
            "ai": ai_hits / count,
            "stat": stat_hits / count,
            "day": day_hits / count,
            "steps": count
        }

    # --------------------------------------------------------------------------
    # Analyze
    # --------------------------------------------------------------------------

    def analyze(
        self,
        target_date,
        target_dow
    ):

        if self.n < 30:
            return None

        # ----------------------------------------------------------------------
        # Create FUTURE row
        #
        # target is NaN
        # so the future result can NEVER leak into features.
        # ----------------------------------------------------------------------

        future = {
            "date": target_date,
            "draw_num": "000",
            "hundred": np.nan,
            "ten": np.nan,
            "unit": np.nan,
            "bot_ten": np.nan,
            "bot_unit": np.nan
        }

        extended = pd.concat(
            [
                self.df,
                pd.DataFrame([future])
            ],
            ignore_index=True
        )

        features = build_features_cached(
            extended,
            self.target_col,
            self.lags,
            self.rolls
        )

        drop_cols = [
            "date",
            "draw_num",
            "hundred",
            "ten",
            "unit",
            "bot_ten",
            "bot_unit",
            self.target_col
        ]

        X_all = features.iloc[:-1].drop(
            columns=drop_cols,
            errors="ignore"
        )

        X_predict = features.iloc[[-1]][
            X_all.columns
        ]

        y_all = self.df[
            self.target_col
        ].astype(int)

        # ----------------------------------------------------------------------
        # Walk Forward
        # ----------------------------------------------------------------------

        bt = self.walk_forward(
            X_all,
            y_all,
            self.df
        )

        # ----------------------------------------------------------------------
        # Dynamic ensemble weights
        #
        # Top-7 survival accuracy is converted into stable weights.
        # Avoids cubic amplification which could overreact to a few samples.
        # ----------------------------------------------------------------------

        base_ai = 0.55
        base_stat = 0.30
        base_day = 0.15

        if bt["steps"] > 0:

            ai_score = (
                0.50 +
                0.50 * bt["ai"]
            )

            stat_score = (
                0.50 +
                0.50 * bt["stat"]
            )

            day_score = (
                0.50 +
                0.50 * bt["day"]
            )

            wa = (
                base_ai *
                ai_score
            )

            ws = (
                base_stat *
                stat_score
            )

            wd = (
                base_day *
                day_score
            )

            total = wa + ws + wd

            if total > 0:

                w_ai = wa / total
                w_stat = ws / total
                w_day = wd / total

            else:

                w_ai = base_ai
                w_stat = base_stat
                w_day = base_day

        else:

            w_ai = base_ai
            w_stat = base_stat
            w_day = base_day

        # ----------------------------------------------------------------------
        # AI FINAL TRAIN
        # ----------------------------------------------------------------------

        ai_probs = self.train_predict(
            X_all,
            y_all,
            X_predict
        )

        # ----------------------------------------------------------------------
        # STAT FINAL
        # ----------------------------------------------------------------------

        seq = y_all.to_numpy(
            dtype=int
        )

        p_markov = (
            FastStatSystem.markov(seq)
        )

        p_freq = (
            FastStatSystem.frequency_skip(seq)
        )

        stat_probs = normalize_probs(
            0.55 * p_markov +
            0.45 * p_freq
        )

        # ----------------------------------------------------------------------
        # DAY FINAL
        # ----------------------------------------------------------------------

        day_probs = (
            FastStatSystem.day_probability(
                self.df,
                self.target_col,
                target_dow
            )
        )

        # ----------------------------------------------------------------------
        # FINAL
        # ----------------------------------------------------------------------

        final = normalize_probs(
            w_ai * ai_probs +
            w_stat * stat_probs +
            w_day * day_probs
        )

        bt_msg = (
            f"⚡ Strict WF {bt['steps']} งวด | "
            f"Top-7 รอด: "
            f"AI {bt['ai']:.0%} | "
            f"สถิติ {bt['stat']:.0%} | "
            f"วัน {bt['day']:.0%}"
        )

        return {
            "ai": ai_probs,
            "stat": stat_probs,
            "day": day_probs,
            "final": final,

            "w_ai": w_ai,
            "w_stat": w_stat,
            "w_day": w_day,

            "bt_msg": bt_msg
        }


# ==============================================================================
# 8. UI HELPERS
# ==============================================================================

def get_dead_numbers(
    probs,
    k=7
):

    idx = np.argsort(
        probs
    )[:k]

    return [
        (
            int(i),
            float(probs[i])
        )
        for i in idx
    ]


def format_dead(
    dead_list
):

    return " • ".join(
        str(num)
        for num, _ in dead_list
    )


def target_date_from_last(
    df,
    dow_input
):

    last_date = df["date"].iloc[-1]

    # Auto
    if dow_input is None:

        if len(df) >= 2:

            gap = (
                df["date"].iloc[-1] -
                df["date"].iloc[-2]
            ).days

            gap = max(
                1,
                min(gap, 31)
            )

        else:

            gap = 7

        target_date = (
            last_date +
            timedelta(days=gap)
        )

        target_dow = (
            target_date.weekday()
        )

    else:

        target_dow = dow_input

        delta = (
            target_dow -
            last_date.weekday()
        ) % 7

        if delta == 0:
            delta = 7

        target_date = (
            last_date +
            timedelta(days=delta)
        )

    return target_date, target_dow


# ==============================================================================
# 9. MAIN UI
# ==============================================================================

st.markdown(
    '<div class="main-title">'
    '🛑 ระบบวิเคราะห์เลขดับ PRO V5.0 TURBO MAX'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Strict Walk-Forward • Leakage Safe • AI Ensemble • Candidate Elimination TOP-7'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==============================================================================
# Selection
# ==============================================================================

col1, col2 = st.columns(2)

with col1:

    target_lotto = st.selectbox(
        "🎯 เลือกหวย",
        list(LOTTO_URLS.keys()),
        index=0
    )

with col2:

    day_options = {
        "อัตโนมัติ (จากงวดล่าสุด)": None,
        "วันจันทร์": 0,
        "วันอังคาร": 1,
        "วันพุธ": 2,
        "วันพฤหัสบดี": 3,
        "วันศุกร์": 4,
        "วันเสาร์": 5,
        "วันอาทิตย์": 6
    }

    day_label = st.selectbox(
        "📅 ออกวัน",
        list(day_options.keys()),
        index=0
    )

    dow_input = day_options[
        day_label
    ]


# ==============================================================================
# ANALYZE
# ==============================================================================

if st.button(
    "🛑 วิเคราะห์เลขดับ 7 ตัว ⚡ TURBO MAX",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "⚡ AI กำลังวิเคราะห์แบบ Strict Walk-Forward..."
    ):

        df = fetch_data(
            target_lotto
        )

        if df is None or df.empty:

            st.error(
                "❌ ไม่สามารถดึงข้อมูลได้"
            )

            st.stop()

        # ----------------------------------------------------------------------
        # Target date
        # ----------------------------------------------------------------------

        target_date, target_dow = (
            target_date_from_last(
                df,
                dow_input
            )
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

        st.info(
            f"📅 **งวดเป้าหมาย:** "
            f"วัน{dow_names[target_dow]} "
            f"{target_date.strftime('%d/%m/%Y')} "
            f"| ข้อมูลอ้างอิง {len(df)} งวด"
        )

        cfg = get_adaptive_config(
            len(df)
        )

        st.caption(
            f"⚙️ {cfg['mode']} | "
            f"Trees={cfg['trees']} | "
            f"WF={cfg['bt_steps']} จุด | "
            f"Features แบบ Causal"
        )

        # ----------------------------------------------------------------------
        # Positions
        # ----------------------------------------------------------------------

        positions = {
            "💯 3 ตัวบน (ร้อย)": "hundred",
            "🔟 3 ตัวบน (สิบ)": "ten",
            "1️⃣ 3 ตัวบน (หน่วย)": "unit",
            "🔽 2 ตัวล่าง (สิบ)": "bot_ten",
            "⬇️ 2 ตัวล่าง (หน่วย)": "bot_unit"
        }

        store_final_probs = {}

        # ----------------------------------------------------------------------
        # Analyze each position
        # ----------------------------------------------------------------------

        progress = st.progress(
            0,
            text="กำลังเริ่ม..."
        )

        total_positions = len(
            positions
        )

        for pos_idx, (
            position_name,
            col
        ) in enumerate(
            positions.items(),
            start=1
        ):

            progress.progress(
                pos_idx / total_positions,
                text=f"⚡ วิเคราะห์ {position_name}"
            )

            system = TurboAI(
                df,
                col
            )

            result = system.analyze(
                target_date,
                target_dow
            )

            if result is None:
                continue

            store_final_probs[
                col
            ] = result["final"]

            dead_final = (
                get_dead_numbers(
                    result["final"],
                    7
                )
            )

            dead_ai = (
                get_dead_numbers(
                    result["ai"],
                    7
                )
            )

            dead_stat = (
                get_dead_numbers(
                    result["stat"],
                    7
                )
            )

            dead_day = (
                get_dead_numbers(
                    result["day"],
                    7
                )
            )

            st.markdown(
                f"""
                <div class="dead-card">

                    <div class="position-title">
                        {position_name}
                    </div>

                    <div style="
                        text-align:center;
                        color:#555;
                        font-weight:bold;
                    ">
                        🚫 สรุปเลขดับ 7 ตัว
                    </div>

                    <div class="dead-number-highlight">
                        {format_dead(dead_final)}
                    </div>

                    <div style="
                        margin-top:16px;
                        border-top:1px dashed #ddd;
                        padding-top:12px;
                    ">

                        <div class="info-row">
                            🤖 <b>AI:</b>
                            <span class="badge badge-ai">
                                {format_dead(dead_ai)}
                            </span>
                        </div>

                        <div class="info-row">
                            📊 <b>สถิติ:</b>
                            <span class="badge badge-stat">
                                {format_dead(dead_stat)}
                            </span>
                        </div>

                        <div class="info-row">
                            📅 <b>วัน:</b>
                            <span class="badge badge-day">
                                {format_dead(dead_day)}
                            </span>
                        </div>

                    </div>

                    <div style="
                        margin-top:10px;
                        font-size:12px;
                        color:#888;
                    ">

                        {result["bt_msg"]}<br>

                        ⚖️ น้ำหนัก:
                        AI {result["w_ai"]:.0%} |
                        สถิติ {result["w_stat"]:.0%} |
                        วัน {result["w_day"]:.0%}

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        progress.empty()

        # ==============================================================================
        # SUMMARY
        # ==============================================================================

        st.subheader(
            "🔥 สรุปภาพรวมเลขดับ"
        )

        col_sum1, col_sum2 = st.columns(2)

        # ----------------------------------------------------------------------
        # TOP
        # ----------------------------------------------------------------------

        if all(
            x in store_final_probs
            for x in [
                "hundred",
                "ten",
                "unit"
            ]
        ):

            top_probs = normalize_probs(
                (
                    store_final_probs["hundred"] +
                    store_final_probs["ten"] +
                    store_final_probs["unit"]
                ) / 3.0
            )

            dead_top = get_dead_numbers(
                top_probs,
                7
            )

            with col_sum1:

                st.markdown(
                    f"""
                    <div style="
                        background:#fff3f3;
                        padding:15px;
                        border-radius:10px;
                        border:2px solid #ffcdd2;
                        text-align:center;
                    ">

                        <div style="
                            font-weight:bold;
                            color:#d32f2f;
                        ">
                            🚫 ดับบนรวม 7 ตัว
                        </div>

                        <div style="
                            font-size:24px;
                            font-weight:900;
                            color:#b71c1c;
                            margin-top:10px;
                            letter-spacing:2px;
                        ">
                            {format_dead(dead_top)}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------------------------
        # BOTTOM
        # ----------------------------------------------------------------------

        if all(
            x in store_final_probs
            for x in [
                "bot_ten",
                "bot_unit"
            ]
        ):

            bot_probs = normalize_probs(
                (
                    store_final_probs["bot_ten"] +
                    store_final_probs["bot_unit"]
                ) / 2.0
            )

            dead_bot = get_dead_numbers(
                bot_probs,
                7
            )

            with col_sum2:

                st.markdown(
                    f"""
                    <div style="
                        background:#fff3f3;
                        padding:15px;
                        border-radius:10px;
                        border:2px solid #ffcdd2;
                        text-align:center;
                    ">

                        <div style="
                            font-weight:bold;
                            color:#d32f2f;
                        ">
                            🚫 ดับล่างรวม 7 ตัว
                        </div>

                        <div style="
                            font-size:24px;
                            font-weight:900;
                            color:#b71c1c;
                            margin-top:10px;
                            letter-spacing:2px;
                        ">
                            {format_dead(dead_bot)}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------------------------
        # Footer
        # ----------------------------------------------------------------------

        st.divider()

        st.caption(
            "🛡️ V5.0: Strict Causal Features • "
            "No Current-Draw Leakage • "
            "Fast Walk-Forward • "
            "ET + RF + HGB"
        )
