import streamlit as st
import requests

# ============================================================
# 🚀 LOTTO AI
# ============================================================

st.set_page_config(
    page_title="ระบบวิเคราะห์หวย สูตรคำนวณ AI",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 🔗 GitHub Raw
# ============================================================

URL_LEKDEN = (
    "https://raw.githubusercontent.com/"
    "suksanhankutlo-stack/lotto-ai-app/"
    "refs/heads/main/lotto_lekden.py"
)

URL_LEKDUB = (
    "https://raw.githubusercontent.com/"
    "suksanhankutlo-stack/lotto-ai-app/"
    "refs/heads/main/lotto_lekdub.py"
)

# ============================================================
# 🎨 CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.main-title {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 4px;
}

.sub-title {
    text-align: center;
    color: #777;
    font-size: 14px;
    margin-bottom: 25px;
}

/* Analysis cards */
.analysis-card {
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 12px;
    background: #ffffff;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

.card-title {
    font-size: 20px;
    font-weight: 750;
    margin-bottom: 5px;
}

.card-desc {
    color: #777;
    font-size: 13px;
    line-height: 1.5;
}

/* Buttons */
.stButton > button {
    width: 100%;
    min-height: 52px;
    border-radius: 14px;
    font-size: 17px;
    font-weight: 700;
    border: 1px solid #ddd;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #999;
}

/* Divider */
.soft-divider {
    height: 1px;
    background: #eeeeee;
    margin: 25px 0;
}

.footer {
    text-align: center;
    color: #999;
    font-size: 12px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# 📥 ดึงโค้ดจาก GitHub
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_code(url):

    try:

        response = requests.get(
            url,
            timeout=15
        )

        if response.status_code == 200:

            return response.text

        return None

    except Exception:

        return None


# ============================================================
# ▶️ Execute ระบบ
# ============================================================

def run_analysis(url, system_name):

    with st.spinner(f"กำลังโหลด {system_name}..."):

        code = fetch_code(url)

    if not code:

        st.error(
            "❌ ไม่สามารถโหลดระบบจาก GitHub ได้\n\n"
            "กรุณาตรวจสอบ Internet หรือไฟล์บน GitHub"
        )

        return

    try:

        # แยก namespace เพื่อไม่ให้ตัวแปรชนกันมากเกินไป
        namespace = {
            "__name__": "__main__"
        }

        exec(code, namespace)

    except Exception as e:

        st.error(
            f"❌ เกิดข้อผิดพลาดในระบบ {system_name}"
        )

        st.exception(e)


# ============================================================
# 🏠 HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚀 ระบบวิเคราะห์หวย AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'ระบบคำนวณเลขเด่นและเลขดับด้วย AI'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# 🔴 / 🌑 ANALYSIS
# ============================================================

col1, col2 = st.columns(2, gap="medium")

# ------------------------------------------------------------
# 🔴 เลขเด่น
# ------------------------------------------------------------

with col1:

    st.markdown(
        """
        <div class="analysis-card">
            <div class="card-title">🔴 เลขเด่น</div>
            <div class="card-desc">
                วิเคราะห์ตัวเลขที่มีแนวโน้มโดดเด่น
                จากระบบ AI
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "🔴 วิเคราะห์เลขเด่น",
        key="btn_lekden",
        use_container_width=True
    ):

        st.session_state["analysis_mode"] = "lekden"


# ------------------------------------------------------------
# 🌑 เลขดับ
# ------------------------------------------------------------

with col2:

    st.markdown(
        """
        <div class="analysis-card">
            <div class="card-title">🌑 เลขดับ</div>
            <div class="card-desc">
                วิเคราะห์ตัวเลขที่มีแนวโน้ม
                หลุดออกจากผลรางวัล
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "🌑 วิเคราะห์เลขดับ",
        key="btn_lekdub",
        use_container_width=True
    ):

        st.session_state["analysis_mode"] = "lekdub"


# ============================================================
# 📊 RUN SELECTED SYSTEM
# ============================================================

if "analysis_mode" in st.session_state:

    st.markdown(
        '<div class="soft-divider"></div>',
        unsafe_allow_html=True
    )

    mode = st.session_state["analysis_mode"]

    if mode == "lekden":

        run_analysis(
            URL_LEKDEN,
            "ระบบวิเคราะห์เลขเด่น"
        )

    elif mode == "lekdub":

        run_analysis(
            URL_LEKDUB,
            "ระบบวิเคราะห์เลขดับ"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🤖 AI Lottery Analysis System
        <br>
        วิเคราะห์จากข้อมูลที่ระบบกำหนด
    </div>
    """,
    unsafe_allow_html=True
        )
