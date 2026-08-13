import streamlit as st
import requests

# ============================================================
# 🚀 ตั้งค่าหน้าเว็บ
# ============================================================

st.set_page_config(
    page_title="ระบบวิเคราะห์หวย สูตรคำนวณ AI",
    page_icon="🚀",
    layout="centered"
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
# 📥 โหลดโค้ดจาก GitHub
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

        st.error(
            f"❌ โหลดระบบไม่สำเร็จ "
            f"(HTTP {response.status_code})"
        )

        return None

    except Exception as e:

        st.error(
            f"❌ ไม่สามารถเชื่อมต่อ GitHub ได้: {e}"
        )

        return None


# ============================================================
# 🎨 DESIGN
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 850px;
    padding-top: 35px;
    padding-bottom: 40px;
}

/* =========================
   HEADER
========================= */

.logo-title {
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.25;
    margin-bottom: 8px;
}

.logo-subtitle {
    text-align: center;
    color: #777;
    font-size: 15px;
    margin-bottom: 35px;
}


/* =========================
   SYSTEM CARD
========================= */

.system-card {
    border: 1px solid #e8e8e8;
    border-radius: 20px;
    padding: 22px 20px 18px 20px;
    background: white;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 12px;
}

.system-icon {
    font-size: 34px;
    margin-bottom: 5px;
}

.system-name {
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 5px;
}

.system-detail {
    font-size: 14px;
    color: #777;
    line-height: 1.5;
    min-height: 44px;
}


/* =========================
   BUTTON
========================= */

.stButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 15px;
    font-size: 17px;
    font-weight: 750;
    border: 1px solid #dddddd;
    background: white;
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #999999;
}


/* =========================
   DIVIDER
========================= */

.clean-divider {
    height: 1px;
    background: #eeeeee;
    margin: 28px 0;
}


/* =========================
   STATUS
========================= */

.loading-box {
    text-align: center;
    padding: 15px;
    color: #777;
    font-size: 14px;
}


/* =========================
   FOOTER
========================= */

.footer {
    text-align: center;
    color: #999;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 🏠 HEADER
# ============================================================

st.markdown(
    """
    <div class="logo-title">
        🚀 ระบบวิเคราะห์หวย<br>
        สูตรคำนวณ AI
    </div>

    <div class="logo-subtitle">
        เลือกระบบที่ต้องการวิเคราะห์
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 🔴 🌑 2 ระบบ
# ============================================================

col1, col2 = st.columns(2, gap="medium")


# ============================================================
# 🔴 เลขเด่น
# ============================================================

with col1:

    st.markdown(
        """
        <div class="system-card">

            <div class="system-icon">
                🔴
            </div>

            <div class="system-name">
                ค้นหาเลขเด่น
            </div>

            <div class="system-detail">
                วิเคราะห์เลขที่มีแนวโน้ม
                มาแรงด้วยระบบ AI
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    btn_lekden = st.button(
        "🚀 วิเคราะห์เลขเด่น",
        key="btn_lekden",
        use_container_width=True
    )


# ============================================================
# 🌑 เลขดับ
# ============================================================

with col2:

    st.markdown(
        """
        <div class="system-card">

            <div class="system-icon">
                🌑
            </div>

            <div class="system-name">
                ค้นหาเลขดับ
            </div>

            <div class="system-detail">
                วิเคราะห์เลขที่มีแนวโน้ม
                หลุดด้วยระบบ AI
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    btn_lekdub = st.button(
        "🚀 วิเคราะห์เลขดับ",
        key="btn_lekdub",
        use_container_width=True
    )


# ============================================================
# 🔴 RUN เลขเด่น
# ============================================================

if btn_lekden:

    st.markdown(
        '<div class="clean-divider"></div>',
        unsafe_allow_html=True
    )

    with st.spinner("กำลังโหลดระบบวิเคราะห์เลขเด่น..."):

        code = fetch_code(URL_LEKDEN)

    if code:

        try:

            exec(code, globals())

        except Exception as e:

            st.error(
                "❌ เกิดข้อผิดพลาดในระบบเลขเด่น"
            )

            st.exception(e)


# ============================================================
# 🌑 RUN เลขดับ
# ============================================================

elif btn_lekdub:

    st.markdown(
        '<div class="clean-divider"></div>',
        unsafe_allow_html=True
    )

    with st.spinner("กำลังโหลดระบบวิเคราะห์เลขดับ..."):

        code = fetch_code(URL_LEKDUB)

    if code:

        try:

            exec(code, globals())

        except Exception as e:

            st.error(
                "❌ เกิดข้อผิดพลาดในระบบเลขดับ"
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🤖 AI Lottery Analysis System
        <br>
        ระบบวิเคราะห์เพื่อประกอบการพิจารณา
    </div>
    """,
    unsafe_allow_html=True
)
