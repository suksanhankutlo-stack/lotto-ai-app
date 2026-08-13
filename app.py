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
# 🎨 CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 760px;
    padding-top: 25px;
    padding-bottom: 40px;
}

/* หัวข้อ */
.main-title {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    line-height: 1.25;
    margin-bottom: 8px;
}

.sub-title {
    text-align: center;
    color: #777;
    font-size: 14px;
    margin-bottom: 28px;
}

/* กล่องระบบ */
.card-title {
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 5px;
}

.card-description {
    color: #777;
    font-size: 14px;
    line-height: 1.5;
    margin-bottom: 15px;
}

/* ปุ่ม */
.stButton > button {
    width: 100%;
    min-height: 52px;
    border-radius: 14px;
    font-size: 16px;
    font-weight: 700;
}

/* เส้น */
.divider {
    height: 1px;
    background: #eeeeee;
    margin: 28px 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #999;
    font-size: 12px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 🏠 HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚀 ระบบวิเคราะห์หวย<br>สูตรคำนวณ AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">เลือกระบบที่ต้องการวิเคราะห์</div>',
    unsafe_allow_html=True
)


# ============================================================
# 🔴 🌑 ระบบวิเคราะห์
# ============================================================

col1, col2 = st.columns(2, gap="medium")


# ============================================================
# 🔴 เลขเด่น
# ============================================================

with col1:

    with st.container(border=True):

        st.markdown(
            '<div class="card-title">🔴 เลขเด่น</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-description">'
            'ค้นหาเลขเด่นที่มีแนวโน้มมาแรง '
            'ด้วยระบบ AI'
            '</div>',
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

    with st.container(border=True):

        st.markdown(
            '<div class="card-title">🌑 เลขดับ</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-description">'
            'ค้นหาเลขดับที่มีแนวโน้มหลุด '
            'ด้วยระบบ AI'
            '</div>',
            unsafe_allow_html=True
        )

        btn_lekdub = st.button(
            "🌑 วิเคราะห์เลขดับ",
            key="btn_lekdub",
            use_container_width=True
        )


# ============================================================
# 🔴 วิเคราะห์เลขเด่น
# ============================================================

if btn_lekden:

    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )

    with st.spinner("🔄 กำลังโหลดระบบวิเคราะห์เลขเด่น..."):

        code = fetch_code(URL_LEKDEN)

    if code:

        try:

            exec(code, globals())

        except Exception as e:

            st.error(
                "❌ เกิดข้อผิดพลาดในระบบวิเคราะห์เลขเด่น"
            )

            st.exception(e)


# ============================================================
# 🌑 วิเคราะห์เลขดับ
# ============================================================

elif btn_lekdub:

    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )

    with st.spinner("🔄 กำลังโหลดระบบวิเคราะห์เลขดับ..."):

        code = fetch_code(URL_LEKDUB)

    if code:

        try:

            exec(code, globals())

        except Exception as e:

            st.error(
                "❌ เกิดข้อผิดพลาดในระบบวิเคราะห์เลขดับ"
            )

            st.exception(e)


# ============================================================
# Footer
# ============================================================

st.markdown(
    '<div class="footer">'
    '🤖 AI Lottery Analysis System'
    '</div>',
    unsafe_allow_html=True
)
