import streamlit as st
import requests

# ==========================================
# 1. ตั้งค่าหน้าเว็บหลัก
# ==========================================
st.set_page_config(page_title="ระบบวิเคราะห์หวย สูตรคำนวณAi", page_icon="🚀", layout="centered")

# ==========================================
# 2. กำหนดลิงก์ Raw จาก GitHub
# ==========================================
URL_LEKDEN = "https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app/refs/heads/main/lotto_lekden.py"
URL_LEKDUB = "https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app/refs/heads/main/lotto_lekdub.py"

# ==========================================
# 3. ฟังก์ชันดึงโค้ด
# ==========================================
@st.cache_data(ttl=600)
def fetch_code(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            st.error(f"❌ ไม่สามารถดึงข้อมูลจากลิงก์ได้ (HTTP {response.status_code})")
            return None
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
        return None

# ==========================================
# 4. หน้า UI หลัก (เลือกหวย -> เลือกวัน -> กดรัน)
# ==========================================
st.markdown("<h3 style='text-align: center;'>🚀 ระบบวิเคราะห์หวย สูตรคำนวณAi</h3>", unsafe_allow_html=True)
st.write("") 

if 'active_mode' not in st.session_state:
    st.session_state.active_mode = None

# --- ส่วนของการเลือกหวย ---
lotto_list = [
    "1. หวยรัฐบาลไทย", 
    "2. หวยลาวพัฒนา", 
    "3. หวยฮานอย (ปกติ)", 
    "4. หวยฮานอย (พิเศษ)", 
    "5. หวยฮานอย (VIP)", 
    "6. หวยมาเลย์",
    "7. หวยออมสิน",
    "8. หวย ธ.ก.ส.",
    "9. หวยหุ้นไทย",
    "10. หวยหุ้นต่างประเทศ"
]
st.session_state.selected_lotto = st.selectbox("🎯 เลือกหวย:", lotto_list)

# --- ส่วนของการเลือกวัน (จันทร์ - อาทิตย์) ---
day_list = [
    "วันจันทร์",
    "วันอังคาร",
    "วันพุธ",
    "วันพฤหัสบดี",
    "วันศุกร์",
    "วันเสาร์",
    "วันอาทิตย์"
]
st.session_state.selected_date = st.selectbox("📅 ออกวัน:", day_list)

st.write("") 

# --- ส่วนของปุ่มกดวิเคราะห์ ---
if st.button("🔴 วิเคราะห์เลขเด่น (มาแรง)", type="primary", use_container_width=True):
    st.session_state.active_mode = "LEKDEN"

if st.button("🌑 วิเคราะห์เลขดับ (หลุดแน่นอน)", type="primary", use_container_width=True):
    st.session_state.active_mode = "LEKDUB"

st.divider()

# ==========================================
# 5. รันโค้ดและแสดงผลทันทีที่กดปุ่ม
# ==========================================
if st.session_state.active_mode == "LEKDEN":
    code = fetch_code(URL_LEKDEN)
    if code:
        exec(code, globals())

elif st.session_state.active_mode == "LEKDUB":
    code = fetch_code(URL_LEKDUB)
    if code:
        exec(code, globals())
