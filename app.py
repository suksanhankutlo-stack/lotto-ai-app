import streamlit as st
import requests

# ==========================================
# 1. ตั้งค่าหน้าเว็บหลัก (ต้องมีแค่ในไฟล์นี้ไฟล์เดียว)
# ==========================================
st.set_page_config(page_title="ระบบวิเคราะห์หวย สูตรคำนวณAi", page_icon="🚀", layout="centered")

# ==========================================
# 2. กำหนดลิงก์ Raw จาก GitHub
# ==========================================
URL_LEKDEN = "https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app/refs/heads/main/lotto_lekden.py"
URL_LEKDUB = "https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app/refs/heads/main/lotto_lekdub.py"

# ==========================================
# 3. ฟังก์ชันดึงโค้ด (แคชไว้ 10 นาทีเพื่อให้เว็บโหลดเร็วขึ้น)
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
# 4. หน้า UI หลัก (เปลี่ยนจาก Radio เป็นปุ่มกดเรียงแนวตั้ง)
# ==========================================
st.title("ระบบวิเคราะห์หวย สูตรคำนวณAi")

# สร้าง Session State เพื่อจำว่าผู้ใช้กำลังเลือกโหมดไหนอยู่
if 'active_mode' not in st.session_state:
    st.session_state.active_mode = None

# สร้างปุ่ม 2 ปุ่มเรียงกันในแนวตั้ง และใช้ use_container_width=True เพื่อให้ปุ่มกว้างเต็มหน้าจอ
if st.button("🔴 วิเคราะห์เลขเด่น (มาแรง)", type="primary", use_container_width=True):
    st.session_state.active_mode = "LEKDEN"

if st.button("🌑 วิเคราะห์เลขดับ (หลุดแน่นอน)", use_container_width=True):
    st.session_state.active_mode = "LEKDUB"

st.divider()

# ==========================================
# 5. ดึงโค้ดมา Execute ตามปุ่มที่กดค้างไว้
# ==========================================
if st.session_state.active_mode == "LEKDEN":
    code = fetch_code(URL_LEKDEN)
    if code:
        # รันโค้ดไฟล์เลขเด่น
        exec(code, globals())

elif st.session_state.active_mode == "LEKDUB":
    code = fetch_code(URL_LEKDUB)
    if code:
        # รันโค้ดไฟล์เลขดับ
        exec(code, globals())
