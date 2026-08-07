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
# 4. เมนูเลือกโหมดการทำงาน
# ==========================================
st.title(" ระบบวิเคราะห์หวย  สูตรคำนวณAi")
mode = st.radio("⚙️ เลือกระบบการประมวลผล:", ["🔴 ค้นหา เลขเด่น (มาแรง)", "🌑 ค้นหา เลขดับ (หลุดแน่นอน)"], horizontal=True)
st.divider()

# ==========================================
# 5. ดึงโค้ดมา Execute ตามโหมดที่เลือก
# ==========================================
if mode == "🔴 ค้นหา เลขเด่น (มาแรง)":
    code = fetch_code(URL_LEKDEN)
    if code:
        # รันโค้ดไฟล์เลขเด่น
        exec(code, globals())
        
elif mode == "🌑 ค้นหา เลขดับ (หลุดแน่นอน)":
    code = fetch_code(URL_LEKDUB)
    if code:
        # รันโค้ดไฟล์เลขดับ
        exec(code, globals())
