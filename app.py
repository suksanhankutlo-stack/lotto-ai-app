import streamlit as st
import requests
import os

# ตั้งค่าหน้าเพจหลัก
st.set_page_config(
    page_title="Lotto AI - ระบบวิเคราะห์สลาก",
    page_icon="🎯",
    layout="wide"
)

# กำหนด URL ของไฟล์ต้นฉบับจาก GitHub
URLS = {
    "🎯 วิเคราะห์เลขเด่น": "https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app/refs/heads/main/lotto_lekden.py",
    "🛑 วิเคราะห์เลขดับ": "https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app/refs/heads/main/lotto_lekdub.py"
}

@st.cache_resource(show_spinner=False, ttl=3600) # อัปเดตแคชทุก 1 ชั่วโมง
def fetch_code(url):
    """ฟังก์ชันดึงโค้ดจาก GitHub พร้อมทำ Caching"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"st.error('ไม่สามารถดึงข้อมูลจาก GitHub ได้: {e}')"

# ==========================================
# ส่วน UI แถบเมนูด้านข้าง (Sidebar)
# ==========================================
st.sidebar.title("เมนูการวิเคราะห์")
app_mode = st.sidebar.radio("เลือกระบบการทำงาน:", list(URLS.keys()))

st.sidebar.markdown("---")
st.sidebar.caption("แอปพลิเคชันจะดึงโค้ดเวอร์ชันล่าสุดจาก GitHub มาประมวลผลโดยอัตโนมัติ")

# ==========================================
# ส่วนประมวลผลและแสดงผล (Main Area)
# ==========================================
with st.spinner(f"กำลังโหลดระบบ {app_mode}..."):
    # 1. ดึงโค้ดจาก URL ที่เลือก
    script_code = fetch_code(URLS[app_mode])
    
    # 2. จำลอง Namespace ใหม่เพื่อป้องกันตัวแปรชนกันระหว่างสคริปต์
    module_namespace = {
        '__name__': '__main__',
        'st': st,
        'requests': requests,
        'os': os
    }
    
    # 3. สั่งรันโค้ดที่ดาวน์โหลดมา (Exec)
    try:
        exec(script_code, module_namespace)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการรันระบบ {app_mode}: {e}")
