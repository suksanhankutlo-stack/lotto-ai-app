import streamlit as st
import requests

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Page Config - ต้องอยู่บรรทัดแรกสุดเสมอ)
# ==========================================
st.set_page_config(
    page_title="AI วิเคราะห์หวย ครบวงจร", 
    page_icon="🎯", 
    layout="centered"
)

# ==========================================
# 2. กำหนด URL ของระบบต่างๆ (System URLs)
# ==========================================
# จัดระเบียบ URL ให้อ่านและแก้ไขง่ายขึ้น
GITHUB_RAW_URL = "https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app/refs/heads/main/"

URL_LEKDEN = f"{GITHUB_RAW_URL}lotto_lekden.py"  
URL_LEKDUB = f"{GITHUB_RAW_URL}lotto_lekdub.py"

# ==========================================
# 3. ฟังก์ชันสำหรับดึงและรันโค้ดจาก URL
# ==========================================
def run_script_from_url(url):
    """ดึงไฟล์ Python จาก GitHub และรันผลลัพธ์แสดงบน Streamlit ทันที"""
    try:
            
            # ใช้ exec() เพื่อประมวลผลโค้ดที่ดึงมา
            exec(response.text, globals())
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูลจากลิงก์ได้ (ปัญหาการเชื่อมต่อ): {e}")
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลระบบ: {e}")

# ==========================================
# 4. ส่วนแสดงผล UI (User Interface)
# ==========================================

# สร้าง UI แบบ Tabs
tab1, tab2 = st.tabs(["🟢 ระบบวิเคราะห์เลขเด่น", "🔴 ระบบวิเคราะห์เลขดับ"])

with tab1:
    st.subheader("✨ วิเคราะห์เลขเด่น")
    run_script_from_url(URL_LEKDEN)

with tab2:
    st.subheader("🛑 วิเคราะห์เลขดับ")
    run_script_from_url(URL_LEKDUB)

# ส่วนท้าย (Footer)
st.divider()
st.caption("📌 หมายเหตุ: ข้อมูลนี้เป็นเพียงการวิเคราะห์ทางสถิติและการคำนวณ โปรดใช้วิจารณญาณในการตัดสินใจ")
