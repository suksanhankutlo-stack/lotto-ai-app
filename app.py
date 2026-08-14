import streamlit as st
import requests

# --- 1. ตั้งค่าหน้าเว็บ (ต้องอยู่บนสุด และเรียกใช้ได้แค่ครั้งเดียว) ---
st.set_page_config(page_title="AI วิเคราะห์หวย ครบวงจร", page_icon="🎯", layout="centered")

# --- 2. กำหนด URL ของระบบต่างๆ ---
URL_LEKDEN = "https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app/refs/heads/main/lotto_lekden.py"  
URL_LEKDUB = "https://raw.githubusercontent.com/suksanhankutlo-stack/lotto-ai-app/refs/heads/main/lotto_lekdub.py"

# --- 3. ฟังก์ชันสำหรับดึงและรันโค้ดจาก URL ---
def run_script_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # ใช้คำสั่ง exec() เพื่อประมวลผลโค้ดที่ดึงมาจากลิงก์
        # โดยให้รันในบริบท (context) ปัจจุบัน เพื่อให้ UI ไปแสดงใน Tab ที่ถูกต้อง
        exec(response.text, globals())
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ ไม่สามารถดึงข้อมูลจากลิงก์ได้: {e}")
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการรันระบบ: {e}")

# --- 4. สร้าง UI แบบ Tabs (ไม่ต้องซ่อนเมนูใน Sidebar) ---

tab1, tab2 = st.tabs(["🟢 ระบบวิเคราะห์เลขเด่น", "🔴 ระบบวิเคราะห์เลขดับ"])

with tab1:
    # เมื่ออยู่ใน tab1 ให้ดึงโค้ดเลขเด่นมารัน UI จะแสดงในกรอบนี้ทันที
    run_script_from_url(URL_LEKDEN)

with tab2:
    # เมื่ออยู่ใน tab2 ให้ดึงโค้ดเลขดับมารัน
    run_script_from_url(URL_LEKDUB)

