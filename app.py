import streamlit as st
import lotto_lekden
import lotto_lekdub

st.set_page_config(
    page_title="Lotto AI PRO",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Lotto AI PRO V4")

lotto = st.selectbox(
    "เลือกประเภทหวย",
    [
        "หวยไทย",
        "หวยธกส",
        "หวยออมสิน",
        "หวยลาว",
        "หวยฮานอย",
        "หวยมาเลย์",
        "หวยหุ้นไทยเย็น",
        "หวยหุ้นนิเคอิบ่าย",
        "หวยหุ้นฮั่งเส็งบ่าย",
        "หวยหุ้นจีนบ่าย",
    ]
)

mode = st.radio(
    "เลือกระบบ",
    ["เลขเด่น", "เลขดับ"],
    horizontal=True
)

if st.button("🚀 วิเคราะห์", use_container_width=True):

    with st.spinner("กำลังวิเคราะห์..."):

        if mode == "เลขเด่น":
            result = lotto_lekden.analyze_lotto(
                lotto_name=lotto,
                target_day=None
            )
        else:
            result = lotto_lekdub.analyze_lotto(
                lotto_name=lotto,
                target_day=None
            )

    st.success("วิเคราะห์เสร็จแล้ว")
    st.write(result)
