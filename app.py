import gradio as gr

# -------------------------
# เรียก Engine จาก GitHub
# -------------------------
# สมมุติว่าโหลด lekden และ lekdub เรียบร้อยแล้ว

LOTTO_LIST = [
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

def analyze(lotto, mode):

    if mode == "เลขเด่น":
        result = lekden.analyze_lotto(
            lotto_name=lotto,
            target_day=None
        )
    else:
        result = lekdub.analyze_lotto(
            lotto_name=lotto,
            target_day=None
        )

    return str(result)

css = """
.gradio-container{
    max-width:900px!important;
    margin:auto;
    background:#111827;
}

button{
    height:55px;
    font-size:18px!important;
    border-radius:15px!important;
}

textarea{
    font-size:18px!important;
}

h1{
    text-align:center;
}

footer{
    display:none;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="red",
        secondary_hue="blue"
    ),
    css=css
) as demo:

    gr.Markdown(
        """
# 🎯 Lotto AI PRO V4

### Ultimate Lottery AI
เลขเด่น • เลขดับ • AI 4 สำนัก • Markov • Statistics
"""
    )

    with gr.Row():

        lotto = gr.Dropdown(
            choices=LOTTO_LIST,
            value="หวยไทย",
            label="🎯 เลือกประเภทหวย"
        )

        mode = gr.Radio(
            ["เลขเด่น","เลขดับ"],
            value="เลขเด่น",
            label="🧠 ระบบวิเคราะห์"
        )

    btn = gr.Button(
        "🚀 วิเคราะห์",
        variant="primary"
    )

    output = gr.Textbox(
        lines=25,
        label="📊 ผลการวิเคราะห์"
    )

    btn.click(
        analyze,
        [lotto,mode],
        output
    )

demo.launch()
