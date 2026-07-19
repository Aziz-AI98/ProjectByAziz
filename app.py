import gradio as gr

# استدعاء النماذج مباشرة عبر الـ API المجاني لـ Hugging Face بدون تحميلها على سيرفرك
try:
    classifier_en = gr.Interface.load("models/distilbert-base-uncased")
    classifier_ar = gr.Interface.load("models/CAMeL-Lab/bert-base-arabic-camelbert-mix")
except Exception:
    # طريقة بديلة للاستدعاء عبر الـ API إذا لم تنجح الأولى
    import requests
    def query_model(text, model_url):
        API_URL = f"https://api-inference.huggingface.co/models/{model_url}"
        response = requests.post(API_URL, json={"inputs": text})
        return response.json()

# دالة التحليل الخفيفة
def predict_stress(text):
    if not text.strip():
        return "الرجاء إدخال نص للتحليل."
    
    is_arabic = any("\u0600" <= char <= "\u06FF" for char in text)
    
    try:
        if is_arabic:
            # استخدام نموذج لغة الضاد للاستدلال
            res = gr.Interface.load("models/CAMeL-Lab/bert-base-arabic-camelbert-mix")(text)
        else:
            # استخدام النموذج الإنجليزي
            res = gr.Interface.load("models/distilbert-base-uncased")(text)
        return f"تحليل مستويات الإجهاد: {res}"
    except Exception as e:
        return f"جاري الاتصال بالسيرفر، يرجى المحاولة مرة أخرى خلال ثوانٍ..."

# بناء واجهة المستخدم
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 نظام التنبؤ بمستويات الإجهاد النفسي - النسخة السحابية السريعة")
    input_text = gr.Textbox(label="أدخل النص (يدعم العربية والإنجليزية)", placeholder="اكتب هنا...")
    output_text = gr.Textbox(label="النتيجة")
    submit_btn = gr.Button("تحليل النص")
    submit_btn.click(fn=predict_stress, inputs=input_text, outputs=output_text)

import os
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
