import gradio as gr
import requests
import os

# دالة برمجية للاتصال المباشر بـ Hugging Face Inference API
def query_model(text, model_id):
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    # نرسل الطلب إلى السيرفر
    response = requests.post(API_URL, json={"inputs": text})
    return response.json()

def predict_stress(text):
    if not text.strip():
        return "الرجاء إدخال نص للتحليل."
    
    # فحص إذا كان النص يحتوي على أحرف عربية
    is_arabic = any("\u0600" <= char <= "\u06FF" for char in text)
    
    try:
        if is_arabic:
            # استخدام نموذج اللغة العربية
            model_id = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
            data = query_model(text, model_id)
            
            # إذا كان النموذج في وضع الخمول ويتحمل لأول مرة
            if isinstance(data, dict) and "estimated_time" in data:
                return f"جاري تشغيل النموذج على سيرفرات Hugging Face، يرجى إعادة الضغط خلال {int(data['estimated_time'])} ثانية."
                
            # استخراج النتيجة وتنسيقها
            label = data[0][0]['label']
            score = data[0][0]['score'] * 100
            return f"تحليل النص العربي -> النتيجة: {label} (نسبة التأكد: {score:.2f}%)"
            
        else:
            # استخدام النموذج الإنجليزي
            model_id = "distilbert-base-uncased"
            data = query_model(text, model_id)
            
            if isinstance(data, dict) and "estimated_time" in data:
                return f"Model is loading, please retry in {int(data['estimated_time'])} seconds."
                
            label = data[0][0]['label']
            score = data[0][0]['score'] * 100
            return f"English Analysis -> Label: {label} (Confidence: {score:.2f}%)"
            
    except Exception as e:
        return f"حدث خطأ أثناء معالجة البيانات: {str(e)}"

# بناء واجهة المستخدم
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 نظام التنبؤ بمستويات الإجهاد النفسي - النسخة السحابية المستقرة")
    input_text = gr.Textbox(label="أدخل النص (يدعم العربية والإنجليزية)", placeholder="اكتب هنا...")
    output_text = gr.Textbox(label="النتيجة")
    submit_btn = gr.Button("تحليل النص")
    submit_btn.click(fn=predict_stress, inputs=input_text, outputs=output_text)

# إعدادات المنفذ الخاصة بـ Render
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
