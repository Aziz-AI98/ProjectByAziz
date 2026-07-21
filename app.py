import gradio as gr
import requests
import os

def query_model(text, model_id):
    # استخدام الرابط التوجيهي الجديد لـ Hugging Face API
    API_URL = f"https://router.huggingface.co/hf-inference/v1/pred/{model_id}"
    
    # رأس الطلب المباشر
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=10)
    return response.json()

def predict_stress(text):
    if not text.strip():
        return "الرجاء إدخال نص للتحليل."
    
    is_arabic = any("\u0600" <= char <= "\u06FF" for char in text)
    
    try:
        if is_arabic:
            model_id = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
            data = query_model(text, model_id)
            
            if isinstance(data, dict) and "error" in data:
                return f"تنبيه السيرفر: {data['error']}"
                
            label = data[0][0]['label'] if isinstance(data, list) else data['label']
            score = (data[0][0]['score'] if isinstance(data, list) else data['score']) * 100
            return f"النتيجة (عربي): {label} - نسبة الثقة: {score:.1f}%"
        else:
            model_id = "distilbert-base-uncased"
            data = query_model(text, model_id)
            
            if isinstance(data, dict) and "error" in data:
                return f"Server Notice: {data['error']}"
                
            label = data[0][0]['label'] if isinstance(data, list) else data['label']
            score = (data[0][0]['score'] if isinstance(data, list) else data['score']) * 100
            return f"Result (English): {label} - Confidence: {score:.1f}%"
            
    except Exception as e:
        return f"جاري مزامنة الاتصال، يرجى إعادة المحاولة خلال ثوانٍ: {str(e)}"

# بناء الواجهة
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 نظام التنبؤ بمستويات الإجهاد النفسي")
    input_text = gr.Textbox(label="أدخل النص المراد تحليله", placeholder="اكتب شعورك هنا...")
    output_text = gr.Textbox(label="نتيجة التحليل")
    submit_btn = gr.Button("تحليل النص")
    submit_btn.click(fn=predict_stress, inputs=input_text, outputs=output_text)

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
