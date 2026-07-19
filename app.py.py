import gradio as gr
from transformers import pipeline

# 1. Load both Arabic and English optimized models locally
print("Loading English and Arabic AI Models... Please wait...")
try:
    # Optimized English Model
    en_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Optimized & Stable Arabic Model
    ar_classifier = pipeline("sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment")
    
    models_loaded = True
    print("✅ All models loaded successfully!")
except Exception as e:
    models_loaded = False
    error_msg = str(e)
    print(f"❌ Error loading models: {error_msg}")

# 2. Define the core analysis function supporting both languages
def analyze_stress(user_text, language):
    if not user_text.strip():
        if language == "English":
            return "⚠️ Please enter some text first.", ""
        else:
            return "⚠️ الرجاء كتابة نص أولاً.", ""
            
    if not models_loaded:
        return f"❌ Error loading models: {error_msg}", ""
    
    # --- English Analysis Logic ---
    if language == "English":
        result = en_classifier(user_text)[0]
        label = result['label']
        score = result['score'] * 100
        
        if label == 'NEGATIVE':
            status = f"🚨 Early Stress Detected! (Confidence: {score:.1f}%)"
            recommendations = (
                "💡 **Early Preventive Recommendations:**\n"
                "* Your text indicates high tension or emotional fatigue.\n"
                "* Take a 10-minute break away from screens immediately.\n"
                "* Practice deep breathing exercises (4-7-8 technique).\n"
                "* Consider offloading urgent tasks to reduce cognitive load."
            )
        else:
            status = f"🟢 Stable Mental State / Positive Tone (Confidence: {score:.1f}%)"
            recommendations = (
                "💪 **Keep it up!**\n"
                "* No immediate signs of psychological stress detected.\n"
                "* Continue managing your schedule efficiently to prevent sudden burnout."
            )
        return status, recommendations

    # --- Arabic Analysis Logic ---
    else:
        result = ar_classifier(user_text)[0]
        label = result['label']
        score = result['score'] * 100
        
        if label.lower() == 'negative':
            status = f"🚨 تم رصد مؤشرات إجهاد نفسي مبكر! (نسبة الثقة: {score:.1f}%)"
            recommendations = (
                "💡 **التوصيات الوقائية المبكرة:**\n"
                "* تشير الكلمات إلى وجود ضغوطات نفسية أو إرهاق عاطفي متراكم.\n"
                "* خذ استراحة لمدة 10 دقائق بعيداً عن الشاشات فوراً.\n"
                "* مارس تمارين التنفس العميق (تقنية 4-7-8).\n"
                "* حاول تنظيم مهامك العاجلة للتقليل من العبء الذهني اليومي."
            )
        elif label.lower() == 'positive':
            status = f"🟢 الحالة النفسية مستقرة / نبرة إيجابية (نسبة الثقة: {score:.1f}%)"
            recommendations = (
                "💪 **استمر في هذا الأداء الرائع!**\n"
                "* لا توجد علامات فورية تدل على الإجهاد النفسي الحاد في النص.\n"
                "* حافظ على تنظيم وقتك بشكل متوازن للوقاية من الاحتراق النفسي المفاجئ."
            )
        else:
            status = f"🟡 نبرة محايدة / حالة مستقرة (نسبة الثقة: {score:.1f}%)"
            recommendations = (
                "الحالة تبدو متوازنة حالياً، احرص دائماً على أخذ فترات راحة منتظمة أثناء الدراسة أو العمل."
            )
        return status, recommendations

# 3. Build the Dual-Language Gradio UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Early Psychological Stress Detection System | نظام كشف الإجهاد النفسي المبكر")
    gr.Markdown("### Project Prototype for the Evaluation Committee | نسخة تجريبية للجنة التقييم")
    
    with gr.Row():
        with gr.Column():
            lang_select = gr.Radio(
                choices=["English", "العربية"], 
                value="English", 
                label="Choose Input Language | اختر لغة الإدخال:"
            )
            
            input_text = gr.Textbox(
                label="Express your feelings or daily thoughts here | عبر عن مشاعرك أو يومياتك هنا:", 
                placeholder="Type your text based on the selected language...", 
                lines=5
            )
            submit_btn = gr.Button("Analyze Mental State | تحليل الحالة النفسية 📊", variant="primary")
            
        with gr.Column():
            output_status = gr.Textbox(label="Analysis Result | نتيجة التحليل:")
            output_tips = gr.Markdown(label="Recommendations | التوصيات المخصصة:")
            
    gr.Markdown("---")
    gr.Markdown("_✨ Future Roadmap: This backend will transition to a FastAPI server, connecting seamlessly to a mobile application (built via FlutterFlow) supporting speech and facial expression recognition._")

    # [مصحح] ربط الأحداث يجب أن يكون داخل نطاق Blocks
    submit_btn.click(
        fn=analyze_stress, 
        inputs=[input_text, lang_select], 
        outputs=[output_status, output_tips]
    )

# 4. Launch locally
demo.launch(share=False)