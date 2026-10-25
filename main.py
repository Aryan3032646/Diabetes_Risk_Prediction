import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. Train the model
print("Training BALANCED model on ALL features...")
try:
    df = pd.read_csv('final_dataset (1)_2.csv')
except FileNotFoundError:
    df = pd.read_csv('final_dataset (1).csv')

X = df.drop('diabetes', axis=1)
y = df['diabetes']
rf_model_full = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
rf_model_full.fit(X, y)
print("✅ Model ready!")

# 2. Prediction logic
def predict_diabetes(high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
                     phys_activity, fruits, veggies, heavy_alcohol, healthcare, no_doc_cost,
                     gen_hlth, ment_hlth, phys_hlth, diff_walk, sex, age, education, income):

    yes_no = lambda val: 1.0 if val == "Yes" else 0.0

    input_data = [[
        yes_no(high_bp), yes_no(high_chol), yes_no(chol_check), float(bmi),
        yes_no(smoker), yes_no(stroke), yes_no(heart_disease),
        yes_no(phys_activity), yes_no(fruits), yes_no(veggies),
        yes_no(heavy_alcohol), yes_no(healthcare), yes_no(no_doc_cost),
        float(gen_hlth), float(ment_hlth), float(phys_hlth), yes_no(diff_walk),
        1.0 if sex == "Male" else 0.0, float(age), float(education), float(income)
    ]]

    risk_prob = rf_model_full.predict_proba(input_data)[0][1]

    if risk_prob > 0.5:
        return f"""
        <div style='background: linear-gradient(to right, #fee2e2, #fecaca); padding: 30px; border-radius: 15px; border-left: 10px solid #ef4444; color: #7f1d1d; text-align: center; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);'>
            <h2 style='margin: 0; font-size: 36px; font-weight: 900; letter-spacing: 1px;'>⚠️ HIGH RISK DETECTED</h2>
            <p style='margin: 15px 0 0 0; font-size: 22px;'>Estimated Probability: <b>{risk_prob:.1%}</b></p>
        </div>
        """
    else:
        return f"""
        <div style='background: linear-gradient(to right, #dcfce3, #bbf7d0); padding: 30px; border-radius: 15px; border-left: 10px solid #22c55e; color: #14532d; text-align: center; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);'>
            <h2 style='margin: 0; font-size: 36px; font-weight: 900; letter-spacing: 1px;'>✅ LOW RISK</h2>
            <p style='margin: 15px 0 0 0; font-size: 22px;'>Estimated Probability: <b>{risk_prob:.1%}</b></p>
        </div>
        """

# 3. Custom CSS
css = """
.custom-header { text-align: center; padding: 30px; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
.custom-header h1 { margin: 0; font-size: 40px; font-weight: 800; color: white !important; }
.custom-header p { margin: 10px 0 0 0; font-size: 18px; opacity: 0.9; }
.custom-header a { color: #bfdbfe; text-decoration: none; font-weight: bold; }
.custom-header a:hover { text-decoration: underline; color: white; }
.section-title { font-size: 1.3rem !important; font-weight: 700 !important; color: white !important; padding-bottom: 8px; border-bottom: 2px solid #475569; margin-bottom: 15px !important;}
.custom-footer { text-align: center; padding: 20px; margin-top: 30px; font-size: 15px; color: #94a3b8; border-top: 1px solid #334155; }
.custom-footer a { color: #60a5fa; text-decoration: none; font-weight: 600; }
.custom-footer a:hover { text-decoration: underline; }
"""

# 4. Build the UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=css, title="Diabetes Risk Predictor") as app:

    # Header with Contact Info
    gr.HTML('''
    <div class="custom-header">
        <h1>🩺 Advanced Patient Dashboard</h1>
        <p>AI-Powered Diabetes Risk Assessment</p>
        <p style="font-size: 14px; margin-top: 5px;">Contact: <a href="mailto:aryansrivastava737@gmail.com">aryansrivastava737@gmail.com</a></p>
    </div>
    ''')

    with gr.Row():
        with gr.Column():
            with gr.Group():
                gr.Markdown("<div class='section-title'>📊 Vitals & Health Scores</div>")
                bmi = gr.Number(label="BMI (Body Mass Index)", value=25.0)
                gen_hlth = gr.Slider(minimum=1, maximum=5, step=1, label="General Health Score (1=Excellent to 5=Poor)", value=2)
                phys_hlth = gr.Slider(minimum=0, maximum=30, step=1, label="Days of Poor Physical Health (past 30 days)", value=0)
                ment_hlth = gr.Slider(minimum=0, maximum=30, step=1, label="Days of Poor Mental Health (past 30 days)", value=0)

            with gr.Group():
                gr.Markdown("<div class='section-title'>👤 Demographics & Access</div>")
                sex = gr.Radio(choices=["Female", "Male"], label="Sex", value="Female")
                age = gr.Slider(minimum=1, maximum=13, step=1, label="Age Category (1=18-24 ... 13=80+)", value=5)
                education = gr.Slider(minimum=1, maximum=6, step=1, label="Education Level (1=No school ... 6=College Grad)", value=5)
                income = gr.Slider(minimum=1, maximum=8, step=1, label="Income Level (1=<$10k ... 8=>=$75k)", value=6)
                healthcare = gr.Radio(choices=["No", "Yes"], label="Has Healthcare Coverage", value="Yes")
                no_doc_cost = gr.Radio(choices=["No", "Yes"], label="Avoided Doctor Due to Cost", value="No")

        with gr.Column():
            with gr.Group():
                gr.Markdown("<div class='section-title'>🏥 Medical Conditions</div>")
                high_bp = gr.Radio(choices=["No", "Yes"], label="High Blood Pressure", value="No")
                high_chol = gr.Radio(choices=["No", "Yes"], label="High Cholesterol", value="No")
                chol_check = gr.Radio(choices=["No", "Yes"], label="Cholesterol Checked (Last 5 yrs)", value="Yes")
                stroke = gr.Radio(choices=["No", "Yes"], label="History of Stroke", value="No")
                heart_disease = gr.Radio(choices=["No", "Yes"], label="History of Heart Disease", value="No")
                diff_walk = gr.Radio(choices=["No", "Yes"], label="Difficulty Walking or Climbing Stairs", value="No")

            with gr.Group():
                gr.Markdown("<div class='section-title'>🏃 Lifestyle & Habits</div>")
                smoker = gr.Radio(choices=["No", "Yes"], label="Smoker (100+ lifetime cigarettes)", value="No")
                phys_activity = gr.Radio(choices=["No", "Yes"], label="Physical Activity (past 30 days)", value="Yes")
                fruits = gr.Radio(choices=["No", "Yes"], label="Consume Fruits Daily", value="Yes")
                veggies = gr.Radio(choices=["No", "Yes"], label="Consume Veggies Daily", value="Yes")
                heavy_alcohol = gr.Radio(choices=["No", "Yes"], label="Heavy Alcohol Consumer", value="No")

    # Action Area
    gr.Markdown("<br>")
    submit_btn = gr.Button("🔍 Analyze Patient Data & Calculate Risk", variant="primary", size="lg")

    with gr.Row():
        result_html = gr.HTML()

    input_components = [
        high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
        phys_activity, fruits, veggies, heavy_alcohol, healthcare, no_doc_cost,
        gen_hlth, ment_hlth, phys_hlth, diff_walk, sex, age, education, income
    ]

    submit_btn.click(fn=predict_diabetes, inputs=input_components, outputs=result_html)

    # Contact Footer
    gr.HTML('''
    <div class="custom-footer">
        For support, inquiries, or feedback, please contact: <a href="mailto:aryansrivastava737@gmail.com">aryansrivastava737@gmail.com</a>
    </div>
    ''')

# Launch
app.launch(share=True)
