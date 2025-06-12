import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap

V = pd.read_csv("变量.csv", encoding="gbk")

title = "Arthritis Risk Prediction Model"

model_path = "models/GBDT_best_model.pkl"
model = joblib.load(model_path)
explainer = shap.TreeExplainer(model)

selected_col = ['Stooping_difficulty', 'Age', 'Sex', 'Weight', 'TyG_WHtR', 'Race', 'HbA1c', 'Smoking']

st.set_page_config(page_title=f"{title}", layout="wide", page_icon="🖥️")

st.markdown(f'''
    <h1 style="text-align: center; font-size: 26px; font-weight: bold; color: white; background: #3478CE; border-radius: 0.5rem; margin-bottom: 15px;">
        {title}
    </h1>''', unsafe_allow_html=True)
    
BOOL = {"Yes":1, "No":0}

data = {}
with st.form("inputform"):
    col = st.columns(4)
    for i in range(V.shape[0]):
        v = eval(V.iloc[i]["变量类型"])
        if "step" in eval(V.iloc[i]["变量类型"]):
            if v["step"]<1:
                data[V.iloc[i]["变量名称"]] = col[i%4].number_input(V.iloc[i]["变量描述"], max_value=v["max"]-v["step"]+v["step"], min_value=v["min"]-v["step"]+v["step"], step=v["step"], value=v["default"]-v["step"]+v["step"])
            else:
                data[V.iloc[i]["变量名称"]] = col[i%4].number_input(V.iloc[i]["变量描述"], max_value=int(v["max"]), min_value=int(v["min"]), step=int(v["step"]), value=int(v["default"]))
        else:
            v1 = [j for j in v.keys() if j!="default"]
            data[V.iloc[i]["变量名称"]] = v[col[i%4].selectbox(V.iloc[i]["变量描述"], v1, index=v1.index(v["default"]))]

    c1 = st.columns(3)
    bt = c1[1].form_submit_button("**Start prediction**", use_container_width=True, type="primary")

def compute_tygwhtr(d):
    tyg = np.log(d['TG']*d['FPG']/2) 
    tygwhtr = tyg * (d['WC'] / d['Height'])
    d['TyG_WHtR'] = tygwhtr
    return d

data = compute_tygwhtr(data)
d = pd.DataFrame([data])
d = d[selected_col]

if bt:
    y_pred = model.predict(d)
    y_prob = model.predict_proba(d)[:, 1]
    
    shap_values = explainer.shap_values(d)
    
    res = round(float(y_prob)*100, 2)
    
    with st.expander("**Predict result**", True):
        # st.markdown(f'''
            # <div style="text-align: center; font-size: 26px; color: black; margin-bottom: 5px; font-family: Times New Roman; border-bottom: 1px solid black;">
            # Prediction result: {res}%
            # </div>''', unsafe_allow_html=True)
            
        txt = f'''
        * Your TyG-WHtR is **{d['TyG_WHtR'][0]:.3f}**.
        * Considering the provided health parameters, the estimated risk of having arthritis is approximately **{y_prob[0]:.3f}**.  
        '''
        
        st.info(txt)
        
        probs = [1 - float(y_prob), float(y_prob)]
        labels = ['Non-Arthritis', 'Arthritis']
        colors = ['#FF0D57', '#1E88E5'][::-1]

        fig = plt.figure(figsize=(6, 2))
        plt.barh(labels, probs, color=colors)
        plt.xlim(0, 1)
        plt.xlabel('Predicted Risk')
        plt.title('Predicted Risk Probability')
        for index, value in enumerate(probs):
            plt.text(value + 0.01, index, f"{value:.2f}", va='center')
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        col = st.columns([1, 2, 1])
        col[1].pyplot(fig, use_container_width=True)
        
        #shap.force_plot(explainer.expected_value, shap_values, d, matplotlib=True)
        #st.pyplot(plt.gcf(), use_container_width=True)
    
    with st.expander("**Notes**", True):
        txt1 = f"""
        * This result is a preliminary evaluation using algorithm, and specific risks need to be comprehensively judged based on clinical examinations.
        * TyG-WHtR = ln [triglyceride (mg/dl) × fasting blood glucose (mg/dl)/2] × waist circumference/height. 
        """
        
        st.success(txt1)

else:
    st.markdown(f'''
    <div style="text-align: center; font-size: 20px; color: white; margin-bottom: 5px; 
                background: #017AD5; padding: 1rem; border-radius: 0.5rem;">
    Click 'Start prediction' button to start predict!
    </div>''', unsafe_allow_html=True)

st.markdown('''
    <style>
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-zy6yx3.en45cdb4 > div > div > div:nth-child(4) > details > div > div > div > div > div > div {
            background: #EEEEEE !important;
            color: black;
        }
        summary {
            font-size: 40px !important;
        }
    </style>''', unsafe_allow_html=True)