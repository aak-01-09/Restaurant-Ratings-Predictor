import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍴",
    layout="wide"
)

# ------------------------
# Background gradient
# ------------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #ffe6f0 0%, #fff0f5 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------
# Load model & scaler
# ------------------------
model = joblib.load("mlmodel.pkl")
scaler = joblib.load("Scaler.pkl")

# ------------------------
# Gradient Header
# ------------------------
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #ff758c 0%, #ff7eb3 100%);
                padding:25px; border-radius:15px; margin-bottom:20px;">
        <h1 style="text-align:center; color:white; font-size:42px;">🍴 Restaurant Rating Predictor</h1>
        <p style="text-align:center; color:white; font-size:18px;">
        Enter restaurant details and get the predicted review instantly!</p>
    </div>
    """, unsafe_allow_html=True
)

# ------------------------
# Sidebar Instructions
# ------------------------
with st.sidebar:
    st.header("Instructions 📝")
    st.write(
        """
        1️⃣ Enter estimated average cost for two.  
        2️⃣ Select if table booking and online delivery are available.  
        3️⃣ Choose price range (1 = Cheapest, 4 = Most Expensive).  
        4️⃣ Click **Predict Review Class**.  
        """
    )
    st.info("💡 Hover over inputs for hints!")
    st.write("---")
    st.write("👩‍💻 Made By:Aayushi Kataria | Himanshu Sharma")

# ------------------------
# Input Section in 2 columns
# ------------------------
st.subheader("Restaurant Data Inputs 📊")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """<div style="background-color:#ffffff;padding:20px;border-radius:15px;
           box-shadow:3px 3px 10px #ccc;">""",
        unsafe_allow_html=True
    )
    averagecost = st.number_input("💰 Average cost for two", min_value=50, max_value=999999, value=1000, step=200)
    tablebooking = st.selectbox("🪑 Table booking available?", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """<div style="background-color:#ffffff;padding:20px;border-radius:15px;
           box-shadow:3px 3px 10px #ccc;">""",
        unsafe_allow_html=True
    )
    onlinedelivery = st.selectbox("🚚 Online delivery available?", ["Yes", "No"])
    pricerange = st.selectbox("💵 Price range (1 = Cheapest, 4 = Most Expensive)", [1,2,3,4])
    st.markdown("</div>", unsafe_allow_html=True)

predictbutton = st.button("Predict Review Class 🍽️")
st.divider()

# ------------------------
# Prediction logic
# ------------------------
if predictbutton:
    st.balloons()

    bookingstatus = 1 if tablebooking=="Yes" else 0
    deliverystatus = 1 if onlinedelivery=="Yes" else 0

    input_data = np.array([[averagecost, bookingstatus, deliverystatus, pricerange]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Map prediction to text & color
    if prediction < 2.5:
        pred_text = "Poor"
        color = "red"
    elif prediction < 3.5:
        pred_text = "Average"
        color = "orange"
    elif prediction < 4.0:
        pred_text = "Good"
        color = "yellow"
    elif prediction < 4.5:
        pred_text = "Very Good"
        color = "lightgreen"
    else:
        pred_text = "Excellent"
        color = "green"

    # ------------------------
    # Prediction Card
    # ------------------------
    st.markdown(
        f"""
        <div style="background-color:{color};padding:25px;border-radius:15px;
                    box-shadow: 3px 3px 10px #aaa;">
            <h2 style="text-align:center;color:white;">Predicted Review: {pred_text} ⭐</h2>
            <p style="text-align:center;color:white;font-size:18px;">
                Predicted Rating: {prediction:.2f} / 5
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    # ------------------------
    # Semi-circle "Speedometer" Gauge using Plotly
    # ------------------------
    st.subheader("Visual Rating Gauge 📊")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        number={'suffix':' / 5', 'font': {'size': 24}},
        delta={'reference': 3, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range':[0,5]},
            'bar': {'color': color},
            'steps': [
                {'range':[0,2.5], 'color':'#ff4b4b'},
                {'range':[2.5,3.5], 'color':'#ffa500'},
                {'range':[3.5,4.0], 'color':'#ffd700'},
                {'range':[4.0,4.5], 'color':'#90ee90'},
                {'range':[4.5,5.0], 'color':'#32cd32'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    fig.update_layout(width=600, height=350, margin={'t':0,'b':0,'l':0,'r':0})
    st.plotly_chart(fig)