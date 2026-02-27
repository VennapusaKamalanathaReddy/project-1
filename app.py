import streamlit as st
import numpy as np
import joblib
import pyttsx3

# ---------- Voice Function ----------
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

# ---------- Load Model ----------
model = joblib.load("model.pkl")
label = joblib.load("label.pkl")

# ---------- Page Setup ----------
st.set_page_config(page_title="Air Quality Predictor", layout="centered")
st.title("Air Quality Illness Predictor")
st.caption("AI-based health risk prediction with voice alert")

voice_on = st.checkbox("Enable Voice Alert", value=True)

# ---------- Input Fields ----------
col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0)
    pm10 = st.number_input("PM10", min_value=0.0)
    no2 = st.number_input("NO2", min_value=0.0)
    so2 = st.number_input("SO2", min_value=0.0)

with col2:
    o3 = st.number_input("O3", min_value=0.0)
    temp = st.number_input("Temperature", min_value=0.0)
    humidity = st.number_input("Humidity", min_value=0.0)
    ph = st.number_input("pH", min_value=0.0)

# ---------- Predict ----------
if st.button("Predict"):

    data = np.array([[pm25, pm10, no2, so2, o3, temp, humidity, ph]])
    prediction = model.predict(data)
    result = label.inverse_transform(prediction)[0]

    st.subheader("Health Risk Level: " + result)

    # ---------- Outdoor Indicator ----------
    if result == "Low":
        st.success("Outdoor Status: SAFE to go outside")
    elif result == "Moderate":
        st.warning("Outdoor Status: LIMIT outdoor activity")
    else:
        st.error("Outdoor Status: NOT SAFE to go outside")

    # ---------- Risk Meter ----------
    risk_map = {"Low":20, "Moderate":40, "High":65, "Very High":85, "Critical":100}
    st.progress(risk_map[result])

    # ---------- General Advice ----------
    if result == "Low":
        st.success("Air quality is safe. Normal outdoor activities are fine.")
    elif result == "Moderate":
        st.warning("Moderate pollution detected. Reduce long outdoor exposure.")
    elif result == "High":
        st.error("High pollution level. Wear mask and avoid long outdoor exposure.")
    elif result == "Very High":
        st.error("Very high pollution. Stay indoors and avoid outdoor activity.")
    else:
        st.error("Critical pollution alert! Avoid going outside.")

    # ---------- Sensitive Group Advice ----------
    st.subheader("Sensitive Groups Advice")

    if result == "Low":
        st.write("Children: Safe to go outside")
        st.write("Asthma / Breathing patients: Safe to go outside")
        st.write("Elderly: Safe to go outside")

    elif result == "Moderate":
        st.write("Children: Limit long outdoor play")
        st.write("Asthma / Breathing patients: Avoid heavy outdoor activity")
        st.write("Elderly: Prefer short outdoor exposure")

    elif result == "High":
        st.write("Children: Not recommended for long outdoor stay")
        st.write("Asthma / Breathing patients: Avoid going outside")
        st.write("Elderly: Stay indoors as much as possible")

    elif result == "Very High":
        st.write("Children: Should stay indoors")
        st.write("Asthma / Breathing patients: Strictly avoid outdoor exposure")
        st.write("Elderly: Must stay indoors")

    else:
        st.write("Children: DO NOT go outside")
        st.write("Asthma / Breathing patients: Dangerous to go outside")
        st.write("Elderly: High health risk — remain indoors")

    # ---------- Precautions ----------
    st.subheader("Precautions")
    st.write("- Wear N95 mask in polluted areas")
    st.write("- Drink more water")
    st.write("- Avoid traffic and industrial zones")
    st.write("- Keep indoor plants")
    st.write("- Use air purifier if possible")

    # ---------- Voice Alert ----------
    if result == "Low":
        voice_msg = "Air quality is safe. You can go outside."
    elif result == "Moderate":
        voice_msg = "Moderate pollution detected. Sensitive people should be careful."
    elif result == "High":
        voice_msg = "High pollution level. Avoid long outdoor exposure."
    elif result == "Very High":
        voice_msg = "Very high pollution. Stay indoors."
    else:
        voice_msg = "Critical pollution alert. Do not go outside. Health risk."

    voice_msg += " Children, asthma patients, elderly and breathing problem patients should follow safety precautions."

    if voice_on:
        speak(voice_msg)