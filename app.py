import pandas as pd
import joblib
import streamlit as st

if not os.path.exists("trip_cost_model.pkl"):
    st.error("🚨 Model file not found! Please download `trip_cost_model.pkl` from the README release page.")
    st.stop()
# Load trained ML model
model = joblib.load("trip_cost_model.pkl")

def predict_single_trip(trip):
    df_trip = pd.DataFrame([trip])
    predicted_total = model.predict(df_trip)[0]
    return round(predicted_total, 2)

# Streamlit UI
st.title("💰 ML-Based Trip Cost Predictor")

source = st.text_input("Source city")
destination = st.text_input("Destination city")
duration_days = st.number_input("Trip duration in days", min_value=1)
num_people = st.number_input("Number of people", min_value=1)
accommodation = st.selectbox("Accommodation type", ["budget", "mid", "luxury"])
travel_mode = st.selectbox("Travel mode", ["bus", "train", "flight"])
num_activities = st.number_input("Number of activities", min_value=0)
travel_cost_input = st.text_input("Base travel cost (leave blank if unknown)")

if st.button("Predict Trip Cost"):
    travel_cost = float(travel_cost_input) if travel_cost_input.strip() != "" else 0.0

    trip_data = {
        "source": source,
        "destination": destination,
        "duration_days": int(duration_days),
        "num_people": int(num_people),
        "accommodation": accommodation,
        "travel_mode": travel_mode,
        "num_activities": int(num_activities),
        "travel_cost": travel_cost
    }

    total_cost = predict_single_trip(trip_data)
    cost_per_person = round(total_cost / num_people, 2)

    st.success(f"🤖 Predicted Total Trip Cost: ₹{total_cost}")
    st.info(f"💰 Cost Per Person: ₹{cost_per_person}")

