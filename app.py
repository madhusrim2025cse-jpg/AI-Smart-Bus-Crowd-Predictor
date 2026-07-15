from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# -----------------------------
# Store Prediction History
# -----------------------------
history_data = []

# -----------------------------
# Load Bus Details
# -----------------------------
bus_details = pd.read_csv("dataset/buses.csv")

# -----------------------------
# Load AI Model
# -----------------------------
model = joblib.load("model/bus_model.pkl")

# -----------------------------
# Load Encoders
# -----------------------------
le_bus = joblib.load("model/le_bus.pkl")
le_day = joblib.load("model/le_day.pkl")
le_time = joblib.load("model/le_time.pkl")
le_weather = joblib.load("model/le_weather.pkl")
le_crowd = joblib.load("model/le_crowd.pkl")


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    bus = request.form["bus"]
    stop = request.form["stop"]
    destination = request.form["destination"]
    day = request.form["day"]
    time = request.form["time"]
    weather = request.form["weather"]

    # -----------------------------
    # AI Prediction
    # -----------------------------
    try:

        bus_encoded = le_bus.transform([bus])[0]
        day_encoded = le_day.transform([day])[0]
        time_encoded = le_time.transform([time])[0]
        weather_encoded = le_weather.transform([weather])[0]

        prediction = model.predict([[
            bus_encoded,
            day_encoded,
            time_encoded,
            weather_encoded
        ]])

        crowd = le_crowd.inverse_transform(prediction)[0]

    except:

        crowd = "High"

    # -----------------------------
    # Get Bus Details
    # -----------------------------
    bus_info = bus_details[bus_details["Bus"] == bus]

    if not bus_info.empty:

        route = bus_info.iloc[0]["Route"]
        arrival = bus_info.iloc[0]["Arrival"]
        seats = bus_info.iloc[0]["Seats"]

    else:

        route = stop + " ➜ " + destination
        arrival = "Not Available"
        seats = "Not Available"

    # -----------------------------
    # Alternative Bus
    # -----------------------------
    if crowd == "High":

        next_bus = "21A"

    elif crowd == "Medium":

        next_bus = "47A"

    else:

        next_bus = bus

    # -----------------------------
    # Save History
    # -----------------------------
    history_data.append({

        "bus": bus,
        "crowd": crowd,
        "next_bus": next_bus

    })

    # -----------------------------
    # Result Page
    # -----------------------------
    return render_template(

        "result.html",

        bus=bus,
        stop=stop,
        destination=destination,
        route=route,
        crowd=crowd,
        next_bus=next_bus,
        arrival=arrival,
        seats=seats

    )


# -----------------------------
# Dashboard
# -----------------------------
@app.route("/dashboard")
def dashboard():

    total = len(history_data)

    high = sum(1 for x in history_data if x["crowd"] == "High")
    medium = sum(1 for x in history_data if x["crowd"] == "Medium")
    low = sum(1 for x in history_data if x["crowd"] == "Low")

    return render_template(

        "dashboard.html",

        total=total,
        high=high,
        medium=medium,
        low=low

    )


# -----------------------------
# History
# -----------------------------
@app.route("/history")
def history():

    return render_template(

        "history.html",

        history=history_data

    )


# -----------------------------
# About
# -----------------------------
@app.route("/about")
def about():

    return render_template("about.html")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    app.run(debug=True)