import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Read dataset
data = pd.read_csv("dataset/bus_data.csv")

# Create Label Encoders
le_bus = LabelEncoder()
le_day = LabelEncoder()
le_time = LabelEncoder()
le_weather = LabelEncoder()
le_crowd = LabelEncoder()

# Convert text to numbers
data["Bus"] = le_bus.fit_transform(data["Bus"])
data["Day"] = le_day.fit_transform(data["Day"])
data["Time"] = le_time.fit_transform(data["Time"])
data["Weather"] = le_weather.fit_transform(data["Weather"])
data["Crowd"] = le_crowd.fit_transform(data["Crowd"])

# Input and Output
X = data[["Bus", "Day", "Time", "Weather"]]
y = data["Crowd"]

# Train Model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save Model
joblib.dump(model, "model/bus_model.pkl")

# Save Encoders
joblib.dump(le_bus, "model/le_bus.pkl")
joblib.dump(le_day, "model/le_day.pkl")
joblib.dump(le_time, "model/le_time.pkl")
joblib.dump(le_weather, "model/le_weather.pkl")
joblib.dump(le_crowd, "model/le_crowd.pkl")

print("Model and Encoders saved successfully!")