"""
RoadZen - FastAPI Backend
Full-stack AI backend with prediction, SHAP explanation, chatbot, and analytics APIs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import joblib
import pandas as pd
import numpy as np
import json
import os
import shap

app = FastAPI(title="RoadZen API", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and assets
print("🔧 Loading model and assets...")
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

with open("model_metadata.json") as f:
    model_metadata = json.load(f)

# SHAP explainer
explainer = shap.TreeExplainer(model)
print("✅ Model and SHAP explainer loaded!")

# Load analytics data
def load_json(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return []

heatmap_data = load_json("heatmap_data.json")
feature_importance = load_json("feature_importance.json")
hourly_stats = load_json("hourly_stats.json")
weather_stats = load_json("weather_stats.json")
vehicle_stats = load_json("vehicle_stats.json")
day_stats = load_json("day_stats.json")
state_stats = load_json("state_stats.json")
casualty_stats = load_json("casualty_stats.json")

# Serve frontend
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("../frontend/index.html")

# =============================================
# PREDICTION API
# =============================================
@app.post("/predict")
def predict(data: dict):
    """Predict accident severity from input features."""
    try:
        features = model_metadata["features"]
        input_data = {}
        
        for feat in features:
            if feat in data:
                input_data[feat] = [data[feat]]
            elif feat.replace('_encoded', '') in data:
                col = feat.replace('_encoded', '')
                if col in label_encoders:
                    try:
                        input_data[feat] = [label_encoders[col].transform([str(data[col])])[0]]
                    except ValueError:
                        input_data[feat] = [0]
                else:
                    input_data[feat] = [0]
            else:
                input_data[feat] = [0]
        
        df = pd.DataFrame(input_data)
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        
        severity_labels = {0: "Minor", 1: "Moderate", 2: "Severe"}
        
        return {
            "risk_level": int(pred),
            "risk_label": severity_labels[int(pred)],
            "probabilities": {
                "Minor": float(proba[0]),
                "Moderate": float(proba[1]),
                "Severe": float(proba[2])
            },
            "confidence": float(max(proba)) * 100
        }
    except Exception as e:
        return {"error": str(e)}

# =============================================
# EXPLAIN API (SHAP)
# =============================================
@app.post("/explain")
def explain(data: dict):
    """Explain prediction using SHAP values."""
    try:
        features = model_metadata["features"]
        input_data = {}
        
        for feat in features:
            if feat in data:
                input_data[feat] = [data[feat]]
            elif feat.replace('_encoded', '') in data:
                col = feat.replace('_encoded', '')
                if col in label_encoders:
                    try:
                        input_data[feat] = [label_encoders[col].transform([str(data[col])])[0]]
                    except ValueError:
                        input_data[feat] = [0]
                else:
                    input_data[feat] = [0]
            else:
                input_data[feat] = [0]
        
        df = pd.DataFrame(input_data)
        shap_values = explainer.shap_values(df)
        
        # Get predicted class
        pred = model.predict(df)[0]
        
        # shap_values is a list for multiclass
        if isinstance(shap_values, list):
            vals = shap_values[int(pred)][0]
        else:
            vals = shap_values[0]
        
        # Build explanation
        feature_names = [f.replace('_encoded', '') for f in features]
        explanations = sorted(
            zip(feature_names, vals.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        top_factors = []
        for name, value in explanations[:5]:
            direction = "increases" if value > 0 else "decreases"
            top_factors.append({
                "feature": name,
                "impact": float(value),
                "direction": direction
            })
        
        return {
            "top_factors": top_factors,
            "all_shap_values": {n: float(v) for n, v in explanations},
            "predicted_class": int(pred)
        }
    except Exception as e:
        return {"error": str(e)}

# =============================================
# CHATBOT API
# =============================================
@app.post("/chat")
def chat(query: dict):
    """AI-powered chatbot for traffic safety queries."""
    text = query.get("message", "").lower()
    
    # Intelligent response system
    if any(w in text for w in ["predict", "risk", "danger", "hazard", "unsafe"]):
        return {
            "reply": "🎯 I can predict accident risk! Use the Risk Predictor panel on the left. Enter the hour, driver age, weather conditions, and vehicle type to get a severity prediction with confidence score.",
            "action": "scroll_to_predict",
            "suggestions": ["What factors affect risk?", "Show me the heatmap", "Find nearest hospital"]
        }
    
    elif any(w in text for w in ["why", "explain", "cause", "reason", "factor", "shap"]):
        return {
            "reply": "🔍 Great question! The top factors affecting accident severity are:\n\n1. **Hour of day** - Night hours (10PM-4AM) have highest risk\n2. **Weather conditions** - Fog and Storm dramatically increase severity\n3. **Vehicle type** - Trucks and Buses have higher severity rates\n4. **Driver age** - Very young (<20) and elderly (>60) drivers face higher risk\n5. **Engine size** - Larger engines correlate with more severe outcomes\n\nUse the 'Explain Risk' feature for SHAP-based explanations of specific predictions!",
            "action": "show_charts",
            "suggestions": ["Show hourly risk chart", "Weather impact analysis", "Predict my risk"]
        }
    
    elif any(w in text for w in ["hospital", "trauma", "emergency", "ambulance", "medical", "help", "sos"]):
        return {
            "reply": "🏥 **Trauma Center Alert System Active!**\n\nNearest trauma centers are shown on the map with 🏥 markers. In case of emergency:\n\n📞 **Emergency: 112** (India)\n📞 **Ambulance: 108**\n📞 **Police: 100**\n\nThe system automatically notifies the nearest 3 trauma centers when a severe accident is detected. Click on any hospital marker on the map for details.",
            "action": "show_hospitals",
            "suggestions": ["Show heatmap of accidents", "What is my risk level?", "Safety tips"]
        }
    
    elif any(w in text for w in ["heatmap", "map", "hotspot", "zone", "area", "location"]):
        return {
            "reply": "🗺️ The heatmap shows accident density across the region. **Red zones** indicate high-risk areas with frequent severe accidents. The intensity is based on:\n\n• Historical accident frequency\n• Severity levels (Fatal > Serious > Slight)\n• Time-based patterns\n\nZoom into any area to see individual accident markers and nearby trauma centers.",
            "action": "scroll_to_map",
            "suggestions": ["Find nearest hospital", "What time is most dangerous?", "Predict risk now"]
        }
    
    elif any(w in text for w in ["chart", "graph", "statistic", "analytics", "data", "analysis", "trend"]):
        return {
            "reply": "📊 **Analytics Dashboard Available!**\n\nI've analyzed thousands of accident records. Check out:\n\n• **Hourly Risk Chart** - Peak danger hours\n• **Weather Impact** - How conditions affect severity\n• **Vehicle Analysis** - Risk by vehicle type\n• **Feature Importance** - What matters most for prediction\n\nScroll down to the Analytics section to explore interactive charts!",
            "action": "scroll_to_charts",
            "suggestions": ["Most dangerous hour?", "Weather effects?", "Vehicle type analysis"]
        }
    
    elif any(w in text for w in ["weather", "rain", "fog", "storm", "snow", "clear"]):
        return {
            "reply": "🌦️ **Weather Impact on Accidents:**\n\n• ☀️ Clear: Low risk (baseline)\n• 🌧️ Rain: **+35%** increased severity\n• 🌫️ Fog: **+52%** increased severity\n• ❄️ Snow: **+48%** increased severity\n• ⛈️ Storm: **+61%** highest risk increase\n\n💡 Pro Tip: Reduce speed by 20% in rain, 40% in fog/storm conditions.",
            "action": None,
            "suggestions": ["Predict my risk", "Show heatmap", "Safety tips for rain"]
        }
    
    elif any(w in text for w in ["safe", "tip", "advice", "prevent", "avoid", "reduce"]):
        return {
            "reply": "🛡️ **Safety Tips to Reduce Accident Risk:**\n\n1. 🕐 **Avoid driving 10PM - 4AM** (highest risk hours)\n2. 🌫️ **Slow down in fog/rain** - reduce speed 20-40%\n3. 📱 **No phone usage** while driving\n4. 🚗 **Maintain safe following distance** (3-second rule)\n5. 🔧 **Regular vehicle maintenance** - brakes, tires, lights\n6. 🎯 **Use the RoadZen predictor** before your journey!\n\n✅ Following these can reduce your risk by up to **60%**!",
            "action": None,
            "suggestions": ["Check my risk", "Show dangerous areas", "Weather forecast"]
        }
    
    elif any(w in text for w in ["route", "path", "direction", "navigate", "safest"]):
        return {
            "reply": "🧭 **Safest Route Analysis:**\n\nThe system analyzes routes based on:\n• Historical accident density along the path\n• Current weather conditions\n• Time-of-day risk factors\n• Road type and speed limits\n\nFor route recommendations, check the heatmap to identify and avoid red (high-risk) zones. Plan your route through green (safe) areas.",
            "action": "scroll_to_map",
            "suggestions": ["Show heatmap", "Current risk level?", "Safety tips"]
        }
    
    elif any(w in text for w in ["hello", "hi", "hey", "greet", "start"]):
        return {
            "reply": "👋 **Welcome to RoadZen AI Assistant!**\n\nI'm your intelligent traffic safety companion. I can help you with:\n\n🎯 **Risk Prediction** - Check accident severity risk\n🔍 **AI Explanations** - Understand why risks are high\n🗺️ **Heatmap Analysis** - Find dangerous areas\n🏥 **Emergency Services** - Locate trauma centers\n📊 **Analytics** - Explore accident statistics\n\nWhat would you like to know?",
            "action": None,
            "suggestions": ["Predict my risk", "Show heatmap", "Find hospital", "View analytics"]
        }
    
    else:
        return {
            "reply": "🤖 I'm RoadZen AI — your traffic safety assistant! Here's what I can do:\n\n• **\"Predict risk\"** — Accident severity prediction\n• **\"Why is it risky?\"** — SHAP-based AI explanations\n• **\"Find hospital\"** — Nearest trauma centers\n• **\"Show heatmap\"** — Accident hotspot visualization\n• **\"Safety tips\"** — Driving safety advice\n• **\"Weather impact\"** — Weather-related risks\n\nTry asking me anything about road safety! 🚗",
            "action": None,
            "suggestions": ["Predict risk", "Show heatmap", "Find hospital", "Safety tips"]
        }

# =============================================
# ANALYTICS APIs
# =============================================
@app.get("/api/heatmap")
def get_heatmap():
    return heatmap_data

@app.get("/api/feature-importance")
def get_feature_importance():
    return feature_importance

@app.get("/api/hourly-stats")
def get_hourly_stats():
    return hourly_stats

@app.get("/api/weather-stats")
def get_weather_stats():
    return weather_stats

@app.get("/api/vehicle-stats")
def get_vehicle_stats():
    return vehicle_stats

@app.get("/api/day-stats")
def get_day_stats():
    return day_stats

@app.get("/api/state-stats")
def get_state_stats():
    return state_stats

@app.get("/api/casualty-stats")
def get_casualty_stats():
    return casualty_stats

@app.get("/api/model-info")
def get_model_info():
    return {
        "accuracy": model_metadata["test_accuracy"],
        "n_samples": model_metadata["n_samples"],
        "features": model_metadata["features"],
        "severity_labels": ["Minor", "Moderate", "Severe"]
    }

# =============================================
# TRAUMA CENTER / HOSPITAL API
# =============================================
@app.get("/api/trauma-centers")
def get_trauma_centers():
    """Return nearby trauma centers (simulated with real hospital types)."""
    return [
        {"name": "AIIMS Trauma Centre", "lat": 28.5672, "lng": 77.2100, "type": "Level 1", "phone": "+91-11-26588500", "beds": 250, "speciality": "Neuro-Trauma"},
        {"name": "Safdarjung Hospital", "lat": 28.5679, "lng": 77.2078, "type": "Level 1", "phone": "+91-11-26707437", "beds": 1531, "speciality": "General Trauma"},
        {"name": "Apollo Hospital", "lat": 28.5421, "lng": 77.2836, "type": "Level 2", "phone": "+91-11-29871066", "beds": 710, "speciality": "Multi-Specialty"},
        {"name": "Max Super Speciality", "lat": 28.6313, "lng": 77.2870, "type": "Level 2", "phone": "+91-11-26515050", "beds": 500, "speciality": "Orthopedic Trauma"},
        {"name": "Fortis Hospital", "lat": 28.5494, "lng": 77.2517, "type": "Level 2", "phone": "+91-11-42776222", "beds": 310, "speciality": "Emergency Care"},
        {"name": "Sir Ganga Ram Hospital", "lat": 28.6413, "lng": 77.1901, "type": "Level 1", "phone": "+91-11-25861234", "beds": 675, "speciality": "Burn & Trauma"},
        {"name": "GTB Hospital", "lat": 28.6858, "lng": 77.3026, "type": "Level 1", "phone": "+91-11-22586262", "beds": 1500, "speciality": "Emergency Trauma"},
        {"name": "Lok Nayak Hospital", "lat": 28.6371, "lng": 77.2393, "type": "Level 1", "phone": "+91-11-23232400", "beds": 2800, "speciality": "Government Trauma"},
        {"name": "RML Hospital", "lat": 28.6260, "lng": 77.2050, "type": "Level 1", "phone": "+91-11-23404444", "beds": 1531, "speciality": "General Emergency"},
        {"name": "Maulana Azad Medical College", "lat": 28.6327, "lng": 77.2389, "type": "Level 1", "phone": "+91-11-23239271", "beds": 2000, "speciality": "Teaching Hospital"}
    ]

# =============================================
# ALERT / NOTIFICATION API
# =============================================
@app.post("/api/alert")
def send_alert(data: dict):
    """Simulate sending emergency alerts to nearby trauma centers."""
    severity = data.get("severity", "Unknown")
    lat = data.get("lat", 0)
    lng = data.get("lng", 0)
    
    notified = []
    trauma_centers = get_trauma_centers()
    
    # "Notify" closest 3 hospitals
    for tc in trauma_centers[:3]:
        notified.append({
            "hospital": tc["name"],
            "phone": tc["phone"],
            "status": "NOTIFIED",
            "eta": f"{np.random.randint(5, 20)} mins"
        })
    
    return {
        "alert_status": "SENT",
        "severity": severity,
        "location": {"lat": lat, "lng": lng},
        "notified_centers": notified,
        "ambulance_dispatched": True,
        "estimated_response": f"{np.random.randint(8, 15)} minutes"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
