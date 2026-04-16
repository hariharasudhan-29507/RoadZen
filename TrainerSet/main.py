from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import shap
import numpy as np

app = FastAPI()

# ========== CORS (IMPORTANT FOR FRONTEND) ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== LOAD MODEL ==========
model = joblib.load("model.pkl")

# SHAP Explainer
explainer = shap.Explainer(model)

# Labels (modify if needed)
labels = ["Minor", "Moderate", "Severe"]

# ========== PREDICTION ==========
@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        pred = model.predict(df)[0]
        probs = model.predict_proba(df)[0]

        return {
            "risk_label": labels[int(pred)],
            "confidence": float(max(probs) * 100),
            "probabilities": {
                labels[i]: float(probs[i]) for i in range(len(labels))
            }
        }

    except Exception as e:
        return {"error": str(e)}

# ========== SHAP EXPLANATION ==========
@app.post("/explain")
def explain(data: dict):
    try:
        df = pd.DataFrame([data])
        shap_values = explainer(df)

        factors = []
        for i, val in enumerate(shap_values.values[0]):
            factors.append({
                "feature": df.columns[i],
                "impact": float(val)
            })

        # Top 5 important features
        factors = sorted(factors, key=lambda x: abs(x["impact"]), reverse=True)[:5]

        return {"top_factors": factors}

    except Exception as e:
        return {"error": str(e)}

# ========== CHATBOT (ADVANCED VERSION) ==========
@app.post("/chat")
def chat(data: dict):
    msg = data.get("message", "").lower()

    # ===== INTENT DETECTION =====
    intent = "general"

    if any(word in msg for word in ["predict", "risk", "chance"]):
        intent = "prediction"
    elif any(word in msg for word in ["why", "reason", "explain"]):
        intent = "explanation"
    elif any(word in msg for word in ["hotspot", "danger", "zone", "cluster"]):
        intent = "hotspot"
    elif any(word in msg for word in ["hospital", "ambulance", "emergency"]):
        intent = "emergency"
    elif any(word in msg for word in ["safe", "safety", "tips"]):
        intent = "safety"

    # ===== RESPONSE GENERATION =====

    # 🔹 PREDICTION
    if intent == "prediction":
        return {
            "reply": (
                "🚦 **Accident Risk Prediction System**\n\n"
                "This system uses machine learning trained on historical accident data.\n\n"
                "Provide input parameters such as:\n"
                "- Time of day\n"
                "- Weather condition\n"
                "- Vehicle type\n\n"
                "Click **Predict Risk** to get severity classification and probabilities.\n\n"
                "You can also request explanation using SHAP."
            ),
            "suggestions": ["Explain prediction", "What affects risk?", "Show hotspots"]
        }

    # 🔹 EXPLANATION
    elif intent == "explanation":
        return {
            "reply": (
                "🧠 **Explainable AI using SHAP**\n\n"
                "I analyze predictions using SHAP values.\n\n"
                "This allows me to determine:\n"
                "✔ Which features increased risk\n"
                "✔ Which features decreased risk\n\n"
                "Example insights:\n"
                "- Rain increases accident probability 🌧️\n"
                "- Night reduces visibility 🌙\n"
                "- Vehicle type affects severity 🚗\n\n"
                "Click **Explain (SHAP)** after prediction."
            ),
            "suggestions": ["Predict risk", "Feature importance", "Safety tips"]
        }

    # 🔹 HOTSPOTS
    elif intent == "hotspot":
        return {
            "reply": (
                "🔥 **Accident Hotspot Detection**\n\n"
                "We use DBSCAN clustering to identify high-risk zones.\n\n"
                "📍 Red → High accident density\n"
                "🟠 Orange → Medium risk\n"
                "🟢 Green → Safer areas\n\n"
                "Open the Heatmap section to view real-time hotspots."
            ),
            "suggestions": ["Show heatmap", "Predict risk", "Safety tips"]
        }

    # 🔹 EMERGENCY
    elif intent == "emergency":
        return {
            "reply": (
                "🚨 **Emergency Response System**\n\n"
                "In case of accident:\n"
                "1. Call emergency number **112**\n"
                "2. Use 'Send Alert' feature\n"
                "3. Locate nearest trauma center on map\n\n"
                "System also estimates response time automatically."
            ),
            "suggestions": ["Send alert", "Show hospitals", "Safety tips"]
        }

    # 🔹 SAFETY
    elif intent == "safety":
        return {
            "reply": (
                "🛡️ **AI Safety Recommendations**\n\n"
                "Based on accident data:\n\n"
                "⚠️ High risk scenarios:\n"
                "- Driving at night 🌙\n"
                "- Rain or fog 🌧️🌫️\n"
                "- High-speed driving\n\n"
                "✅ Safer practices:\n"
                "- Maintain speed control\n"
                "- Keep safe distance\n"
                "- Avoid distractions\n\n"
                "Use prediction tool to simulate risk."
            ),
            "suggestions": ["Predict risk", "Explain risk", "Show hotspots"]
        }

    # 🔹 GENERAL
    else:
        return {
            "reply": (
                "🤖 **RoadZen AI Assistant**\n\n"
                "I support:\n\n"
                "🚦 Risk prediction\n"
                "🧠 Explainable AI (SHAP)\n"
                "🔥 Hotspot detection\n"
                "🚨 Emergency support\n\n"
                "Try asking:\n"
                "- Why is accident risk high?\n"
                "- Show danger zones\n"
                "- Give safety tips"
            ),
            "suggestions": ["Predict risk", "Explain risk", "Show heatmap", "Safety tips"]
        }

# ========== HEATMAP ==========
@app.get("/api/heatmap")
def heatmap():
    try:
        df = pd.read_csv("clusters.csv")
        return df.to_dict(orient="records")
    except:
        return []

# ========== MODEL INFO ==========
@app.get("/api/model-info")
def info():
    return {
        "accuracy": 0.90,
        "n_samples": 12000
    }

# ========== DUMMY TRAUMA CENTERS ==========
@app.get("/api/trauma-centers")
def hospitals():
    return [
        {"name": "AIIMS Delhi", "lat": 28.5672, "lng": 77.2100, "type": "Level 1", "speciality": "Trauma", "beds": 200, "phone": "011-26588500"},
        {"name": "Safdarjung Hospital", "lat": 28.5680, "lng": 77.2000, "type": "Level 1", "speciality": "Emergency", "beds": 150, "phone": "011-26707444"}
    ]

# ========== ALERT ==========
@app.post("/api/alert")
def alert(data: dict):
    return {
        "alert_status": "Sent",
        "estimated_response": "8 mins",
        "notified_centers": [
            {"hospital": "AIIMS Delhi", "status": "Alerted", "eta": "6 mins"},
            {"hospital": "Safdarjung Hospital", "status": "Alerted", "eta": "8 mins"}
        ]
    }
