import pandas as pd
from pycaret.classification import *
from sklearn.cluster import DBSCAN
import joblib

# ================= LOAD DATA =================
df1 = pd.read_csv("combined_accident_data.csv")
df2 = pd.read_csv("combined_accident_data1.csv")
df3 = pd.read_csv("only_road_accidents_data_month2.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)

# ================= CLEAN =================
df = df.dropna()

# Example target (modify based on your dataset)
df['risk'] = df['severity']  # adjust column name

# ================= AUTO ML =================
clf = setup(
    data=df,
    target='risk',
    session_id=42,
    normalize=True,
    remove_outliers=True,
    silent=True
)

best_model = compare_models()
tuned_model = tune_model(best_model)
final_model = finalize_model(tuned_model)

save_model(final_model, 'model')

# ================= DBSCAN =================
if 'lat' in df.columns and 'lon' in df.columns:
    db = DBSCAN(eps=0.01, min_samples=5)
    df['cluster'] = db.fit_predict(df[['lat', 'lon']])
    df[['lat', 'lon', 'cluster']].to_csv("clusters.csv", index=False)

# ================= SAVE =================
joblib.dump(final_model, "model.pkl")

print("✅ Training Complete")
