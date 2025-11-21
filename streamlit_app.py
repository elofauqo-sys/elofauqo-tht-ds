import streamlit as st 
import pandas as pd

# --- Load Data --- 
@st.cache_data 
def load_data(): 
    return pd.read_csv("data/Food_Delivery_Clean.csv")

df = load_data()

# --- App Title ---
st.title("📦 Food Delivery Time Analysis")
st.write("Aplikasi ini menampilkan informasi awal mengenai dataset Food Delivery Times.")

# --- Dataset Description ---
st.header("📌 Deskripsi Dataset")
st.write("""
Dataset ini berisi informasi mengenai berbagai faktor yang memengaruhi waktu pengantaran makanan. Kolom-kolom yang tersedia meliputi:
- **Distance_km** : Jarak tempuh pengiriman (km)
- **Weather** : Kondisi cuaca
- **Traffic_Level** : Tingkat kemacetan 
- **Time_of_Day** : Waktu pengantaran 
- **Vehicle_Type** : Jenis kendaraan kurir 
- **Preparation_Time_min** : Waktu persiapan makanan 
- **Courier_Experience_yrs** : Lama pengalaman kurir 
- **Delivery_Time_min** : Lama pengiriman (target) """)

# --- Show Data --- 
st.header("📊 Dataframe Preview")
st.dataframe(df)

# --- Dataset Information --- 
st.header("ℹ️ Informasi Dataset") 
st.write(f"Jumlah baris: **{df.shape[0]}**") 
st.write(f"Jumlah kolom: **{df.shape[1]}**")

st.subheader("Tipe Data") 
st.write(df.dtypes)

import streamlit as st 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

# === EDA === 
st.title("🔍 Exploratory Data Analysis (EDA)")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns 
categorical_cols = df.select_dtypes(include=['object', 
                                             'category']).columns

# --------------------------------------- # 
# 1️⃣ DISTRIBUSI VARIABEL NUMERIK 
# ---------------------------------------
st.header("📊 Distribusi Variabel Numerik")

selected_num = st.selectbox( 
    "Pilih variabel numerik:", 
    numeric_cols 
    )

fig, ax = plt.subplots() 
sns.histplot(df[selected_num], kde=True, ax=ax) 
ax.set_title(f"Distribusi {selected_num}") 
st.pyplot(fig)
st.markdown("""
### 🔍 **Insight**
- Distance_km : Distribusi normal, tanpa outlier signifikan.
- Preparation_Time_min : Stabil dan merata, tidak ada nilai ekstrem.
- Courier_Experience_yrs : Variasi wajar, distribusi normal.
- Delivery_Time_min : Right-skewed, ada nilai tinggi tetapi masih realistis.
""")


# --------------------------------------- 
# 2️⃣ DISTRIBUSI VARIABEL KATEGORIKAL
# ---------------------------------------
st.header("📘 Distribusi Variabel Kategorikal")

if len(categorical_cols) > 0: 
    selected_cat = st.selectbox( 
        "Pilih variabel kategorikal:", 
        categorical_cols 
        )
    
    fig, ax = plt.subplots() 
    df[selected_cat].value_counts().plot(kind='bar', ax=ax) 
    ax.set_xlabel(selected_cat) 
    ax.set_ylabel("Count") 
    ax.set_title(f"Distribusi {selected_cat}") 
    st.pyplot(fig) 
else: 
    st.write("Tidak ada variabel kategorikal dalam dataset.")
st.markdown("""
### 🔍 **Insight**
1.Weather
- Cuaca cerah/normal mendominasi data.
- Kondisi ekstrem seperti hujan lebat atau badai jauh lebih sedikit.
2. Traffic_Level
- Mayoritas data berada pada traffic sedang dan tinggi.
- Situasi lalu lintas rendah relatif sedikit.
3. Time_of_Day
- Pesanan banyak terjadi pada siang & sore hari.
- Malam hari paling sedikit.
4. Vehicle_Type
- Motor paling sering digunakan.
- Mobil dan kendaraan lain jauh lebih sedikit.
""")

# --------------------------------------- 
# 3️⃣ KORELASI VARIABEL NUMERIK
# ---------------------------------------
st.header("🧩 Korelasi Variabel Numerik")

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6)) 
sns.heatmap(corr, annot=True, cmap="Blues", ax=ax) 
ax.set_title("Heatmap Korelasi Variabel Numerik") 
st.pyplot(fig)
st.markdown("""
### 🔍 Insight Heatmap Korelasi

- **Distance_km → Delivery_Time_min (0.78)**  
  Faktor yang paling mempengaruhi durasi pengiriman.

- **Preparation_Time_min → Delivery_Time_min (0.31)**  
  Semakin lama persiapan, semakin lama waktu keseluruhan.

- **Courier_Experience_yrs (-0.089)**  
  Pengaruh kecil dan tidak signifikan.

- **Order_ID**  
- Tidak memiliki hubungan dengan variabel lain (identifier saja).
- Tidak ada indikasi **multikolinearitas** antar fitur.
""")



st.header("Spliting")
target = "Delivery_Time_min"
X = df.drop(columns=[target])
y = df[target]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

transformers = []

# numeric
if len(numeric_cols) > 0:
    transformers.append(("num", "passthrough", numeric_cols))

# categorical (sparse=False supaya bisa dipakai matplotlib)
if len(categorical_cols) > 0:
    transformers.append(
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    )

preprocessor = ColumnTransformer(transformers=transformers)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

st.write("X_train shape:", X_train.shape)
st.write("X_test shape:", X_test.shape)
st.write("y_train shape:", y_train.shape)
st.write("y_test shape:", y_test.shape)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================
# Pipeline Linear Regression
# ============================
from sklearn.pipeline import Pipeline

lr_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

# ============================
# Metrics
# ============================
mae = mean_absolute_error(y_test, lr_pred)
mse = mean_squared_error(y_test, lr_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, lr_pred)

st.header("📌 Linear Regression — Evaluation Metrics")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**MSE:** {mse:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.3f}")

st.markdown("""
### 🔍 Insight 

- Model memiliki performa yang baik.
- Dengan MAE ~6 menit dan RMSE ~9 menit, rata-rata prediksi waktu pengiriman hanya meleset beberapa menit dari nilai aktual.
- Nilai R² = 0.825 menunjukkan bahwa model mampu menjelaskan 82.5% variasi waktu pengiriman, sehingga model cukup akurat dan reliabel untuk memprediksi delivery time.
""")

# ============================
# Visualisasi: Actual vs Predicted
# ============================
st.subheader("📊 Actual vs Predicted — Linear Regression")

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(y_test, lr_pred, alpha=0.7)
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()], linestyle="--", linewidth=2)

ax.set_xlabel("Actual Delivery Time")
ax.set_ylabel("Predicted Delivery Time")
ax.set_title("Actual vs Predicted — Linear Regression")

st.pyplot(fig)

# ============================
# 🌳 RANDOM FOREST REGRESSOR
# ============================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === RANDOM FOREST REGRESSOR ===
st.subheader("Random Forest Regressor")

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Train model
rf_pipeline.fit(X_train, y_train)

# Prediction
y_pred_rf = rf_pipeline.predict(X_test)

# Metrics
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_pred_rf)

st.write("### 📊 Random Forest Metrics")
st.write(f"- **MAE:** {rf_mae:.4f}")
st.write(f"- **MSE:** {rf_mse:.4f}")
st.write(f"- **RMSE:** {rf_rmse:.4f}")
st.write(f"- **R²:** {rf_r2:.4f}")
st.markdown("""
### 🔍 Insight 

- Akurasi model sedang.
- Dengan MAE ~7 menit dan RMSE ~10.3 menit, model masih memiliki error yang cukup besar dibanding model sebelumnya.
- Nilai R² = 0.763 berarti model hanya mampu menjelaskan 76% variasi waktu pengiriman, sehingga prediksinya kurang stabil dan tidak sebaik Linear Regression.
""")

# Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(y_test, y_pred_rf)
ax.set_xlabel("Actual Delivery Time")
ax.set_ylabel("Predicted Delivery Time")
ax.set_title("Random Forest: Actual vs Predicted")
st.pyplot(fig)

# === XGBOOST REGRESSOR ===

from xgboost import XGBRegressor
st.subheader("XGBoost Regressor")

xgb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse"
    ))
])

# Train model
xgb_pipeline.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_pipeline.predict(X_test)

# Metrics
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, y_pred_xgb)

st.write("### 📊 XGBoost Metrics")
st.write(f"- **MAE:** {xgb_mae:.4f}")
st.write(f"- **MSE:** {xgb_mse:.4f}")
st.write(f"- **RMSE:** {xgb_rmse:.4f}")
st.write(f"- **R²:** {xgb_r2:.4f}")
st.markdown("""
### 🔍 Insight 

- Model XGBoost memiliki performa lebih baik dari Random Forest, tetapi masih di bawah Linear Regression.
- MAE ~6.83 menit → rata-rata kesalahan prediksi masih cukup baik, tapi tidak seakurat Linear Regression.
- RMSE ~9.82 menit → kesalahan prediksi cukup tinggi pada beberapa kasus ekstrem.
- R² = 0.785 → model mampu menjelaskan 78.5% variabilitas waktu pengiriman, cukup kuat tetapi belum optimal.
""")

# Actual vs Predicted Plot
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(y_test, y_pred_xgb)
ax.set_xlabel("Actual Delivery Time")
ax.set_ylabel("Predicted Delivery Time")
ax.set_title("XGBoost: Actual vs Predicted")
st.pyplot(fig)


# ==========================
#   MODEL COMPARISON
# ==========================

st.header("📌 Model Comparison")

# --- 1. LINEAR REGRESSION ---
lr_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
pred_lr = lr_pipeline.predict(X_test)

lr_results = {
    "MAE": mean_absolute_error(y_test, pred_lr),
    "MSE": mean_squared_error(y_test, pred_lr),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred_lr)),
    "R²": r2_score(y_test, pred_lr)
}

# --- 2. RANDOM FOREST ---
rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
pred_rf = rf_pipeline.predict(X_test)

rf_results = {
    "MAE": mean_absolute_error(y_test, pred_rf),
    "MSE": mean_squared_error(y_test, pred_rf),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred_rf)),
    "R²": r2_score(y_test, pred_rf)
}

# --- 3. XGBOOST ---
xgb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse"
    ))
])
xgb_pipeline.fit(X_train, y_train)
pred_xgb = xgb_pipeline.predict(X_test)

xgb_results = {
    "MAE": mean_absolute_error(y_test, pred_xgb),
    "MSE": mean_squared_error(y_test, pred_xgb),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred_xgb)),
    "R²": r2_score(y_test, pred_xgb)
}
results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "MAE": [lr_results["MAE"], rf_results["MAE"], xgb_results["MAE"]],
    "MSE": [lr_results["MSE"], rf_results["MSE"], xgb_results["MSE"]],
    "RMSE": [lr_results["RMSE"], rf_results["RMSE"], xgb_results["RMSE"]],
    "R²": [lr_results["R²"], rf_results["R²"], xgb_results["R²"]],
})

st.write("### 📊 Perbandingan Kinerja Model")
st.dataframe(results_df)

st.write("### 📈 Model Performance Comparison (RMSE)")

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(["LR", "RF", "XGB"], results_df["RMSE"])
ax.set_ylabel("RMSE")
ax.set_title("RMSE Comparison")
st.pyplot(fig)
st.markdown("""
### 🔍 Insight 

- Linear Regression memiliki performa terbaik, ditunjukkan oleh MAE dan RMSE paling kecil serta R² paling tinggi.
- XGBoost berada di posisi tengah, masih cukup baik tetapi tidak mengungguli model linear.
- Random Forest memiliki performa terendah, kemungkinan karena data tidak terlalu kompleks sehingga model sederhana lebih efektif.
""")

import streamlit as st
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import time


st.header("🔧 Hyperparameter Tuning XGBoost")

# Tombol untuk mulai tuning
if st.button("🚀 Mulai Tuning Model XGBoost"):

    with st.spinner("Proses tuning sedang berlangsung... Mohon tunggu 🙏"):
        
        progress = st.progress(0)
        status_text = st.empty()

        # 1. Preprocessor
# Tentukan kolom numerik dan kategorikal
numerical_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
categorical_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ]
        )

    # 2. Base Model
xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            enable_categorical=False,
            tree_method="hist"
        )

        # 3. Pipeline
pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_model)
        ])

        # 4. Parameter Tuning
param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.01, 0.1],
            "model__subsample": [0.7, 1.0],
            "model__colsample_bytree": [0.7, 1.0]
        }

        # 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 6. GridSearch
grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=0
        )
progress = st.progress(0)
status_text = st.empty()

for i in range(101):
    progress.progress(i)
    status_text.text(f"Progress: {i}%")
    time.sleep(0.05)

        # 7. Fit model
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_mae = -grid_search.best_score_

st.success("Tuning selesai! 🎉")
    
    # --- Tampilkan hasil ---
st.subheader("🏆 Hasil Tuning XGBoost")

col1, col2 = st.columns(2)

with col1:
        st.metric("MAE Terbaik", f"{best_mae:.4f}")

with col2:
        st.metric("Total Kombinasi Dicoba", len(param_grid["model__n_estimators"]) *
                                          len(param_grid["model__max_depth"]) *
                                          len(param_grid["model__learning_rate"]) *
                                          len(param_grid["model__subsample"]) *
                                          len(param_grid["model__colsample_bytree"]))

st.write("### 🔧 Parameter Terbaik")
st.json(best_params)

    # --- Ekspander untuk detail hasil tuning ---
with st.expander("📄 Lihat semua parameter grid search"):
        st.write(param_grid)

with st.expander("📘 Model Lengkap Setelah Tuning"):
        st.write(grid_search.best_estimator_)

pipeline_xgb = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("xgb", XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    ))
])

pipeline_xgb.fit(X_train, y_train)
model_final = pipeline_xgb

st.subheader("📊 Actual vs Predicted")

# Prediksi
y_pred = model_final.predict(X_test)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)
st.markdown("""
### 🔍 Insight 
Model menunjukkan hubungan yang kuat antara nilai aktual dan prediksi. Titik data mengikuti garis diagonal, menandakan prediksi model stabil dan akurat.
""")

st.subheader("📉 Residual Plot")

residuals = y_test - y_pred

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_pred, residuals)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot")

st.pyplot(fig)
st.markdown("""
### 🔍 Insight 
Residual tersebar acak di sekitar garis nol, menandakan model tidak mengalami bias dan mampu menangkap pola data dengan baik.
""")

st.subheader("🔥 Feature Importance (XGBoost)")

# Ambil model XGB-nya
xgb_model = model_final.named_steps["xgb"]

# Ambil nama fitur dari OneHotEncoder
feature_names = (
    model_final.named_steps["preprocess"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
)

# Gabung dengan numeric columns
all_features = list(numerical_cols) + list(feature_names)

importances = xgb_model.feature_importances_

# Plot
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(all_features, importances)
ax.set_title("XGBoost Feature Importance")
ax.set_xlabel("Importance Score")

st.pyplot(fig)
st.markdown("""
### 🔍 Insight 
- Dua fitur paling berpengaruh adalah Log_Distance dan Distance_km (menyumbang skor penting terbesar).
- Fitur lain memiliki kontribusi jauh lebih kecil terhadap prediksi.
- Model sangat bergantung pada fitur utama tersebut untuk menentukan waktu pengiriman. Ini wajar karena semakin jauh jaraknya → semakin lama waktu antar.
""")
