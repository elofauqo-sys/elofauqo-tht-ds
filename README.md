# 📦 Prediksi Waktu Pengiriman Makanan Menggunakan Machine Learning
<img src = "https://tse4.mm.bing.net/th/id/OIP.-hE8RzfjzaId0mR-OO2qrwHaFj?cb=ucfimgc2&rs=1&pid=ImgDetMain&o=7&rm=3">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://animated-garbanzo-q7g665jj57gxf95-8501.app.github.dev/)

## 📌 Project Overview

Project ini bertujuan untuk memprediksi waktu pengantaran menggunakan dataset pengiriman yang berisi informasi seperti jarak, cuaca, kondisi lalu lintas, jenis kendaraan, waktu pengantaran, serta pengalaman kurir.
Dengan pendekatan Machine Learning, project ini membantu memahami faktor apa yang paling memengaruhi keterlambatan serta menyediakan model prediksi yang akurat.

## 🎯 Objectives

Membangun model Machine Learning untuk memprediksi waktu pengantaran secara akurat.

Mengidentifikasi faktor-faktor penting yang paling memengaruhi durasi pengiriman.

Meningkatkan efisiensi operasional perusahaan logistik melalui prediksi ETA yang lebih baik.

Membandingkan performa beberapa model seperti Linear Regression, Random Forest, dan XGBoost.

Memberikan rekomendasi bisnis berdasarkan hasil analisis dan model.

## 📚 Data Understanding

Dataset berisi data operasional pengiriman yang merekam kondisi pengantaran dan karakteristik kurir.
Variabel-variabel dalam dataset mencakup:
- Informasi pesanan (Order_ID, Distance)
- Kondisi eksternal (Weather, Traffic Level, Time of Day)
- Spesifikasi pengiriman (Vehicle Type, Preparation Time)
- Pengalaman kurir
- Tujuan utama adalah memahami bagaimana variabel-variabel tersebut memengaruhi durasi pengantaran.

## 🗂️ Dataset Description
Feature : Description
Order_ID	: Identifikasi unik pesanan
Distance_km	: Jarak pengantaran dalam kilometer
Weather : Kondisi cuaca saat pengiriman (Clear, Rainy, Stormy, etc.)
Traffic_Level	: Tingkat kemacetan (Low, Medium, High)
Time_of_Day	: Waktu pengiriman (Morning, Afternoon, Evening)
Vehicle_Type :	Jenis kendaraan (Motorcycle, Car, etc.)
Preparation_Time_min :	Lama waktu persiapan restoran
Courier_Experience_yrs	: Pengalaman kurir dalam tahun
Delivery_Time_min	Target : Total waktu pengiriman

## 📊 Exploratory Data Analysis (EDA)
🔹 1. Distribution of Delivery Time
- Rata-rata pengiriman berada pada rentang 20–40 menit.
- Beberapa outlier menunjukkan kasus keterlambatan ekstrem.

🔹 2. Relationship Analysis
- Distance memiliki hubungan positif dengan waktu pengantaran.
- Cuaca buruk (rainy/stormy) cenderung meningkatkan durasi.
- Tingkat kemacetan “High” memiliki waktu pengiriman tertinggi.
- Waktu persiapan restoran memberikan kontribusi signifikan pada total waktu pengiriman.

🔹 3. Categorical Variables
- Motor lebih cepat dibanding mobil.
- Pengalaman kurir menurunkan waktu pengiriman tetapi tidak terlalu signifikan.

🔹 4. Correlation
- Fitur dengan korelasi tertinggi terhadap Delivery Time:
   - Distance_km
   - Preparation_Time_min
   - Traffic_Level
   -Weather

## 📝 Insights Summary

- XGBoost menjadi model terbaik, dengan MAE ~5.9 dan R² ~0.82.
- Variabel yang paling memengaruhi waktu pengiriman:
   - Jarak
   - Waktu persiapan
   - Tingkat kemacetan
   - Cuaca
- Motor adalah kendaraan paling efektif dalam kondisi lalu lintas padat.
- Cuaca buruk dan jam sibuk meningkatkan risiko keterlambatan.
- Ada beberapa outlier yang kemungkinan menunjukkan kasus ekstrem seperti hujan deras atau kemacetan parah.

## 💼 Business Recommendations

✔️ 1. Optimalkan Alur Persiapan di Restoran
- Karena preparation time sangat berpengaruh, restoran dapat:
   - Mengurangi bottleneck dapur
   - Menerapkan sistem early prep untuk pesanan jarak jauh

✔️ 2. Manajemen Alokasi Kurir
- Prioritaskan kurir berpengalaman untuk rute padat dan kondisi cuaca buruk.
- Gunakan motor pada jalur yang sering macet.

✔️ 3. Improve ETA Prediction System
- Implementasi model XGBoost secara real-time meningkatkan akurasi ETA.
- Gunakan prediksi untuk memberikan estimasi waktu yang lebih realistis ke pelanggan.

✔️ 4. Traffic & Weather-Based Routing
- Integrasikan peta lalu lintas
- Atur rute otomatis yang menghindari area macet / hujan lebat

✔️ 5. Monitoring Outliers
- Outlier perlu dianalisis sebagai potensi masalah operasional:
   - Cuaca ekstrem
   - Kurir overload
   - Macet parah
   - Restoran dengan prep time tidak stabil
