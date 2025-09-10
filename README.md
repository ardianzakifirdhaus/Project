# 🏡 Prediksi Harga Rumah  
*Course DQ Lab Bootcamp AI & Machine Learning for Beginners - Batch 18*  

🔗 **Demo Aplikasi:** [Streamlit App](https://project-prediksi-harga-rumah.streamlit.app/)  
📂 **Dataset:** [Kaggle - Housing Prices Regression](https://www.kaggle.com/datasets/denkuznetz/housing-prices-regression/data?select=real_estate_dataset.csv)  

---

## 🎯 Tujuan Proyek
1. Melakukan **analisis dataset** properti.  
2. Membuat model **prediksi harga rumah** berdasarkan atribut-atribut rumah.  

---

## 📊 Dataset & Kolom
Dataset berisi informasi properti dengan kolom berikut:  

- **ID** : ID unik untuk setiap properti  
- **Square_Feet** : Luas rumah (m²)  
- **Num_Bedrooms** : Jumlah kamar tidur  
- **Num_Bathrooms** : Jumlah kamar mandi  
- **Num_Floors** : Jumlah lantai  
- **Year_Built** : Tahun pembangunan rumah  
- **Has_Garden** : Apakah ada taman (1 = ya, 0 = tidak)  
- **Has_Pool** : Apakah ada kolam renang (1 = ya, 0 = tidak)  
- **Garage_Size** : Luas garasi (m²)  
- **Location_Score** : Skor lingkungan (0–10, makin tinggi makin bagus)  
- **Distance_to_Center** : Jarak ke pusat kota (km)  
- **Price** : Harga rumah (variabel target yang diprediksi)
- 
## 🗂️ Arsitektur Workflow

flowchart LR
    A[Dataset] --> B[EDA & Preprocessing]
    B --> C[Split Data (Train/Test)]
    C --> D[Model Training]
    D --> E[Model Evaluation]
    E --> F{Pilih Model Terbaik}
    F --> G[Deploy via Streamlit]
---

## 📈 Hasil Visualisasi Model

### 🔹 Linear Regression
![Linear Evaluation](https://github.com/ardianzakifirdhaus/Project/blob/main/linear_evaluation.png)

### 🔹 Decision Tree
![Decision Tree Evaluation](https://github.com/ardianzakifirdhaus/Project/blob/main/decision_tree_evaluation.png)

### 🔹 Random Forest
![Random Forest Evaluation](https://github.com/ardianzakifirdhaus/Project/blob/main/random_forest_evaluation.png)

### 🔹 AdaBoost
![AdaBoost Evaluation](https://github.com/ardianzakifirdhaus/Project/blob/main/adaboost_evaluation.png)

### 🔹 Gradient Boosting
![Gradient Boosting Evaluation](https://github.com/ardianzakifirdhaus/Project/blob/main/gradient_boosting_evaluation.png)

---

## 🏆 Evaluasi Model

| Model                     | MSE              | RMSE            | R²   |
|----------------------------|-----------------|----------------|------|
| **Linear Regression**      | 437,730,359.71  | 20,922.01      | 0.97 |
| Decision Tree Regressor    | 5,218,762,529.85| 72,241.00      | 0.65 |
| Random Forest Regressor    | 2,621,324,915.51| 51,198.88      | 0.82 |
| AdaBoost Regressor         | 2,996,533,905.68| 54,740.61      | 0.80 |
| Gradient Boosting Regressor| 1,109,632,350.13| 33,311.14      | 0.92 |

---

## 📌 Kesimpulan
Berdasarkan hasil evaluasi:  
- **Linear Regression** memberikan performa terbaik dengan nilai **R² = 0.97** dan error yang relatif rendah.  
- Model lain (Decision Tree, Random Forest, AdaBoost, Gradient Boosting) masih cukup baik, namun tidak sebaik Linear Regression pada dataset ini.  

👉 Maka dipilih **Linear Regression sebagai model utama** untuk prediksi harga rumah.  

---

## 👨‍💻 Pengembang
- [@ardianzakifirdhaus](https://github.com/ardianzakifirdhaus)  
