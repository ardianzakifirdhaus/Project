import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from scipy import stats
import pickle
import numpy as np
import time
from sklearn.pipeline import Pipeline
import warnings

st.set_page_config(page_title="Dashboard Real Estate App", layout="wide")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ardianzakifirdhaus/Project/main/real_estate_dataset.csv" 
    return pd.read_csv(url)

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Beranda", "📊 Eksplorasi Data", "🤖 Prediksi Harga Rumah", "📈 Performa Model"])
st.sidebar.info("Projek Aplikasi ini dibuat untuk Final Projek dari DQ Lab Bootcamp AI & Machine Learning for Beginner Batch 18 oleh MAI1824_Ardian Zaki Firdhaus.")

# --- Beranda ---
if page == "🏠 Beranda":
    st.title("🏠 Aplikasi Prediksi Harga Rumah")
    st.write("""
    Aplikasi ini menggunakan dataset **real_estate_dataset.csv** dari Kaggle — https://www.kaggle.com/datasets/denkuznetz/housing-prices-regression/data?select=real_estate_dataset.csv 
    untuk demonstrasi pemodelan harga unit area berdasarkan atribut properti.""")
    st.write("""
    **Dataset ini berisi informasi tentang properti real estate, termasuk:**    
    Dataset Columns:
             
    1. ID: A unique identifier for each property   
    2. Square_Feet: The area of the property in square meters.      
    3. Num_Bedrooms: The number of bedrooms in the property.
    4. Num_Bathrooms: The number of bathrooms in the property.
    5. Num_Floors: The number of floors in the property.    
    6. Year_Built: The year the property was built.
    7. Has_Garden: Indicates whether the property has a garden (1 for yes, 0 for no).  
    8. Has_Pool: Indicates whether the property has a pool (1 for yes, 0 for no).     
    9. Garage_Size: The size of the garage in square meters.     
    10. Location_Score: A score from 0 to 10 indicating the quality of the neighborhood (higher scores indicate better neighborhoods).    
    11. Distance_to_Center: The distance from the property to the city center in kilometers.     
    12. Price: The target variable that represents the price of the property. This is the value we aim to predict.
    """)
    st.write("Contoh 5 baris tabulasi data:")
    st.dataframe(df.head())
    st.write("Total data:", len(df))
    
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    col1, col2 = st.columns(2)
    with col1:
        st.text("Informasi tipe data:\n" + info_str)
    with col2:
        st.write("Tabel atribut data null/kosong:", df.isna().sum())
        st.write("Total data duplikat:", df.duplicated().sum())
    
# --- Eksplorasi Data ---
elif page == "📊 Eksplorasi Data":
    # Preparing the dataset
    X = df.drop(columns=['Price','ID']) # Feature untuk memprediksi
    y = df[['Price']] # Target yang akan diprediksi

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    st.title("📊 Eksplorasi Data")
    st.write("1. Beberapa statistik dasar data:")
    st.write(df.describe())

    st.write("2. Uji Asumsi Klasik Regresi:")

    # Train a linear regression model
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

    # --- (a) Linearitas: Residual vs Fitted ---
    fitted = model.fittedvalues
    residuals = model.resid

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=fitted, y=residuals, ax=ax1)
        ax1.axhline(0, color='red', linestyle='--')
        ax1.set_title("Uji Linearitas: Residual vs Fitted")
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        st.pyplot(fig1)

    with col2:
        st.write("Uji Independensi Error: Durbin-Watson")
        dw = durbin_watson(residuals)
        st.write(f"Durbin-Watson: {dw:.2f} (≈2 → tidak ada autokorelasi)")

    # --- (c) Normalitas Error ---
    col3, col4 = st.columns(2)

    with col3:
        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title("Distribusi Residual")
        st.pyplot(fig2)

    with col4:
        fig3 = sm.qqplot(residuals, line='45', fit=True)
        plt.title("Q-Q Plot Residual")
        st.pyplot(fig3)

    shapiro_test = shapiro(residuals)
    st.write(f"Shapiro-Wilk p-value: {shapiro_test.pvalue:.4f} (p>0.05 → residual normal)")

    # --- (d) Homoskedastisitas: Breusch-Pagan ---
    X_train_with_constant = sm.add_constant(X_train)
    bp_test = het_breuschpagan(residuals, X_train_with_constant)
    st.write(f"Breusch-Pagan p-value: {bp_test[1]:.4f} (p>0.05 → homoskedastisitas terpenuhi)")

    # ----(e) calculate VIF scores for each feature
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    from statsmodels.tools.tools import add_constant

    X_vif = add_constant(X_train)
    vif_df = pd.DataFrame([vif(X_vif.values, i)
                for i in range(X_vif.shape[1])],
                index=X_vif.columns).reset_index()
    vif_df.columns = ['feature','vif_score']
    vif_df = vif_df.loc[vif_df.feature!='const']
    st.write("VIF Scores:")
    st.dataframe(vif_df)
    st.write("Ketarangan VIF Scores:" \
    "\n- VIF mendekati 1: Tidak ada multikolinearitas yang signifikan" \
    "\n- 1 <= VIF < 4: Ada multikolinearitas yang rendah" \
    "\n- 4 <= VIF < 10: Ada multikolinearitas yang moderat" \
    "\n- VIF >= 10: Ada multikolinearitas yang tinggi")

    st.write("**3. Korelasi Antar Variabel**")
    data_train = pd.concat([X_train, y_train], axis=1)
    corr = data_train.corr()

    fig4, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='crest', ax=ax, annot_kws={"size": 5}, cbar_kws={"shrink": 0.7}) 
    ax.set_title("Korelasi Antar Variabel", fontsize=7)
    ax.set_xlabel("Fitur", fontsize=5)
    ax.set_ylabel("Fitur", fontsize=5)
    ax.tick_params(axis='x', labelsize=5) 
    ax.tick_params(axis='y', labelsize=5) 
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5) 
    st.pyplot(fig4)

    st.write("**4. Sebaran Atribut Data**")
    df_variabel = df.drop(columns=['ID'])
    fig, axs = plt.subplots(nrows=2,ncols=5,sharey=True,figsize=(15,10))
    colors=['blue']
    axs = axs.flatten() # Flatten the 2D array of Axes objects
    for j, i in enumerate(df_variabel.columns[:-1]): # Iterate through columns excluding 'Price'
        sns.scatterplot(data=df_variabel, y='Price', x=i, ax=axs[j], color=colors[j % len(colors)])
    plt.tight_layout()
    st.pyplot(fig) 

    st.write("**5. Scaling Nilai Features (X)**")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.drop(columns=['Price', 'ID']))
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns.drop(['Price', 'ID']))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=scaled_df, orient='h', color='skyblue', ax=ax)
    ax.set_title("Boxplot untuk semua parameter")
    ax.set_xlabel("Nilai Normalisasi")
    ax.set_ylabel("Parameter")
    st.pyplot(fig)

# --- Prediksi ---
elif page == "🤖 Prediksi Harga Rumah":
    st.title("🤖 Prediksi Harga Rumah")
    scaler = StandardScaler()
    X = df.drop(columns=['Price', 'ID'])
    y = df[['Price']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = LinearRegression()
    pipe = Pipeline([
        ("preprocessor", scaler),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)

    pklname = "linear_harga_rumah.pkl"
    with open(pklname, 'wb') as file:
        pickle.dump(pipe, file)

    st.write("Masukkan atribut rumah untuk prediksi harga:")
    def user_input_features():
        st.header('Input Manual')
        data = {}
        for col in df.columns:
            if col not in ['Price', 'ID']:
                # Slider bilangan bulat untuk fitur tertentu
                if col in ['Num_Bedrooms', 'Num_Bathrooms', 'Num_Floors', 'Year_Built', 'Has_Garden', 'Has_Pool']:
                    min_val = int(df[col].min())
                    max_val = int(df[col].max())
                    mean_val = int(df[col].mean())
                    data[col] = st.slider(col, min_value=min_val, max_value=max_val, value=mean_val, step=1)
                # Slider float untuk fitur numerik lain
                elif df[col].dtype in ['int64', 'float64']:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    data[col] = st.slider(col, min_value=min_val, max_value=max_val, value=mean_val)
                # Selectbox untuk fitur kategorikal
                else:
                    data[col] = st.selectbox(col, options=df[col].unique())
        features = pd.DataFrame(data, index=[0])
        return features
    
    input_df = user_input_features()

    if st.button('Predict!'):
        st.write("Data input:", input_df)
        with open(pklname, 'rb') as file:
            loaded_pipe = pickle.load(file)
        prediction = loaded_pipe.predict(input_df)
        output = float(prediction[0])
        st.subheader('Prediction:')
        with st.spinner('Wait for it...'):
            time.sleep(2)
        st.success(f"Prediksi Harga Rumah adalah {output:.0f}")

# --- Performa Model ---
elif page == "📈 Performa Model":
    st.title("📈 Performa Model")
    st.subheader("Evaluasi Model Regresi Linier dibandingkan algoritma lainnya")
    st.write(
        "Nilai RMSE dan R^2 untuk masing-masing model regresi yang telah dilatih pada dataset ini "
        "menunjukkan performa model—semakin tinggi nilai R^2 maka semakin baik, dan semakin kecil nilai RMSE maka semakin akurat."
    )

    image_items = [
        ("https://raw.githubusercontent.com/ardianzakifirdhaus/Project/main/linear_evaluation.png", "Performa Model Regresi Linier"),
        ("https://raw.githubusercontent.com/ardianzakifirdhaus/Project/main/decision_tree_evaluation.png", "Performa Decision Tree"),
        ("https://raw.githubusercontent.com/ardianzakifirdhaus/Project/main/random_forest_evaluation.png", "Performa Random Forest Regressor"),
        ("https://raw.githubusercontent.com/ardianzakifirdhaus/Project/main/adaboost_evaluation.png", "Performa AdaBoost"),
        ("https://raw.githubusercontent.com/ardianzakifirdhaus/Project/main/gradient_boosting_evaluation.png", "Performa Gradient Boosting Regressor"),
    ]

    cols = st.columns(2)
    for idx, (img_url, caption) in enumerate(image_items):
        with cols[idx % 2]:
            st.image(img_url, caption=caption, use_container_width=True)  # gunakan use_column_width, bukan use_container_width
        try:
            st.image(img_url, caption=caption, use_container_width=True)

    st.write(
        """Kesimpulan:
        
        1. Linear Regresi
        *  Mean Squared Error: 437730359.7082737
        *  Root Mean Squared Error: 20922.006588954933
        *  R^2 Score: 0.97

        2. Decision Tree Regressor
        *  Mean Squared Error: 5218762529.846892
        *  Root Mean Squared Error: 72241.00310659378
        *  R^2 Score: 0.65

        3. Random Forest Regressor
        *  Mean Squared Error: 2621324915.507257
        *  Root Mean Squared Error: 51198.87611566544
        *  R^2 Score: 0.82

        4. AdaBoost Regressor
        *  Mean Squared Error: 2996533905.676023
        *  Root Mean Squared Error: 54740.
        *  R^2 Score: 0.80

        5. Gradient Boosting Regressor
        *  Mean Squared Error: 1109632350.127504
        *  Root Mean Squared Error: 33311.144533436614
        *  R^2 Score: 0.92

        Dari hasil evaluasi performa model, dapat disimpulkan bahwa model Regresi Linier memiliki performa yang baik dalam memprediksi harga rumah pada dataset ini.
        Meskipun ada beberapa model lain yang juga menunjukkan performa yang baik, Regresi Linier tetap menjadi pilihan yang sederhana dan efektif untuk masalah ini."""
    )



