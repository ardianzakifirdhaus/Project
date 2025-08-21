# Project
Course DQ Lab Bootcamp AI & Machine Learning for Beginners 
Batch 18

# **Prediksi Harga Rumah **

**Tujuan:

1. Analisis Dataset
2. Prediksi Harga Rumah berdasarkan atribut nilai rumah**

Dataset:
https://www.kaggle.com/datasets/denkuznetz/housing-prices-regression/data?select=real_estate_dataset.csv

Dataset Columns:

ID: A unique identifier for each property

Square_Feet: The area of the property in square meters.

Num_Bedrooms: The number of bedrooms in the property.

Num_Bathrooms: The number of bathrooms in the property.

Num_Floors: The number of floors in the property.

Year_Built: The year the property was built.

Has_Garden: Indicates whether the property has a garden (1 for yes, 0 for no).

Has_Pool: Indicates whether the property has a pool (1 for yes, 0 for no).

Garage_Size: The size of the garage in square meters.

Location_Score: A score from 0 to 10 indicating the quality of the neighborhood (higher scores indicate better neighborhoods).

Distance_to_Center: The distance from the property to the city center in kilometers.

Price: The target variable that represents the price of the property. This is the value we aim to predict.

Hasil 
https://github.com/ardianzakifirdhaus/Project/blob/main/linear_evaluation.png
https://github.com/ardianzakifirdhaus/Project/blob/main/decision_tree_evaluation.png
https://github.com/ardianzakifirdhaus/Project/blob/main/random_forest_evaluation.png
https://github.com/ardianzakifirdhaus/Project/blob/main/adaboost_evaluation.png
https://github.com/ardianzakifirdhaus/Project/blob/main/gradient_boosting_evaluation.png


# **Kesimpulan: **
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
*  Root Mean Squared Error: 54740.60563855704
*  R^2 Score: 0.80

5. Gradient Boosting Regressor
*  Mean Squared Error: 1109632350.127504
*  Root Mean Squared Error: 33311.144533436614
*  R^2 Score: 0.92

Berdasarkan percobaan pemodelan maka dipilih **model linear regresi**



