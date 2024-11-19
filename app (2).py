import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Load the dataset
df = pd.read_csv("D:/House price/train.csv")

# Fill Missing Values
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.drop(['Alley'], axis=1, inplace=True)
df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu'] = df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'], axis=1, inplace=True)
df['GarageFinish'] = df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond'] = df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
df.drop(['Id'], axis=1, inplace=True)
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df.dropna(inplace=True)

# Encode categorical features
categorical_features = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)




# Split data into X and Y
X = df_encoded.drop(['SalePrice'], axis=1)
Y = df_encoded['SalePrice']

# Split into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
regressor = xgb.XGBRegressor(base_score=0.25, booster='gbtree', learning_rate=0.1, max_depth=2, n_estimators=900)
regressor.fit(X_train, Y_train)

# Extract important features
importance = regressor.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Precompute important categorical and numerical features
top_features = importance_df.head(20)['Feature'].values
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=[np.number]).columns

important_categorical_features = [feature for feature in top_features if any(cat in feature for cat in categorical_features)]
important_numerical_features = [feature for feature in top_features if feature in numerical_features]

# Precompute unique values and ranges
cat_feature_unique_vals = {feature.split('_')[0]: df[feature.split('_')[0]].unique() for feature in important_categorical_features}
num_feature_ranges = {feature: (df[feature].min(), df[feature].max()) for feature in important_numerical_features}

# Streamlit UI Setup
st.title("House Price Prediction")
st.subheader("Enter the details of the house to predict the sale price")

# Collect categorical input
user_inputs = {}

st.sidebar.header("Categorical Inputs")
for feature in important_categorical_features:
    original_feature = feature.split('_')[0]
    unique_values = cat_feature_unique_vals[original_feature]
    user_inputs[feature] = st.sidebar.selectbox(f"Select {original_feature}", options=unique_values)

# Collect numerical input
st.sidebar.header("Numerical Inputs")
for feature in important_numerical_features:
    min_val, max_val = num_feature_ranges[feature]
    user_inputs[feature] = st.sidebar.slider(f"Enter {feature}", min_value=float(min_val), max_value=float(max_val), value=float(min_val))

# Convert user input into a DataFrame
user_input_df = pd.DataFrame([user_inputs])

# Ensure one-hot encoding
user_input_encoded = pd.get_dummies(user_input_df).reindex(columns=X_train.columns, fill_value=0)

# Predict house price based on user input
if st.sidebar.button("Predict House Price"):
    predicted_sale_price = regressor.predict(user_input_encoded)
    st.write(f"### Predicted Sale Price: ${predicted_sale_price[0]:,.2f}")

# Display feature importance as a bar chart
st.subheader("Top 20 Most Important Features for Prediction")
st.bar_chart(importance_df.head(20).set_index('Feature'))
