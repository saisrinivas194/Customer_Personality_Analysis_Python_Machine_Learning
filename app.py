import streamlit as st
import numpy as np
import pandas as pd
import threadpoolctl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load and preprocess the dataset
df = pd.read_excel('marketing_campaign1 (1).xlsx')

# Fill missing values in 'Income' column with median
df['Income'] = df['Income'].fillna(df['Income'].median())

# Drop unnecessary columns
df.drop(columns=["Z_CostContact", "Z_Revenue"], inplace=True)

# Combine conflicting categories in 'Education' column
df['Education'] = df['Education'].replace(['PhD', '2n Cycle', 'Graduation', 'Master'], 'Post Graduate')
df['Education'] = df['Education'].replace(['Basic'], 'Under Graduate')

# Combine conflicting categories in 'Marital_Status' column
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'], 'Relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'], 'Single')

# Combine 'Kidhome' and 'Teenhome' into 'Children' column
df['Children'] = df['Kidhome'] + df['Teenhome']

# Calculate 'Expenditure' by summing up expenditure columns
expenditure_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Expenditure'] = df[expenditure_columns].sum(axis=1)

# Calculate 'Overall_Accepted_Cmp' by summing up campaign acceptance columns
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
df['Overall_Accepted_Cmp'] = df[campaign_columns].sum(axis=1)

# Combine different types of purchases into 'NumTotalPurchases'
df['NumTotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']

# Calculate 'Customer_Age' based on 'Year_Birth'
current_year = pd.Timestamp('now').year
df['Customer_Age'] = current_year - df['Year_Birth']

# Drop unnecessary columns
columns_to_drop = ['Year_Birth', 'ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                   'NumWebVisitsMonth', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases',
                   'Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                   'MntGoldProds']
df.drop(columns=columns_to_drop, inplace=True)

# Convert 'Dt_Customer' to datetime and calculate 'Customer_Shop_Days'
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Customer_Shop_Days'] = (pd.Timestamp('now') - df['Dt_Customer']).dt.days

# Drop unnecessary columns
df.drop(columns=['Dt_Customer', 'Recency', 'Complain', 'Response'], inplace=True)

# Encode categorical variables
df['Education'] = df['Education'].map({'Under Graduate': 0, 'Post Graduate': 1})
df['Marital_Status'] = df['Marital_Status'].map({'Single': 0, 'Relationship': 1})

# Ensure no NaN values are present
df.dropna(inplace=True)

# Scale numeric columns
scaler = StandardScaler()
col_scale = ['Income', 'Children', 'Expenditure', 'Overall_Accepted_Cmp', 'NumTotalPurchases', 'Customer_Age', 'Customer_Shop_Days']
df[col_scale] = scaler.fit_transform(df[col_scale])

# Train clustering models (Example: KMeans)
kmeans_model = KMeans(n_clusters=5, random_state=42)
df['cluster_Kmeans'] = kmeans_model.fit_predict(df)

# Streamlit application
def main():
    st.title("Customer Segmentation and Analysis")

    # Input fields for user to enter data
    st.sidebar.header("Input Customer Data")
    income = st.sidebar.number_input("Income", min_value=0, max_value=1000000, value=50000)
    children = st.sidebar.number_input("Children", min_value=0, max_value=10, value=2)
    expenditure = st.sidebar.number_input("Expenditure", min_value=0, max_value=100000, value=2000)
    accepted_cmp = st.sidebar.number_input("Overall Accepted Campaigns", min_value=0, max_value=10, value=1)
    total_purchases = st.sidebar.number_input("Total Purchases", min_value=0, max_value=100, value=5)
    customer_age = st.sidebar.number_input("Customer Age", min_value=0, max_value=100, value=30)
    shop_days = st.sidebar.number_input("Customer Shop Days", min_value=0, max_value=10000, value=365)
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Relationship"])
    education = st.sidebar.selectbox("Education Level", ["Under Graduate", "Post Graduate"])

    # Encode categorical features
    education_map = {'Under Graduate': 0, 'Post Graduate': 1}
    marital_status_map = {'Single': 0, 'Relationship': 1}

    input_data = np.array([[income, children, expenditure, accepted_cmp, total_purchases, customer_age, shop_days,
                            education_map[education], marital_status_map[marital_status]]])

    # Scale the input data
    input_data[:, :7] = scaler.transform(input_data[:, :7])

    # Predict cluster using KMeans model
    cluster_kmeans = kmeans_model.predict(input_data)

    # Display the results
    st.write(f"The customer belongs to cluster: {cluster_kmeans[0] + 1}")

if __name__ == "__main__":
    main()

