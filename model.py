# demographic segmentation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler

# importing file as df
dataframe = pd.read_csv('sa_client_data_cleaned.csv')

# print df shape
print(dataframe.shape)

# clean data
df2 = dataframe.dropna()
df = df2.drop_duplicates()

# print df shape
print(df.shape)

# getting data stats
print(df.describe())

# getting column names
print(df.columns.tolist())


# removing 9999999 (placeholder values)
features = ['monthly_income_zar', 'avg_monthly_spend_zar', 'card_balance_zar']
qfeatures = df[features]
mask = (qfeatures == 9999999).any(axis=1)
df = df[~mask].copy()

# getting column stats
print(df.describe())

# adding tenure category
df['tenure_category'] = pd.qcut(df['tenure_months'], q=3, labels = ['short', 'medium', 'long'])
print(df.head(5))

# plotting graph
plt.figure(figsize=(10,6))
sns.scatterplot(
    data = df,
    x = "monthly_income_zar",
    y = "avg_monthly_spend_zar",
    hue = "tenure_category"
)
plt.title('Tenure of monthly income (zar) and monthly spend average (zar)')
plt.show()

# adding card category
df['card_category'] = pd.qcut(df['card_balance_zar'], q=3, labels = ['short', 'medium', 'high'])
print(df.head(5))

# plotting graph
plt.figure(figsize=(10,6))
sns.scatterplot(
    data = df,
    x = "monthly_income_zar",
    y = "avg_monthly_spend_zar",
    hue = "card_category"
)
plt.title('Card of monthly income (zar) and monthly spend average (zar)')
plt.show()

# elbow plot
features = ['monthly_income_zar', 'avg_monthly_spend_zar', 'tenure_months']
X = df[features]
X_scaled = RobustScaler().fit_transform(X)

scaler = RobustScaler()                              
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)   
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(np.arange(1, 21), wcss, marker='o')
plt.xticks(range(1, 21))
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow method')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

centroids_scaled = kmeans.cluster_centers_
centroids_raw = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_raw, columns=features)   

# kmeans clustering for tenures

plt.figure(figsize=(10, 6))
plt.scatter(
    X['monthly_income_zar'],
    X['avg_monthly_spend_zar'],
    c=y_kmeans,
    alpha=0.5,
    cmap='viridis'
)
plt.scatter(
    centroids_df['monthly_income_zar'],
    centroids_df['avg_monthly_spend_zar'],
    s=200, marker='X', label='Centroids'
)
plt.xlabel('Monthly income (ZAR)')
plt.ylabel('Avg monthly spend (ZAR)')
plt.title('Tenure cluster my monthly incoe and average monthly spend: KMeans clusters (k=3)')
plt.legend()
plt.show()