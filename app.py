import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Veri yükleme
@st.cache_data
def load_data():
    return pd.read_csv("winequality-red.csv")

df = load_data()

# Streamlit arayüzü
st.title("🍷 Kırmızı Şarap Kalitesi Analizi")
st.subheader("Veri Önizleme")
st.dataframe(df.head())

st.sidebar.header("Kümeleme Ayarları")
n_clusters = st.sidebar.slider("Küme Sayısı", 2, 5, 3)

# Kümeleme işlemi
X = df.drop("quality", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

# Görselleştirmeler
st.subheader(f"{n_clusters} Küme Dağılımı")
fig, ax = plt.subplots()
sns.countplot(x="cluster", data=df, ax=ax)
st.pyplot(fig)

st.subheader("Alkol Seviyesine Göre Kümeler")
fig2, ax2 = plt.subplots()
sns.boxplot(x="cluster", y="alcohol", data=df, ax=ax2)
st.pyplot(fig2)

st.subheader("Küme Ortalamaları")
st.dataframe(df.groupby("cluster").mean())

# PCA Görselleştirme
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

st.subheader("PCA ile Kümeleme")
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.scatterplot(x="PCA1", y="PCA2", hue="cluster", data=df, palette="viridis", ax=ax3)
st.pyplot(fig3)