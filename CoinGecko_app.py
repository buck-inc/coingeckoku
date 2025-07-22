import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Harga Crypto dari CoinGecko", layout="wide")
st.title("📉 Prediksi Harga Crypto dari CoinGecko")

@st.cache_data
def get_data_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "3",
        "interval": "hourly"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["target"] = df["price"].shift(-1)
        return df.dropna()
    except Exception:
        st.error("Gagal mengambil data dari CoinGecko.")
        return pd.DataFrame()

df = get_data_coingecko()

if df.empty or len(df) < 10:
    st.error("Gagal mengambil data dari CoinGecko atau data terlalu sedikit.")
    st.stop()

X = df[["price"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

st.success(f"Model Akurasi: {accuracy:.2f}")
st.subheader("📊 Data Terbaru")
st.dataframe(df.tail())

last_price = df[["price"]].tail(1)
prediksi = model.predict(last_price)
st.success(f"🎯 Prediksi Harga Selanjutnya: ${prediksi[0]:,.2f}")
