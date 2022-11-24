import streamlit as st
import pandas as pd
from PIL import Image
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB


st.write("""
# Web Apps - Klasifikasi Bunga Iris
""")

img = Image.open('iris1.png')
st.image(img, use_column_width=False)

st.sidebar.header('Parameter Inputan')

def input_user():
    sepal_length = st.sidebar.slider('Panjang sepal', 4.3, 7.9, 5.4)
    sepal_width  = st.sidebar.slider('lebar sepal', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Panjang petal', 1.0, 6.9, 1.3)
    petal_width  = st.sidebar.slider('lebar petal',0.1 , 2.5 , 0.2)

    data = {'Panjang Sepal' : sepal_length,
             'lebar sepal' : sepal_width,
             'Panjang Petal' : petal_length,
              'Lebar Petal' : petal_width}
    fitur = pd.DataFrame(data, index=[0])
    return fitur

df = input_user()

st.subheader('Parameter Inputan')
st.write(df)

IRIS = datasets.load_iris()

X = IRIS.data
Y = IRIS.target

model = GaussianNB()
model.fit(X, Y)

prediksi = model.predict(df)
prediksi_proba = model.predict_proba(df)

st.subheader('Label Kelas dan Nomor Indeks yang sesuai')
st.write(IRIS.target_names)

st.subheader('Prediksi (Hasil Klasifikasi)')
st.write(IRIS.target_names[prediksi])

st.subheader('Probabilitas Hasil Prediksi (Klasifikasi)')
st.write(prediksi_proba)
