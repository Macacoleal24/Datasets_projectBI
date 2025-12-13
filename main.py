import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  

st.set_page_config(
    page_title="Spotify Dashboard Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Spotify Dashboard Analysis")
st.markdown("Proyecto final - Inteligencia de Negocio y Soluciones de Ciencia de Datos – Universidad Panamericana CDMX")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Macacoleal24/Datasets_projectBI/refs/heads/main/final_dataset_V4.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Documentación General","EDA","Metodología", "Modelo de ML", "Conclusión"])

with  tab1:
    st.subheader("Documentación General del proyecto")
    st.markdown("""<img src="https://posgrados-panamericana.up.edu.mx/hs-fs/hubfs/logo%20posgrados%20con%20espacio.png?width=343&name=logo%20posgrados%20con%20espacio.png" width="150" >

# Tablero Interactivo de Análisis de Música

## Integrantes
* Alejandro Alvarez Grijalva - 0240272
* Sergio Carlos Mijangos Carbajal - 0246337
* Lael Morales Ponce - 0249034
* Ricardo Alfonso Zepahua Enríquez - 0243352
## Universidad Panamericana - Campus Mixcoac, Ciudad de México

Este repositorio contiene el desarrollo de un streamlit interactivo orientado al análisis de tendencias musicales utilizando temas vistos durante el curso.

* Objetivo del proyecyto
El objetivo es aplicar técnicas de análisis de datos y visualización para identificar patrones relevantes en:
* Popularidad de canciones
* Popularidad y seguidores de artistas
* Géneros más publicados por año
* Comportamientos por mes
* Clustering no supervisado para segmentación

El proyecto integra limpieza de datos y visualizaciones diseñadas para generar insights accionables para el analisis de datos de Inteligencia de Negocios.

## Preparación y Limpieza de Datos
Las principales tareas de preprocesamiento incluyeron:
* Codificación y normalización para modelos de Machine Learning
*

## Modelos de Machine Learning (Aprendizaje No Supervisado)
Se utilizaron diferentes algoritmos de clustering para ver que modelo de ML era el mejor para nuetros datos:

* K-Means
* MeanShift
* Affinity Propagation
* DBSCAN
* HDBSCAN
* OPTICS
* BIRCH
* Agglomerative Clustering
* Spectral Clustering

Evaluandolos por medio de **Silhouette Score** y **Visualización en 2D mediante PCA** para poder determinar cual era el optimo.

## Visualizaciones Incluidas
Espera

3. Exploración general
     - Distribuciones
     - Correlaciones
     - Comparaciones por género

## Software usado
* **Python V.3** Lenguaje de programación
* **Pandas** Libreria para manipulación de datos
* **NumPy**
* **Scikit-learn** Desarrollo de ML
* **HDBSCAN** Analisis de datos
* **Seaborn** Matplotlib Visualización de datos
* **Jupyter** Notebook Desarrollo de proyecto
* **Plotly** Visualización interactiva

## Estructura del proyecto
1. app.py
2. requirements.txt
3. README.txt

## Dataset utilizado
[Dataset utilizado para el proyecto](https://www.kaggle.com/datasets/zinasakr/40k-songs-with-audio-features-and-lyrics) **Autor: Zina Sakr**""")

with  tab2:
    st.subheader("EDA(Exploratory Data Analysis)")
    st.markdown("""Dataset crudo para realizar insights""")
    st.dataframe(df)
    df.info()
    df.describe()










