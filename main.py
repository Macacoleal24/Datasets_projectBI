import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Spotify Dashboard Analysis",
    page_icon="📊",
    layout="wide"
)
st.image("https://posgrados-panamericana.up.edu.mx/hs-fs/hubfs/logo%20posgrados%20con%20espacio.png?width=343&name=logo%20posgrados%20con%20espacio.png",width=100)
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
    st.markdown("""# Tablero Interactivo de Análisis de Música

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

with tab2:
    st.subheader("EDA (Exploratory Data Analysis)")
    st.markdown("Dataset crudo para realizar insights")
    st.dataframe(df)
    st.metric("Número de filas", df.shape[0])
    st.metric("Número de columnas", df.shape[1])
    df = df.rename(columns={"new_artist_popularity": "artist_popularity"})
    df["artist_popularity"] = df["artist_popularity"] * 100
    df["genre"] = df["genres"].str.split(";").str[0].str.strip()
    df.drop(columns="genres", inplace=True)
    df["duration_ms"] = df["duration_ms"] / 60000
    
    columnas = st.multiselect("Selecciona columnas para el pairplot", df.columns)
    if len(columnas) > 1:
        fig = sns.pairplot(df[columnas], corner=True)
        st.pyplot(fig.figure)
    else:
        st.info("Selecciona al menos dos columnas numericos.")

    st.markdown("""
    ## Columnas usadas para el analisis de los datos
    * Variables numéricas relevantes: 
        - artist_popularity
        - acousticness
        - danceability
        - energy
        - instrumentalness
        - liveness
        - loudness
        - speechiness
        - tempo
        - valence
        - duration_ms
        
    * Variables que se eliminaron
        - key
        - mode
        - time_signature
        - lyrics

    ###No se encontraron valores nulos en las variables numéricas, por lo que no se requirió imputación.""")

    n = st.slider("Selecciona la cantidad de artistas", 5, 50, 15)

    top_artists = df["artist_name"].value_counts().head(n).reset_index()
    top_artists.columns = ["artist_name", "song_count"]
    
    fig = px.bar(
        top_artists,
        x="song_count",
        y="artist_name",
        orientation="h",
        color="song_count",
        color_continuous_scale="Viridis",
        title=f"Top {n} Artists With Most Songs"
    )
    
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ---------------
    fig = px.histogram(
    df,
    x="duration_ms",
    nbins=100,
    title="Distribución de Duración de Canciones (ms)",
    color_discrete_sequence=["#4c78a8"]
    )
    
    fig.update_layout(
        xaxis_title="Duration (ms)",
        yaxis_title="Count",
        bargap=0.1,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    # ---------------}
    df_clean = df.dropna(subset=["energy", "artist_popularity"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_clean,
        x="energy",
        y="artist_popularity",
        color="red",
        ax=ax
    )
    
    ax.set_title("Artist Popularity vs Energy")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Artist Popularity")
    
    st.pyplot(fig)
    # ---------------}
    top_genres = df["genre"].value_counts().head(15)

    fig = px.bar(
        x=top_genres.index,
        y=top_genres.values,
        labels={"x": "Genre", "y": "Frequency"},
        title="Most Frequent Genres",
        color=top_genres.values,
        color_continuous_scale="Plasma"
    )
    
    fig.update_layout(
        xaxis_tickangle=45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    #-----------------
    top_genres = df["genre"].value_counts().head(10)
    df_top = df[df["genre"].isin(top_genres.index)]
    
    fig = px.box(
        df_top,
        x="genre",
        y="danceability",
        color="genre",
        title="Danceability Distribution by Genre",
    )
    
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Danceability",
        showlegend=False,
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    #---------------------
    artist_summary = df.groupby("artist_name").agg(
        songs_count=("song_name", "count"),
        avg_popularity=("artist_popularity", "mean")
    ).sort_values("songs_count", ascending=False).head(15)
    
    fig = px.scatter(
        artist_summary.reset_index(),
        x="songs_count",
        y="avg_popularity",
        color="artist_name",
        size="avg_popularity",
        hover_name="artist_name",
        title="Song Count vs Avg Artist Popularity",
    )
    
    fig.update_layout(
        xaxis_title="Song Count",
        yaxis_title="Avg Artist Popularity",
        legend_title="Artist",
    )
    
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="black")))
    
    st.plotly_chart(fig, use_container_width=True)
    #------------------
    features = df.select_dtypes(include=["int64", "float64"])
    cor = features.corr(method="pearson")
    
    fig, axis = plt.subplots(figsize=(10, 10))
    
    sns.heatmap(
        cor,
        annot=True,
        ax=axis,
        mask=np.triu(cor),
        fmt=".2f",
        cmap="coolwarm_r",
        vmin=-1,
        vmax=1
    )
    
    st.pyplot(fig)
                









