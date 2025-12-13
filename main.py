import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

tab1, tab2, tab3, tab4 = st.tabs(["Documentación General","EDA", "Modelo de ML", "Conclusión"])

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
Se utilizaron diferentes algoritmos de clustering para ver que modelo de ML:

* K-Means
* DBSCAN

## Visualizaciones Incluidas
Espera

* Exploración general
* Analisis de la frecuencia con genero
* Comparaciones entre generos
* Distribuciones
* Correlaciones
* Clusterización

## Software usado
* **Python V.3** Lenguaje de programación
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **HDBSCAN**
* **Seaborn**
* **Jupyter**
* **Plotly**

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

with tab3:
    st.subheader("Modelos de Machine Learning (Clustering)")
    st.markdown("En esta sección aplicamos algoritmos no supervisados para agrupar canciones similares.")

    # 1. Selección de Variables
    st.markdown("### 1. Selección de Características")
    # Filtramos solo columnas numéricas útiles para clustering
    numeric_cols = ['artist_popularity', 'acousticness', 'danceability', 'energy', 
                    'instrumentalness', 'liveness', 'loudness', 'speechiness', 
                    'tempo', 'valence', 'duration_ms']
    
    features_selected = st.multiselect(
        "Selecciona las variables para el clustering:", 
        options=numeric_cols,
        default=['danceability', 'energy', 'valence', 'tempo'] # Selección por defecto
    )

    if len(features_selected) < 2:
        st.warning("Por favor selecciona al menos 2 variables para continuar.")
    else:
        # Preparación de datos (Subset y Escalado)
        X = df[features_selected]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. Configuración del Modelo
        st.markdown("### 2. Configuración del Modelo")
        algorithm = st.radio("Selecciona el algoritmo:", ["K-Means", "DBSCAN"], horizontal=True)

        # Variables para guardar resultados
        labels = None
        model = None

        col_params, col_plot = st.columns([1, 3])

        with col_params:
            if algorithm == "K-Means":
                k = st.slider("Número de Clusters (k)", min_value=2, max_value=10, value=4)
                if st.button("Ejecutar K-Means"):
                    model = KMeans(n_clusters=k, random_state=42)
                    model.fit(X_scaled)
                    labels = model.labels_
            
            elif algorithm == "DBSCAN":
                eps = st.slider("Epsilon (Distancia)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                min_samples = st.slider("Min Samples", min_value=2, max_value=20, value=5)
                if st.button("Ejecutar DBSCAN"):
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    model.fit(X_scaled)
                    labels = model.labels_

        # 3. Visualización y Resultados
        with col_plot:
            if labels is not None:
                # Añadimos los clusters al dataframe original para visualizar
                df_viz = df.copy()
                df_viz['Cluster'] = labels.astype(str) # Convertir a string para que plotly use colores discretos

                # Reducción de dimensionalidad con PCA para graficar en 2D
                # (Ya que tenemos muchas variables, las "aplastamos" a 2 dimensiones para verlas)
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_scaled)
                
                df_viz['PCA1'] = components[:, 0]
                df_viz['PCA2'] = components[:, 1]

                st.success(f"Modelo ejecutado. Clusters encontrados: {len(set(labels))}")

                # Gráfico de dispersión
                fig_cluster = px.scatter(
                    df_viz, 
                    x='PCA1', 
                    y='PCA2', 
                    color='Cluster',
                    hover_data=['artist_name', 'song_name'] + features_selected,
                    title=f"Resultados de {algorithm} (Visualización PCA)",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Explicación breve si sale DBSCAN con ruido
                if algorithm == "DBSCAN" and "-1" in df_viz['Cluster'].values:
                    st.info("Nota: En DBSCAN, el cluster '-1' representa 'Ruido' (puntos que no encajan en ningún grupo denso).")

            else:
                st.info("Configura los parámetros a la izquierda y presiona 'Ejecutar'.")


with tab4:
    st.subheader("Conclusiones")
    st.markdown("""
    Durante la realización de este proyecto se aplicó una metodología de Ciencia de Datos para analizar un dataset de canciones utilizando variables acústicas y de popularidad de los artistas. Se realizó un proceso de limpieza, normalización de datos y análisis exploratorio, seguido de la aplicación de técnicas de aprendizaje no supervisado para identificar patrones musicales.
    Mediante el uso de KMeans y DBSCAN, se identificaron clusters de canciones con perfiles acústicos diferenciados, representando distintos estilos musicales como canciones energéticas y bailables, canciones tranquilas y ruidosas, y canciones con características intermedias.
    Los clusters obtenidos pueden ser utilizados como una herramienta de segmentación musical, permitiendo aplicaciones prácticas como la creación automática de playlists, recomendaciones personalizadas y análisis de preferencias musicales sin depender de etiquetas predefinidas como los géneros.
    Una de las principales limitaciones del estudio es que los clusters se basan exclusivamente en características acústicas y no consideran el contexto cultural o subjetivo del género musical. Además, el número de clusters fue seleccionado mediante métodos heurísticos como Elbow y Silhouette.""")




