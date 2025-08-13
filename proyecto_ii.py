import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from numpy import mean, std
import traceback

try:
    st.set_page_config(page_title="Análisis de Fumadores", layout="wide")
    st.title("PCA y Selección de Variables - Data Smoking")
    
    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/malejavanegas0/Proyecto/b9aa27643b2cd3ddb5963d5ff49dc83df09b53da/smoking.csv"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df

    df = load_data()

    st.subheader("Información inicial del DataFrame")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.write("Descripción estadística:", df.describe())
    st.write("Valores nulos por columna:", df.isnull().sum())
    df = df.drop(["ID", "oral"], axis=1) # Eliminamos columnas ID y oral

    # Boxplots
    variables = ['triglyceride', 'serum creatinine', 'systolic']
    st.subheader("Boxplots por variable y fumadores")
    fig, axs = plt.subplots(1, len(variables), figsize=(18, 6))
    for i, variable in enumerate(variables):
        sns.boxplot(x='smoking', y=variable, data=df, ax=axs[i])
        axs[i].set_title(f'{variable} vs. smoking')
    st.pyplot(fig)

    # Gráficos de conteo
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x='dental caries', hue='smoking', data=df, ax=axes[0])
    axes[0].set_title('Distribución de Fumadores por Caries Dentales')
    axes[0].set_xlabel('Caries Dentales')
    axes[0].set_ylabel('Conteo')
    sns.countplot(x='tartar', hue='smoking', data=df, ax=axes[1])
    axes[1].set_title('Distribución de Fumadores por Sarro')
    axes[1].set_xlabel('Sarro')
    axes[1].set_ylabel('Conteo')
    st.pyplot(fig)

    st.subheader("Conversión de variables categóricas a dummies")
    cat_features = ["gender", "tartar"]
    df = pd.get_dummies(df, columns=cat_features)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    st.write(df.head())

    st.write(f"Forma del DataFrame: {df.shape}")
    st.write(f"Número de filas: {df.shape[0]}")
    st.write(f"Número de columnas: {df.shape[1]}")

    scaler = StandardScaler()

    st.subheader("2. Balanceo de la data")
    smoking_distribution = df['smoking'].value_counts()
    st.write("Distribución de la variable 'smoking':", smoking_distribution)

    X = df.drop('smoking', axis=1)
    y = df['smoking']
    st.write("Forma de X:", X.shape)
    st.write("Forma de y:", y.shape)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    st.write("Distribución de la variable 'smoking' después de SMOTE:", y_resampled.value_counts())

    st.subheader("3. Preparación de los datos")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    st.write('Entrenamiento:', X_train.shape, y_train.shape)
    st.write('Prueba:', X_test.shape, y_test.shape)

    #st.subheader("5. Selección de variables numéricas (ANOVA)")
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    #model = LogisticRegression(solver='liblinear')
    #fs = SelectKBest(score_func=f_classif)
    #pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
    #grid = dict()
    #grid['anova__k'] = [i+1 for i in range(X_train.shape[1])]
    #search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)
    #results = search.fit(X_train, y_train)
    #st.write('Mejor precisión media:', results.best_score_)
    #st.write('Mejor configuración:', results.best_params_)

    st.subheader("Prueba F de ANOVA")
    def select_features(X_train, y_train, X_test, score_func, k):
        fs = SelectKBest(score_func=score_func, k=k)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_classif, 'all')
    scores = fs.scores_
    st.bar_chart(scores)

    st.subheader("Selección de variables por información mutua")
    X_train_fs, X_test_fs, fs_mut = select_features(X_train, y_train, X_test, mutual_info_classif, 'all')
    scores_mut = fs_mut.scores_
    st.bar_chart(scores_mut)

    st.subheader("Modelado con características seleccionadas")
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    st.write('Precisión (todas las variables): %.2f' % (accuracy*100))

    st.subheader("Modelo usando características ANOVA")
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_classif, 6)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_fs, y_train)
    yhat = model.predict(X_test_fs)
    accuracy = accuracy_score(y_test, yhat)
    st.write('Precisión (ANOVA): %.2f' % (accuracy*100))

    st.subheader("Modelo usando características de información mutua")
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_classif, 4)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_fs, y_train)
    yhat = model.predict(X_test_fs)
    accuracy = accuracy_score(y_test, yhat)
    st.write('Precisión (Información mutua): %.2f' % (accuracy*100))

    st.subheader("Ajuste del número de variables seleccionadas")
    num_features = [i+1 for i in range(X.shape[1])]
    results_list = []
    for k in num_features:
        model = LogisticRegression(solver='liblinear')
        fs = SelectKBest(score_func=f_classif, k=k)
        pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        results_list.append(scores)
        st.write(f'> {k} variables: Media={mean(scores):.3f}, Std={std(scores):.3f}')
    fig, ax = plt.subplots()
    ax.boxplot(results_list, showmeans=True)
    ax.set_xticklabels(num_features, rotation=45)
    st.pyplot(fig)

    selector = SelectKBest(score_func=f_classif, k=18)
    X_selected = selector.fit_transform(X, y)
    selected_features_mask = selector.get_support()
    selected_feature_names = X.columns[selected_features_mask]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)
    st.write("Data set con las mejores características", X_selected_df.shape)
    st.write(X_selected_df.head())

    st.subheader("PCA")
    X_pca = X_selected_df
    pca = PCA(n_components=5)
    X_pca_transformed = pca.fit_transform(X_pca)
    pca_columns = [f'Componente Principal {i+1}' for i in range(X_pca_transformed.shape[1])]
    X_pca_transformed_df = pd.DataFrame(X_pca_transformed, columns=pca_columns)
    st.write("Data transformada por PCA:", X_pca_transformed_df.shape)
    st.write(X_pca_transformed_df.head())
    fig = px.scatter_matrix(X_pca_transformed_df)
    st.plotly_chart(fig)

except Exception as e:
    st.error("Ocurrió un error al ejecutar la aplicación:")
    st.error(str(e))
    st.text(traceback.format_exc())
