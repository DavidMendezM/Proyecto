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
from sklearn.datasets import make_classification
import traceback

try:
    st.set_page_config(page_title="Análisis de Fumadores 🚭", layout="wide")
    st.title("PCA y Selección de Variables - Data Smoking")
    st.subheader("Elaborado por: Daniela Forero Cardenas , David Mendez Medellin y María Alejandra Vanegas")
    st.subheader("Composición Data Set:")
    st.write("**Total registros:** 55,692")
    st.write("**Total variables:** 27")
    datos = {
        "Variable": [
            "ID", "género", "edad", "altura (cm)", "peso (kg)", "cintura (cm)", "vista (izquierda)", "vista (derecha)",
            "audición (izquierda)", "audición (derecha)", "presión arterial sistólica", "Relajación", "azúcar en sangre en ayunas",
            "Colesterol: total", "triglicéridos", "HDL", "LDL", "hemoglobina", "Proteína en la orina", "creatinina sérica",
            "AST", "ALT", "Gtp", "oral", "caries dental", "sarro", "Fumador"
        ],
        "Descripción": [
            "Índice", "Género", "Diferencia de 5 años", "Altura en centímetros", "Peso en kilogramos", "Longitud de la circunferencia de la cintura",
            "Visión del ojo izquierdo", "Visión del ojo derecho", "Audición del oído izquierdo", "Audición del oído derecho",
            "Presión arterial sistólica", "Presión arterial en relajación", "Glucosa en sangre en ayunas", "Colesterol total", "Triglicéridos",
            "Tipo de colesterol HDL", "Tipo de colesterol LDL", "Hemoglobina", "Proteína en orina", "Creatinina en suero",
            "(AST - aspartato aminotransferasa)", "(ALT - alanina aminotransferasa)", "γ-GTP (guanosín trifosfato)",
            "Estado del examen oral", "Presencia de caries", "Estado del sarro", "Estado de fumador"
        ]
    }

    datos = pd.DataFrame(datos)
    st.table(datos) 

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
    st.text("""
1. Triglicéridos vs. Tabaquismo:

La comparación muestra que los individuos fumadores presentan niveles ligeramente más elevados de triglicéridos en comparación con los no fumadores. 
La mediana de los triglicéridos es más alta en el grupo de fumadores, y se observa una mayor dispersión de los datos, con varios valores atípicos elevados. 
Esto sugiere una posible asociación entre el tabaquismo y un aumento en los niveles de triglicéridos, lo cual podría implicar un mayor riesgo cardiovascular.

2. Creatinina en Suero vs. Tabaquismo:

La variable creatinina en suero, se identifica también una leve elevación en los valores medianos para los fumadores, acompañada de una mayor cantidad de valores 
atípicos en ambos grupos. Sin embargo, el rango de variación de los niveles de creatinina es mayor en el grupo de fumadores. Esta observación puede indicar una 
posible relación entre el consumo de tabaco y una función renal alterada, aunque se recomienda complementar el análisis con pruebas estadísticas para confirmar 
esta hipótesis.

3. Presión Arterial Sistólica vs. Tabaquismo:

Al comparar la presión arterial sistólica entre fumadores y no fumadores, no se observan diferencias significativas en la mediana ni en la dispersión de los datos.
Ambos grupos presentan distribuciones similares y un número considerable de valores extremos elevados. Esto sugiere que, en esta muestra, el tabaquismo no muestra 
una influencia clara sobre la presión sistólica.
""")

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
    st.text("""
    1. Caries Dentales y Tabaquismo:
    
Aunque el número total de no fumadores es mayor en la muestra, se observa que la proporción de fumadores con caries es más alta en relación con su propio grupo. 
Esto sugiere que el tabaquismo podría estar asociado con una mayor incidencia de caries dentales.

2. Sarro Dental y Tabaquismo:

Se observa que los fumadores tienden a presentar mayor frecuencia relativa de sarro dental en comparación con los no fumadores. 
La diferencia es más marcada en el grupo con sarro, lo que podría indicar una relación entre el tabaquismo y la acumulación de placa o falta de higiene bucal
adecuada.
""")
    st.markdown("""

##se eliminan lo campos ID y Examén Oral, dado que el ID no representa una variable de estudio y Examén Oral todas las observaciones tenían un único valor.

##Se transforma a dummy las variables categóricas (Género y Sarro)

""")
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

    st.subheader("Balanceo de la data")
    smoking_distribution = df['smoking'].value_counts()
    st.write("Distribución de la variable 'smoking':", smoking_distribution)

    X = df.drop('smoking', axis=1)
    y = df['smoking']
    st.write("Forma de X:", X.shape)
    st.write("Forma de y:", y.shape)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    st.write("Distribución de la variable 'smoking' después de SMOTE:", y_resampled.value_counts())

    st.subheader("Preparación de los datos")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    st.write('Entrenamiento:', X_train.shape, y_train.shape)
    st.write('Prueba:', X_test.shape, y_test.shape)
    st.markdown("""

Se realiza validación cruzada para calcular el K óptimo para la Prueba F de Anova. 
**Best Config: {'anova__k': 18}**
""")

    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)

    st.subheader("Prueba F de ANOVA")

    def select_features(X_train, y_train, X_test, score_func, k):
        fs = SelectKBest(score_func=score_func, k=k)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    # Selección de variables con ANOVA, k=18
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_classif, 18)

    # Máscara de las variables seleccionadas
    selected_mask = fs.get_support()  # Boolean array

    # Nombres de las variables seleccionadas
    selected_features = X_train.columns[selected_mask]

    # Scores solo de las seleccionadas
    selected_scores = fs.scores_[selected_mask]

    # DataFrame para graficar
    scores_df = pd.DataFrame({'feature': selected_features, 'score': selected_scores}).sort_values('score', ascending=False)
    st.bar_chart(scores_df.set_index('feature'))
    st.markdown("""

Se realiza validación cruzada para calcular el K óptimo para la Selección de variables por información mutua. 
**Best Config: {'anova__k': 18}**
""")
    
    st.subheader("Selección de variables por información mutua")
    X_train_fs, X_test_fs, fs_mut = select_features(X_train, y_train, X_test, mutual_info_classif, 'all')
    selected_mask_mut = fs_mut.get_support()
    selected_features_mut = X_train.columns[selected_mask_mut]
    selected_scores_mut = fs_mut.scores_[selected_mask_mut]
    scores_df_mut = pd.DataFrame({
    'feature': selected_features_mut,
    'score': selected_scores_mut
    }).sort_values('score', ascending=False)
    st.bar_chart(scores_df_mut.set_index('feature'))
    
    st.subheader("Modelado con características seleccionadas")
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    st.write('accuracy(todas las variables): %.2f' % (accuracy*100))

    st.subheader("Modelo usando características ANOVA")
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_classif, 18)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_fs, y_train)
    yhat = model.predict(X_test_fs)
    accuracy = accuracy_score(y_test, yhat)
    st.write('accuracy(ANOVA): %.2f' % (accuracy*100))

    st.subheader("Modelo usando características de información mutua")
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_classif, 'all')
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_fs, y_train)
    yhat = model.predict(X_test_fs)
    accuracy = accuracy_score(y_test, yhat)
    st.write('accuracy(Información mutua): %.2f' % (accuracy*100))
    
    st.subheader("Ajuste del número de variables seleccionadas")
    num_features = [i+1 for i in range(X.shape[1])]
    results_list = []
    
    for k in num_features:
        model = LogisticRegression(solver='liblinear')
        fs = SelectKBest(score_func=f_classif, k=18)  # aquí debe ser k, no 18
        pipeline = Pipeline(steps=[('anova', fs), ('lr', model)])
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
    st.markdown("""
    El mejor modelo para la selección de variables es: 
    ** ** , ya que presenta un valor mayor de exactitud. 
""")

    
    def get_models():
        models = dict()
        for i in range(1, 16):  # Hasta 15 componentes, puedes subirlo si tu máquina lo aguanta
            steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression(solver='liblinear', max_iter=500))]
            models[str(i)] = Pipeline(steps=steps)
        return models

    def evaluate_model(model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)  # Optimizado para todo el dataset
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores

    # Usa todo el dataset real (balanceado)
    X_pca = X
    y_pca = y

    models = get_models()

    st.subheader("Evaluando modelos con diferentes componentes PCA")
    results, names = list(), list()
    progress = st.progress(0)
    for idx, (name, model) in enumerate(models.items()):
        scores = evaluate_model(model, X_pca, y_pca)
        results.append(scores)
        names.append(name)
        st.write(f'Componentes: {name} | Accuracy media: {mean(scores):.3f} (std: {std(scores):.3f})')
        progress.progress((idx+1)/len(models))

    st.subheader("Comparación de desempeño (Boxplot)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(results, tick_labels=names, showmeans=True)
    plt.xticks(rotation=45)
    plt.xlabel("N° de componentes PCA")
    plt.ylabel("Accuracy")
    st.pyplot(fig)

    st.subheader("Entrenamiento final con 15 componentes PCA")
    steps = [('pca', PCA(n_components=15)), ('m', LogisticRegression(solver='liblinear', max_iter=500))]
    model = Pipeline(steps=steps)
    model.fit(X_pca, y_pca)
    pca_step = model.named_steps['pca']
    X_pca_transformed = pca_step.transform(X_pca)
    pca_columns = [f'Principal_Component_{i+1}' for i in range(X_pca_transformed.shape[1])]
    X_pca_transformed_df = pd.DataFrame(X_pca_transformed, columns=pca_columns)

    st.write("Shape del dataset transformado:", X_pca_transformed_df.shape)
    st.write(X_pca_transformed_df.head())
    st.markdown("""
    En el método de análisis de componente principales (PCA), se observa que con 15 componentes se logra explicar **el 72% de exactitud**. 
""")

except Exception as e:
    st.error("Ocurrió un error al ejecutar la aplicación:")
    st.error(str(e))
    st.text(traceback.format_exc())
