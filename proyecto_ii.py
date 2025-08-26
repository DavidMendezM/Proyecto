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
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from numpy import mean, std

# --- Configuración inicial ---
st.set_page_config(page_title="Análisis de Fumadores 🚭", layout="wide")
st.title("PCA y Selección de Variables - Data Smoking")
st.subheader("Elaborado por: Daniela Forero Cardenas , David Mendez Medellin y María Alejandra Vanegas ")
st.subheader("Composición Data Set:")
st.write("**Total registros:** 55,692")
st.write("**Total variables:** 27")

# --- Diccionario de variables ---
datos = {
    "Variable": [
        "ID", "gender", "age", "height(cm)", "weight(kg)", "waist(cm)",
        "eyesight(left)", "eyesight(right)", "hearing(left)", "hearing(right)",
        "systolic", "relaxation", "fasting blood sugar", "Cholestero", "triglyceride",
        "HDL", "LDL", "hemoglobin", "Urine protein", "serum creatinine",
        "AST", "ALT", "Gtp", "oral dental", "caries", "tartar", "smoking"
    ],
    "Nombre": [
        "ID", "género", "edad", "altura (cm)", "peso (kg)", "cintura (cm)",
        "vista (izquierda)", "vista (derecha)", "audición (izquierda)", "audición (derecha)",
        "presión arterial sistólica", "relajación", "azúcar en sangre en ayunas",
        "colesterol total", "triglicéridos", "HDL", "LDL", "hemoglobina",
        "proteína en la orina", "creatinina sérica", "AST", "ALT", "Gtp",
        "oral", "caries dental", "sarro", "fumador"
    ],
    "Descripción": [
        "Índice", "Género", "Diferencia de 5 años", "Altura en centímetros",
        "Peso en kilogramos", "Longitud de la circunferencia de la cintura",
        "Visión del ojo izquierdo", "Visión del ojo derecho",
        "Audición del oído izquierdo", "Audición del oído derecho",
        "Presión arterial sistólica", "Presión arterial en relajación",
        "Glucosa en sangre en ayunas", "Colesterol total", "Triglicéridos",
        "Tipo de colesterol HDL", "Tipo de colesterol LDL", "Hemoglobina",
        "Proteína en orina", "Creatinina en suero",
        "(AST - aspartato aminotransferasa)", "(ALT - alanina aminotransferasa)",
        "γ-GTP (guanosín trifosfato)", "Estado del examen oral",
        "Presencia de caries", "Estado del sarro", "Estado de fumador"
    ]
}

st.table(pd.DataFrame(datos))

# --- Cargar datos ---
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
st.text(buffer.getvalue())
st.write("Descripción estadística:", df.describe())
st.write("Valores nulos por columna:", df.isnull().sum())

# --- Limpieza de columnas ---
df = df.drop(["ID", "oral"], axis=1)

# --- Boxplots ---
numerical_variables = df.select_dtypes(include=np.number).columns.tolist()
if 'smoking' in numerical_variables:
    numerical_variables.remove('smoking')

st.title("Boxplots por variable numérica vs Smoking")

n_cols = 4
for i in range(0, len(numerical_variables), n_cols):
    cols = st.columns(n_cols)
    for j, variable in enumerate(numerical_variables[i:i+n_cols]):
        with cols[j]:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(x='smoking', y=variable, data=df, ax=ax)
            ax.set_title(f'Distribución de {variable} por Smoking')
            st.pyplot(fig)

# --- Countplots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x='caries', hue='smoking', data=df, ax=axes[0])
axes[0].set_title('Distribución de Fumadores por Caries Dentales')
sns.countplot(x='tartar', hue='smoking', data=df, ax=axes[1])
axes[1].set_title('Distribución de Fumadores por Sarro')
st.pyplot(fig)

# --- Dummies seguras ---
st.subheader("Conversión de variables categóricas a dummies")
cat_features = [col for col in ["gender", "tartar"] if col in df.columns]
df = pd.get_dummies(df, columns=cat_features, drop_first=False)
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
st.write(df.head())
st.write(f"Forma del DataFrame: {df.shape}")

# --- Balanceo ---
scaler = StandardScaler()
st.subheader("Balanceo de la data")
smoking_distribution = df['smoking'].value_counts()
st.write("Distribución de la variable 'smoking':", smoking_distribution)

X = df.drop('smoking', axis=1)
y = df['smoking']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
st.write("Distribución después de SMOTE:", y_resampled.value_counts())

# --- Train/Test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# --- Selección de variables (ANOVA + MI) ---
def select_features(X_train, y_train, X_test, score_func, k):
    fs = SelectKBest(score_func=score_func, k=k)
    fs.fit(X_train, y_train)
    return fs.transform(X_train), fs.transform(X_test), fs

cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)

# ANOVA
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_classif, 18)
selected_features = X_train.columns[fs.get_support()]
scores_df = pd.DataFrame({'feature': selected_features, 'score': fs.scores_[fs.get_support()]}).sort_values('score', ascending=False)
st.bar_chart(scores_df.set_index('feature'))

# Info mutua
X_train_fs, X_test_fs, fs_mut = select_features(X_train, y_train, X_test, mutual_info_classif, 24)
selected_features_mut = X_train.columns[fs_mut.get_support()]
scores_df_mut = pd.DataFrame({'feature': selected_features_mut, 'score': fs_mut.scores_[fs_mut.get_support()]}).sort_values('score', ascending=False)
st.bar_chart(scores_df_mut.set_index('feature'))

# --- Modelado ---
st.subheader("Modelado con Logistic Regression")
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)
st.write("Accuracy (todas las variables):", accuracy_score(y_test, model.predict(X_test)))

# ANOVA
X_train_fs, X_test_fs, _ = select_features(X_train, y_train, X_test, f_classif, 18)
model.fit(X_train_fs, y_train)
st.write("Accuracy (ANOVA):", accuracy_score(y_test, model.predict(X_test_fs)))

# Info mutua
X_train_fs, X_test_fs, _ = select_features(X_train, y_train, X_test, mutual_info_classif, 24)
model.fit(X_train_fs, y_train)
st.write("Accuracy (Info mutua):", accuracy_score(y_test, model.predict(X_test_fs)))

# --- PCA exploratorio ---
st.subheader("Exploración PCA con características seleccionadas")
selector = SelectKBest(score_func=f_classif, k=18)
X_selected = selector.fit_transform(X, y)
selected_features_mask = selector.get_support()
X_selected_df = pd.DataFrame(X_selected, columns=X.columns[selected_features_mask])

pca = PCA()
X_pca = pca.fit_transform(X_selected_df)
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['smoking'] = y_resampled.values

# PC1 vs PC2
fig, ax = plt.subplots()
sns.scatterplot(x='PC1', y='PC2', hue='smoking', data=pca_df, alpha=0.5, ax=ax)
st.pyplot(fig)

# Heatmap loadings
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pca.components_, cmap='viridis', annot=True, fmt=".2f",
            xticklabels=X_selected_df.columns,
            yticklabels=[f'PC{i+1}' for i in range(pca.n_components_)],
            ax=ax)
st.pyplot(fig)

# Varianza explicada
fig, ax = plt.subplots()
ax.plot(range(1, len(pca.explained_variance_ratio_.cumsum())+1),
        pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
ax.set_title("Varianza explicada acumulada")
st.pyplot(fig)
