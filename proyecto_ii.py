import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

st.title("EDA + PCA + Modelos con Smoking Dataset")

# Subir archivo
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Información inicial")
        st.write(df.head())
        st.write("Shape:", df.shape)
        st.write("Columnas:", df.columns.tolist())
        st.write("Tipos de datos:")
        st.write(df.dtypes)
        st.write("Valores nulos por columna:")
        st.write(df.isnull().sum())
        st.write("Descripción estadística:")
        st.write(df.describe())

        # Transformación dummies
        df = pd.get_dummies(df, drop_first=True)

        # Balanceo SMOTE
        X = df.drop('smoking', axis=1)
        y = df['smoking']
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        st.subheader("📌 Después de SMOTE")
        st.write("Shape X_resampled:", X_resampled.shape)
        st.write("Distribución balanceada de la variable objetivo:")
        st.bar_chart(pd.Series(y_resampled).value_counts())

        # Selección ANOVA
        selector_anova = SelectKBest(score_func=f_classif, k=18)
        X_selected_anova = selector_anova.fit_transform(X_resampled, y_resampled)
        selected_features_anova = X_resampled.columns[selector_anova.get_support()]
        X_selected_df = pd.DataFrame(X_selected_anova, columns=selected_features_anova)

        st.subheader("🔍 Variables seleccionadas con ANOVA")
        st.write(selected_features_anova)

        # Comparación ANOVA vs MI
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=18)
        selector_mi.fit_transform(X_resampled, y_resampled)
        selected_features_mi = X_resampled.columns[selector_mi.get_support()]
        st.write("Comparación ANOVA vs MI")
        st.write("ANOVA:", set(selected_features_anova))
        st.write("MI:", set(selected_features_mi))
        st.write("Coincidencias:", set(selected_features_anova).intersection(set(selected_features_mi)))

        # === PCA EXPLORATORIO ===
        st.subheader("🔎 Exploración PCA con características seleccionadas (ANOVA)")

        pca = PCA()
        X_pca = pca.fit_transform(X_selected_df)

        pca_df = pd.DataFrame(
            data=X_pca,
            columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
        )
        pca_df['smoking'] = y_resampled.values

        # Scatter PC1 vs PC2
        st.write("### Dispersión PC1 vs PC2")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='smoking', data=pca_df, alpha=0.5, ax=ax)
        ax.set_title('PC1 vs PC2')
        st.pyplot(fig)

        # Heatmap de loadings
        st.write("### Heatmap de cargas de los componentes")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pca.components_,
            cmap='viridis',
            annot=True,
            fmt=".2f",
            xticklabels=X_selected_df.columns,
            yticklabels=[f'PC{i+1}' for i in range(pca.n_components_)],
            ax=ax
        )
        ax.set_title("PCA Loadings Heatmap")
        st.pyplot(fig)

        # Varianza explicada
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = explained_variance_ratio.cumsum()

        st.write("### Varianza explicada acumulada")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(cumulative_explained_variance) + 1),
                cumulative_explained_variance,
                marker='o', linestyle='--')
        ax.set_title("Cumulative Explained Variance by Number of Components")
        ax.set_xlabel("Número de Componentes")
        ax.set_ylabel("Varianza Explicada Acumulada")
        ax.grid(True)
        st.pyplot(fig)

        # === Modelado PCA con LogisticRegression ===
        st.subheader("⚙️ Modelado PCA con Logistic Regression")

        scores = []
        for n_components in range(1, min(X_selected_df.shape[1], 21)):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components)),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ])
            cv_scores = cross_val_score(pipeline, X_selected_df, y_resampled, cv=5, scoring='accuracy')
            scores.append((n_components, cv_scores.mean()))

        st.write("Scores (n_components, accuracy):", scores)

        # Gráfico accuracy vs componentes
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([x[0] for x in scores], [x[1] for x in scores], marker='o', linestyle='-')
        ax.set_xlabel("Número de componentes PCA")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Número de componentes PCA")
        ax.grid(True)
        st.pyplot(fig)

        # Entrenamiento final con 15 componentes
        st.subheader("✅ Modelo final con 15 componentes PCA")
        pipeline_final = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=15)),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
        pipeline_final.fit(X_selected_df, y_resampled)

        y_pred = pipeline_final.predict(X_selected_df)
        st.write("Classification Report:")
        st.text(classification_report(y_resampled, y_pred))
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(y_resampled, y_pred)).plot(ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

