# SMOKING‑Streamlit

**Análisis interactivo de datos SMOKING con Streamlit**  
Aplicación web para explorar prevalencia de condiciones crónicas utilizando **PCA**, con técnicas avanzadas de selección de variables **(Información Mutua y Prueba F ANOVA)**.

---

##  Contenido del repositorio

- `app_pca_acm_nhanes.py`  
  Invluye todo lo relacionado al tratamiento del Dataset, distintos métodos de selección de variables y  análisis de componentes principales (PCA). Adicional se tiene un filtro un interactivo por la variable género. 

- `requirements.txt`  
  Paquetes necesarios para ejecutar la app (Streamlit, scikit-learn, imbalanced-learn, prince, seaborn, matplotlib, numpy, pandas, etc.).

- `README.md`  
  Documentación completa del proyecto.

---

##  Descripción de los análisis

### 1. **Análisis Descriptivo del Dataset**
- Análisis exploratorio de datos; se eliminan lo campos ID y Examén Oral, dado que el ID no representa una variable de estudio y Examén Oral todas las observaciones tenían un único valor.
- Gráficas de comportamiento de las variables que se pueden ver afectadas por la condición de ser fumador.
- Se transforma a dummy las variables categóricas (Género y Sarro)
  
### 2. **PreProcesamiento del Dataset**
- Se balancea el Dataset a través de la función SMOTE
- Se genera la Data de entrenamiento y prueba
  
### 3. **Selección de variables**
Implementa dos metodologías:
- **Filtro (SelectKBest)**: Selecciona variables numéricas con mayor relación estadística (ANOVA) con la clase objetivo.
- **Funciones de información mutua (mutual_info_classif)**: La información mutua es sencilla al considerar la distribución de dos variables discretas

### 3. **Modelado y ajustes de las caracetrísticas seleccionadas**
- Para cada método de selección se evalúa la capacidad predictiva mediante `cross_val_score` sobre un clasificador (Regresión Logística). Así se mide la consistencia y evita overfitting.
- Dataset resultante de la reducción de variables.

### 4. **PCA (Análisis de Componentes Principales)**
- Reducción de dimensionalidad tras aplicar SMOTE y selección RFE.
- Visualización gráfica del espacio PCA
- Dataset resultante del PCA


