import os
import zipfile
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas import read_csv
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import (SelectKBest, chi2, f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score)
from sklearn.model_selection import (train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from numpy import mean, std

st.set_page_config(page_title="Smoking Análisis", layout="wide")
st.title("PCA y Selección de Variables Data Smoking")

@st.cache_data
def load_data():
    url = "https://github.com/malejavanegas0/Proyecto/blob/b9aa27643b2cd3ddb5963d5ff49dc83df09b53da/smoking.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()
df.info()
df.describe()
df.isnull().sum()
df = df.drop(["ID","oral"], axis=1) # eliminamos ID y oral


variables = ['triglyceride', 'serum creatinine', 'systolic']
plt.figure(figsize=(15, 10))
for i, variable in enumerate(variables, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='smoking', y=variable, data=df)
    plt.title(f'{variable} smoking')
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x='dental caries', hue='smoking', data=df, ax=axes[0])
axes[0].set_title('Distribución de Fumar por Caries Dentales')
axes[0].set_xlabel('Caries Dentales')
axes[0].set_ylabel('Conteo')

sns.countplot(x='tartar', hue='smoking', data=df, ax=axes[1])
axes[1].set_title('Distribución de Fumar por Sarro')
axes[1].set_xlabel('Sarro')
axes[1].set_ylabel('Conteo')

plt.tight_layout()
plt.show()

cat_features = ["gender","tartar"]
df = pd.get_dummies(df, columns=cat_features)

# Convertir a dummy (genero, sarro)
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

df.head()

print("Shape:", df.shape)
print("Número de filas:", df.shape[0])
print("Número de columnas:", df.shape[1])

scaler = StandardScaler()

"""2.BALANCEO DE LA DATA"""

smoking_distribution = df['smoking'].value_counts()
print("Distribución de la variable 'smoking':")
print(smoking_distribution)

X = df.drop('smoking', axis=1)
y = df['smoking']
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)



smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Distribución de la variable 'smoking' después de SMOTE:")
print(y_resampled.value_counts())

"""3.PREPARACIÓN DE LOS DATOS"""



X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

"""5.SELECCIÓN DE VARIABLES NUMÉRICAS"""

X_train.head()

# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)



# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define the pipeline to evaluate
model = LogisticRegression(solver='liblinear') # You can add max_iter here if needed
fs = SelectKBest(score_func=f_classif) # Choose your desired score_func

pipeline = Pipeline(steps=[('anova',fs), ('lr', model)]) # Name the steps appropriately

# define the grid of hyperparameters to search
grid = dict()
grid['anova__k'] = [i+1 for i in range(X_train.shape[1])] # Search over possible values of k

# define the grid search
search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)

# perform the search
results = search.fit(X_train, y_train) # Fit on the training data

# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)

"""prueba F de ANOVA"""


# define feature selection function
def select_features(X_train, y_train, X_test, score_func, k):
    # configure to select all features
    fs = SelectKBest(score_func=score_func, k=18)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_classif, 'all')

# what are scores for the features
for i in range(len(fs.scores_)):
  print('Feature %d: %f' % (i, fs.scores_[i]))

# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

"""Selección de funciones de información mutua"""



# define feature selection function (reusing the previous one)
def select_features(X_train, y_train, X_test, score_func, k):
    # configure to select all features
    fs = SelectKBest(score_func=score_func, k=18)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_classif, 'all')

# what are scores for the features
for i in range(len(fs.scores_)):
  print('Feature %d: %f' % (i, fs.scores_[i]))

# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

"""Modelado con características seleccionadas"""

# fit the model
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# evaluate the model
yhat = model.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))

"""Modelo construido utilizando características de la prueba F de ANOVA"""

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_classif, 6)

# fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_fs, y_train)

# evaluate the model
yhat = model.predict(X_test_fs)

# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))

"""Modelo construido utilizando características de información mutua"""

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_classif, 4)

# fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_fs, y_train)

# evaluate the model
yhat = model.predict(X_test_fs)

# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))

"""Ajuste el número de funciones seleccionadas"""



# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define the pipeline to evaluate
model = LogisticRegression(solver='liblinear')

fs = SelectKBest(score_func=f_classif)

pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])

# define the grid
grid = dict()
grid['anova__k'] = [i+1 for i in range(X.shape[1])]

# define the grid search
search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)

# perform the search
results = search.fit(X, y)

# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)



# evaluate a given model using cross-validation
def evaluate_model(model):
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
  scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
  return scores

# define number of features to evaluate
num_features = [i+1 for i in range(X.shape[1])]

# enumerate each number of features
results = list()

for k in num_features:
  # create pipeline
  model = LogisticRegression(solver='liblinear')
  fs = SelectKBest(score_func=f_classif, k=k) # 18
  pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
  # evaluate the model
  scores = evaluate_model(pipeline)
  results.append(scores)
  # summarize the results
  print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))

# plot model performance for comparison
pyplot.boxplot(results, tick_labels=num_features, showmeans=True)
pyplot.show()



# Select the top 18 features based on f_classif scores
selector = SelectKBest(score_func=f_classif, k=18)
X_selected = selector.fit_transform(X, y)

# Get the names of the selected features
selected_features_mask = selector.get_support()
selected_feature_names = X.columns[selected_features_mask]

# Create a new DataFrame with the selected features
X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)

print("Data set", X_selected_df.shape)
display(X_selected_df.head())

"""PCA"""



# get the dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=25, n_informative=20, n_redundant=5, random_state=7)
    return X, y

# get a list of models to evaluate
def get_models():
    models = dict()
    for i in range(1,21):
        steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())]
        models[str(i)] = Pipeline(steps=steps)
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

# define dataset
X, y = get_dataset()

# get the models to evaluate
models = get_models()

# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# plot model performance for comparison
pyplot.boxplot(results, tick_labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.show()

# define dataset
# X, y = make_classification(n_samples=1000, n_features=25, n_informative=20, n_redundant=5, random_state=7) # Commented out synthetic data generation

# define the model
steps = [('pca', PCA(n_components=15)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)

# fit the model on the whole dataset
model.fit(X, y)

# Extract the PCA step from the pipeline
pca_step = model.named_steps['pca']

# Transform the original data using the fitted PCA
X_pca_transformed = pca_step.transform(X)

# Create a DataFrame from the transformed data for better readability
pca_columns = [f'Principal_Component_{i+1}' for i in range(X_pca_transformed.shape[1])]
X_pca_transformed_df = pd.DataFrame(X_pca_transformed, columns=pca_columns)

print("Shape of the transformed dataset:", X_pca_transformed_df.shape)
display(X_pca_transformed_df.head())
