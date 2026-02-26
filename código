# %%
# ============================================================
# PROYECTO MÓDULO 7 – Técnicas avanzadas para ciencia de datos
# Dataset: Your Career Aspirations of GenZ
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# Si el mejor modelo ya está en la sesión (por ejecución previa), guardarlo rápidamente y terminar la celda
try:
    if 'best_rf' in globals():
        import joblib, os
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_rf, 'models/modelo_educacion_superior.pkl')
        print('Modelo existente guardado en: models/modelo_educacion_superior.pkl')
        raise SystemExit('Guardado completado, deteniendo ejecución adicional de la celda.')
except SystemExit:
    pass

# -----------------------------
# CARGA INICIAL DEL DATASET
# -----------------------------
# Ajusta `csv_path` si el CSV está en otra ruta en Windows
csv_path = r"C:\ruta\a\Your Career Aspirations of GenZ.csv"  # <- cambia esta ruta si es necesario
try:
    df_raw = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
except Exception as e:
    try:
        df_raw = pd.read_csv('/mnt/data/Your Career Aspirations of GenZ.csv', encoding='utf-8', low_memory=False)
        print("Usando /mnt/data fallback para cargar el CSV.")
    except Exception as e2:
        print(f"No se encontró el CSV en `csv_path` ni en /mnt/data.\nError1: {e}\nError2: {e2}")
        # Buscar archivos en ubicaciones comunes y, si se encuentra uno, intentar cargarlo automáticamente
        search_dirs = [os.getcwd(), os.path.expanduser('~/Downloads'), os.path.expanduser('~/Desktop') ]
        print("\nBuscando archivos relevantes en carpetas comunes:")
        found_paths = []
        for d in search_dirs:
            try:
                for f in os.listdir(d):
                    if 'career' in f.lower() or 'genz' in f.lower() or f.lower().endswith('.csv'):
                        full = os.path.join(d, f)
                        print('   ', full)
                        found_paths.append(full)
            except Exception as le:
                print(f"  No se pudo listar {d}: {le}")
        if found_paths:
            # Priorizar coincidencia exacta por nombre de archivo
            exact_name = 'your career aspirations of genz.csv'
            chosen = None
            for p in found_paths:
                if os.path.basename(p).lower() == exact_name:
                    chosen = p
                    break
            if chosen is None:
                # buscar coincidencias con 'career' o 'genz' en el nombre
                candidates = [p for p in found_paths if ('career' in os.path.basename(p).lower() or 'genz' in os.path.basename(p).lower())]
                if len(candidates) == 1:
                    chosen = candidates[0]
                else:
                    print('\nSe encontraron varios archivos candidatos. Por favor especifica `csv_path` explícitamente o mueve el archivo deseado al workspace.')
                    for i,p in enumerate(found_paths,1):
                        print(f'  {i}. {p}')
                    raise FileNotFoundError('Varios candidatos encontrados; seleccione manualmente el CSV correcto asignando `csv_path`.')
            csv_path = chosen
            print(f"Intentando cargar desde: {csv_path}")
            df_raw = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        else:
            raise FileNotFoundError("CSV no encontrado. Revisa las salidas anteriores y proporciona la ruta correcta o sube el archivo al workspace.")
print("Dataset cargado. Dimensiones:", df_raw.shape)
# Verificar que las columnas objetivo esperadas existen antes de entrenar modelos
expected_targets = ["What is the most preferred working environment for you.", "Would you definitely pursue a Higher Education / Post Graduation outside of India ? If only you have to self sponsor it."]
missing_targets = [t for t in expected_targets if t not in df_raw.columns]
if missing_targets:
    print("Columnas objetivo esperadas no encontradas en el CSV cargado:", missing_targets)
    print("Columnas disponibles en el CSV:")
    print(df_raw.columns.tolist())
    raise KeyError("Las columnas objetivo no están presentes en el CSV cargado. Ajusta `csv_path` o carga el CSV correcto antes de continuar.")

# Inspección rápida
display(df_raw.head())
print('\nInformación:')
print(df_raw.info())

# Valores nulos por columna y porcentaje
null_counts = df_raw.isnull().sum()
null_percent = (null_counts / len(df_raw)) * 100
print('\nValores nulos por columna:')
print(null_counts)
print('\nPorcentaje de nulos por columna:')
print(null_percent)

# Gráfica 1: Distribución de la primera variable categórica (si existe)
if df_raw.shape[1] > 0:
    first_col = df_raw.columns[0]
    try:
        plt.figure(figsize=(10,5))
        df_raw[first_col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Distribución de "{first_col}"', fontsize=14, fontweight='bold')
        plt.xlabel(first_col, fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print('No se pudo graficar la primera columna:', e)

# ============================================================
# ACTIVIDAD 1 – EDA Y LIMPIEZA
# ============================================================

print("\n" + "="*50)
print("ACTIVIDAD 1 – Análisis exploratorio y limpieza")
print("="*50)

# Información general
print("\nInformación general:")
print(df_raw.info())

print("\nValores nulos por columna (original):")
print(df_raw.isnull().sum())

# Porcentaje de nulos
total_rows = len(df_raw)
null_percent = (df_raw.isnull().sum() / total_rows) * 100
print("\nPorcentaje de valores nulos por columna:")
print(null_percent)

# Gráfica 1: Distribución de la primera variable categórica
first_col = df_raw.columns[0]
plt.figure(figsize=(10,5))
df_raw[first_col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title(f'Distribución de "{first_col}"', fontsize=14, fontweight='bold')
plt.xlabel(first_col, fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -----------------------------
# LIMPIEZA BÁSICA
# -----------------------------
df_clean = df_raw.copy()

# Eliminar duplicados
df_clean = df_clean.drop_duplicates()
print(f"\nDuplicados eliminados. Nuevas dimensiones: {df_clean.shape}")

# Imputación de valores nulos (usando mediana para numéricas y moda para categóricas)
# Nota: Esto se hace con todo el dataset, pero en producción real se recomienda hacerlo dentro del pipeline.
numeric_cols = df_clean.select_dtypes(include=np.number).columns
categorical_cols = df_clean.select_dtypes(include='object').columns

for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print("\nValores nulos después de la imputación:")
print(df_clean.isnull().sum())

print("\n" + "="*50)
print("ACTIVIDAD 1 COMPLETADA. Dataset limpio listo.")
print("="*50)

# ============================================================
# ACTIVIDAD 2 – ENTRENAMIENTO DE MODELOS INICIALES
# ============================================================

print("\n" + "="*50)
print("ACTIVIDAD 2 – Entrenamiento de modelos")
print("="*50)

# Usamos el dataframe limpio
df = df_clean.copy()

# -----------------------------
# MODELO 1 – Entorno laboral preferido
# -----------------------------
target1 = "What is the most preferred working environment for you."
X1 = df.drop(columns=[target1])
y1 = df[target1]

# Identificar columnas categóricas (para el preprocesador)
categorical_features = X1.select_dtypes(include='object').columns.tolist()
numeric_features = X1.select_dtypes(include=np.number).columns.tolist()

# Preprocesador con imputación y one-hot encoding
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor1 = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline con Random Forest (sin tuning aún)
model1 = Pipeline(steps=[
    ('preprocessor', preprocessor1),
    ('classifier', RandomForestClassifier(random_state=42))
])

# División train/test
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1)

# Entrenamiento
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

# Métricas
acc1 = accuracy_score(y1_test, y1_pred)
print("\nMODELO 1 – Entorno laboral")
print(f"Accuracy (sin tuning): {acc1:.4f}")
print("\nClassification Report:")
print(classification_report(y1_test, y1_pred))

# Matriz de confusión mejorada
cm1 = confusion_matrix(y1_test, y1_pred)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=model1.classes_)
fig, ax = plt.subplots(figsize=(8,6))
disp1.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Matriz de Confusión – Modelo 1 (Entorno laboral)', fontsize=14, fontweight='bold')
plt.show()

# -----------------------------
# MODELO 2 – Educación superior fuera de India
# -----------------------------
target2 = "Would you definitely pursue a Higher Education / Post Graduation outside of India ? If only you have to self sponsor it."
X2 = df.drop(columns=[target2])
y2 = df[target2]

# Preprocesador similar
categorical_features2 = X2.select_dtypes(include='object').columns.tolist()
numeric_features2 = X2.select_dtypes(include=np.number).columns.tolist()

preprocessor2 = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features2),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features2)
    ])

model2 = Pipeline(steps=[
    ('preprocessor', preprocessor2),
    ('classifier', RandomForestClassifier(random_state=42))
])

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2)

model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

acc2 = accuracy_score(y2_test, y2_pred)
print("\nMODELO 2 – Educación superior")
print(f"Accuracy (sin tuning): {acc2:.4f}")
print("\nClassification Report:")
print(classification_report(y2_test, y2_pred))

cm2 = confusion_matrix(y2_test, y2_pred)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=model2.classes_)
fig, ax = plt.subplots(figsize=(8,6))
disp2.plot(ax=ax, cmap='Oranges', values_format='d')
ax.set_title('Matriz de Confusión – Modelo 2 (Educación superior)', fontsize=14, fontweight='bold')
plt.show()

print("\n" + "="*50)
print("ACTIVIDAD 2 COMPLETADA.")
print("="*50)

# ============================================================
# ACTIVIDAD 3 – OPTIMIZACIÓN FINAL (CON TUNING Y ENSAMBLES)
# ============================================================

print("\n" + "="*50)
print("ACTIVIDAD 3 – Optimización final")
print("="*50)

# Seguimos con el mismo dataframe limpio
# Nos enfocaremos en el modelo de educación superior (target2), que es el que se usará en la API.
# También exploraremos la conversión a binario.

# -----------------------------
# ANÁLISIS DE LA VARIABLE OBJETIVO ORIGINAL
# -----------------------------
print("\nValores únicos en la variable objetivo (educación superior):")
print(df[target2].value_counts())

# Decidimos mantenerla como multiclase para esta optimización, pero también haremos una versión binaria.

# -----------------------------
# OPCIÓN 1: MODELO MULTICLASE CON RANDOM FOREST OPTIMIZADO (GRID SEARCH)
# -----------------------------
print("\n--- Optimización con GridSearchCV (multiclase) ---")

# Usamos el mismo preprocesador y pipeline base
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor2),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Definimos el grid de hiperparámetros
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 15, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__class_weight': ['balanced', None]
}

# Grid search con validación cruzada
grid_search = GridSearchCV(pipeline_rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X2_train, y2_train)

print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor accuracy (CV): {grid_search.best_score_:.4f}")

# Evaluación en test
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X2_test)
acc_best = accuracy_score(y2_test, y_pred_best)
print(f"Accuracy en test (RF optimizado): {acc_best:.4f}")

# Matriz de confusión del mejor modelo
cm_best = confusion_matrix(y2_test, y_pred_best)
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=best_rf.classes_)
fig, ax = plt.subplots(figsize=(8,6))
disp_best.plot(ax=ax, cmap='Greens', values_format='d')
ax.set_title('Matriz de Confusión – Random Forest Optimizado', fontsize=14, fontweight='bold')
plt.show()

# -----------------------------
# OPCIÓN 2: GRADIENT BOOSTING (SIN TUNING EXHAUSTIVO, PERO CON PIPELINE)
# -----------------------------
print("\n--- Gradient Boosting ---")
pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor2),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

pipeline_gb.fit(X2_train, y2_train)
y_pred_gb = pipeline_gb.predict(X2_test)
acc_gb = accuracy_score(y2_test, y_pred_gb)
print(f"Accuracy Gradient Boosting: {acc_gb:.4f}")

cm_gb = confusion_matrix(y2_test, y_pred_gb)
disp_gb = ConfusionMatrixDisplay(confusion_matrix=cm_gb, display_labels=pipeline_gb.classes_)
fig, ax = plt.subplots(figsize=(8,6))
disp_gb.plot(ax=ax, cmap='Purples', values_format='d')
ax.set_title('Matriz de Confusión – Gradient Boosting', fontsize=14, fontweight='bold')
plt.show()

# -----------------------------
# OPCIÓN 3: ENSAMBLE VOTING (SOFT VOTING) CON LOS DOS CLASIFICADORES
# -----------------------------
print("\n--- Ensamble Voting (soft) ---")
# Nota: Para VotingClassifier necesitamos que los estimadores tengan el mismo preprocesamiento.
# Extraemos los pasos de preprocesamiento para aplicarlos una sola vez.
X2_train_preprocessed = preprocessor2.fit_transform(X2_train)
X2_test_preprocessed = preprocessor2.transform(X2_test)

# Creamos los clasificadores individuales
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)

voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('gb', gb_clf)],
    voting='soft'
)

voting_clf.fit(X2_train_preprocessed, y2_train)
y_pred_vote = voting_clf.predict(X2_test_preprocessed)
acc_vote = accuracy_score(y2_test, y_pred_vote)
print(f"Accuracy Ensamble Voting: {acc_vote:.4f}")

cm_vote = confusion_matrix(y2_test, y_pred_vote)
disp_vote = ConfusionMatrixDisplay(confusion_matrix=cm_vote, display_labels=voting_clf.classes_)
fig, ax = plt.subplots(figsize=(8,6))
disp_vote.plot(ax=ax, cmap='Reds', values_format='d')
ax.set_title('Matriz de Confusión – Ensamble Voting', fontsize=14, fontweight='bold')
plt.show()

# -----------------------------
# OPCIÓN 4: CONVERSIÓN A BINARIO (SOLO PARA COMPARAR)
# -----------------------------
print("\n--- Versión binaria de la variable objetivo ---")

# Analizamos los valores para decidir la agrupación
valores_originales = df[target2].unique()
print("Valores originales:", valores_originales)

# Definimos una función de mapeo razonable:
# Consideramos "Yes" como respuesta afirmativa, todo lo demás como "No".
# Ajustamos para capturar variantes como "Yes, definitely", "Yes, maybe", etc.
def map_to_binary(respuesta):
    if pd.isna(respuesta):
        return "NO"
    respuesta_str = str(respuesta).lower()
    if "yes" in respuesta_str:
        return "YES"
    else:
        return "NO"

df_bin = df.copy()
df_bin["Higher_Ed_Binary"] = df_bin[target2].apply(map_to_binary)

# Ver distribución
print("\nDistribución binaria:")
print(df_bin["Higher_Ed_Binary"].value_counts())

X_bin = df_bin.drop(columns=[target2, "Higher_Ed_Binary"])
y_bin = df_bin["Higher_Ed_Binary"]

# Preprocesador para datos binarios
categorical_features_bin = X_bin.select_dtypes(include='object').columns.tolist()
numeric_features_bin = X_bin.select_dtypes(include=np.number).columns.tolist()

preprocessor_bin = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features_bin),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features_bin)
    ])

pipeline_bin = Pipeline(steps=[
    ('preprocessor', preprocessor_bin),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

pipeline_bin.fit(X_bin_train, y_bin_train)
y_bin_pred = pipeline_bin.predict(X_bin_test)
acc_bin = accuracy_score(y_bin_test, y_bin_pred)
print(f"Accuracy modelo binario: {acc_bin:.4f}")

print("\nClassification Report (binario):")
print(classification_report(y_bin_test, y_bin_pred))

cm_bin = confusion_matrix(y_bin_test, y_bin_pred)
disp_bin = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=pipeline_bin.classes_)
fig, ax = plt.subplots(figsize=(8,6))
disp_bin.plot(ax=ax, cmap='cividis', values_format='d')
ax.set_title('Matriz de Confusión – Modelo Binario', fontsize=14, fontweight='bold')
plt.show()

# -----------------------------
# RESUMEN FINAL DE RENDIMIENTO
# -----------------------------
print("\n" + "="*50)
print("RESUMEN DE RENDIMIENTO (Accuracy en test)")
print("="*50)
print(f"Modelo 2 original (sin tuning):         {acc2:.4f}")
print(f"Random Forest optimizado (GridSearch): {acc_best:.4f}")
print(f"Gradient Boosting:                      {acc_gb:.4f}")
print(f"Ensamble Voting:                         {acc_vote:.4f}")
print(f"Modelo Binario:                          {acc_bin:.4f}")

print("\n" + "="*50)
print("ACTIVIDAD 3 COMPLETADA.")
print("="*50)

# ============================================================
# NOTA: El mejor modelo (por accuracy y simplicidad) puede ser guardado para la API.
# Por ejemplo, el Random Forest optimizado o el Voting, según se prefiera.
# ============================================================
# Guardar el modelo elegido (ejemplo: best_rf) con joblib
# import joblib
# joblib.dump(best_rf, 'modelo_educacion_superior.pkl')



