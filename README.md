# supervisedML

Práctica 1 de **Aprendizaje Automático 1** (Grupo 7, curso 2025-26) del Grado en Ciencia de Datos e Inteligencia Artificial (ETSISI-UPM).

El objetivo es predecir si una persona gana más de 50K al año (`salario`: `<=50K` / `>50K`) a partir de variables demográficas y laborales. Es un problema de clasificación binaria con las clases desbalanceadas (~76 % / ~24 %), por lo que la métrica principal es el **F1-score de la clase positiva (`>50K`)**.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `preprocess.ipynb` | Limpieza, preprocesado, selección de características y exportación del dataset procesado |
| `supervised.ipynb` | Comparativa de modelos, búsqueda de hiperparámetros y elección del modelo |
| `best_model.ipynb` | Entrenamiento del modelo final sobre todos los datos y generación de las predicciones para el conjunto de test |
| `salario.csv` | Datos de entrenamiento |
| `salario_test.csv` | Datos de test sin etiquetas (competición de Kaggle) |
| `salario_test_example.csv` | Ejemplo del formato de entrega |
| `processed_dataset.csv.zip` | Dataset preprocesado (24 características) generado por `preprocess.ipynb` |
| `memoria.pdf` | Memoria de la práctica |

## Cómo se ha trabajado

### 1. Preprocesado (`preprocess.ipynb`)

- **Valores nulos:** `salario.csv` marca los datos faltantes con `?` (en `trabajo`, `trabajo.1` y `pais-origen`). Se recargan los datos con `na_values=['?']` y se imputan con la **moda calculada solo con el conjunto de entrenamiento**.
- **División:** 80 % entrenamiento / 20 % validación, con `stratify=y` y `random_state=42`. La división se hace antes de imputar o escalar para evitar *data leakage*.
- **Codificación:**
  - `estudios` (ordinal): `OrdinalEncoder` con orden explícito de menor a mayor nivel.
  - `sexo` (binaria): `OrdinalEncoder`.
  - Resto de categóricas (`trabajo`, `estado-civil`, `trabajo.1`, `posicion-familiar`, `etnia`, `pais-origen`): `OneHotEncoder`, lo que deja 87 características.
- **Dos pipelines comparados:**
  - **V1:** `RobustScaler` + `SMOTE` para el desbalanceo.
  - **V2:** `StandardScaler` + `class_weight='balanced'` (sin tocar los datos).
- **Selección de características (k = 30):** filtro (`SelectKBest` con `f_classif`), wrapper (`RFE` con regresión logística) y embebido (`SelectFromModel` con regresión logística L1). Se evaluaron con una regresión logística "juez" usando F1, junto con dos consensos: características elegidas por al menos 2 de los 3 métodos o por los 3.
- **Decisión:** se descarta V1 (SMOTE) y se queda **V2 con el consenso "al menos 2": 24 características**. El embebido V2 obtuvo un F1 ligeramente superior (0.6841 frente a 0.6798), pero se prefirió el consenso por ser más robusto.
- Se exportan las figuras de importancia de características y matrices de confusión de V1 y V2, y el dataset procesado a `processed_dataset.csv.zip`.

### 2. Comparativa de modelos (`supervised.ipynb`)

- Se parte del dataset procesado y de la división original: primeras 22.398 filas para entrenamiento y últimas 5.600 para test.
- Validación cruzada con `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Búsqueda de hiperparámetros con **`GridSearchCV`** optimizando `f1`. No se usa `RandomizedSearchCV`.
- Desbalanceo tratado con ponderación de clases (`class_weight='balanced'`, `scale_pos_weight` en XGBoost, `auto_class_weights='Balanced'` en CatBoost).
- Primera búsqueda con rejillas reducidas para todas las familias, para tener una línea base de cada una:

| Modelo | F1 (clase 1) en test |
|---|---|
| Regresión logística | 0.6797 |
| Regresión polinomial (grado 2) | 0.6929 |
| SGD Classifier | 0.6876 |
| Árbol de decisión | 0.6828 |
| Random Forest | 0.7062 |
| KNN | 0.6462 |
| SVM lineal (`LinearSVC`) | 0.6758 |
| SVM (`SVC`, kernel `poly` elegido) | 0.6849 |
| XGBoost | 0.7194 |
| LightGBM | 0.7210 |
| CatBoost | 0.7182 |
| Red neuronal (Keras), umbral 0.5 | 0.6940 |
| Red neuronal (Keras), umbral 0.69 | 0.7109 |

- **Red neuronal:** `Input(24) → Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.3) → Dense(1, sigmoid)`, Adam (lr 0.001), 50 épocas, batch 64 y pesos de clase. Se ajustó el **umbral de decisión** (de 0.5 a 0.69), lo que mejoró la precisión de la clase 1 a costa de algo de recall.
- **Optimización exhaustiva** (`GridSearchCV` con rejillas densas) para los tres modelos de boosting:

| Modelo | Combinaciones × folds | F1 (clase 1) en test |
|---|---|---|
| XGBoost | 1728 × 5 | **0.7230** |
| LightGBM | 1152 × 5 | 0.7215 |
| CatBoost | 1152 × 5 | 0.7185 |

La mejora respecto a la primera búsqueda es mínima, así que se interpreta como el techo de rendimiento con estas 24 características.

### 3. Modelo final (`best_model.ipynb`)

- Se replica el preprocesado V2 (imputación con la moda de entrenamiento, `StandardScaler`, codificaciones y selección por consenso, que vuelve a dar 24 características) entrenando con `salario.csv` completo y aplicando solo `transform` a `salario_test.csv`.
- Aunque XGBoost fue el mejor en la validación local (F1 0.7230), al enviar predicciones a Kaggle la **red neuronal** generalizó mejor (F1 público de 0.861), así que es el modelo final.
- Se entrena con `class_weight` balanceado y se aplica el umbral óptimo 0.69 para generar el archivo de predicciones en formato `ID,salario` (`test_labels.csv`).

## Tecnologías

Python, pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, LightGBM, CatBoost, TensorFlow/Keras con SciKeras, matplotlib y seaborn.

## Autores

- David Santiago Ruiz
- Brian Bedoya Piedrahita

Grado en Ciencia de Datos e Inteligencia Artificial (ETSISI-UPM)
