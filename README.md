# supervisedML

Repositorio de trabajo con **aprendizaje supervisado** en Python. Recoge el código y los experimentos donde se aplican modelos de clasificación y regresión siguiendo un flujo de trabajo reproducible con scikit-learn.

## Cómo se ha trabajado

### 1. Exploración de los datos
Antes de entrenar nada se analiza cada dataset: tipos de variables, valores nulos, distribuciones, valores atípicos y correlaciones. Esto guía las decisiones de preprocesado y ayuda a detectar problemas como el desbalanceo de clases.

### 2. Preprocesado
- Tratamiento de valores nulos y atípicos.
- Codificación de variables categóricas.
- Escalado de variables numéricas cuando el modelo lo requiere (k-NN, SVM, regresión regularizada).
- Todo el preprocesado se encapsula en `Pipeline` y `ColumnTransformer` de scikit-learn, de modo que las transformaciones se ajustan solo con los datos de entrenamiento y se evita la fuga de información (*data leakage*).

### 3. División de los datos
Separación en conjuntos de entrenamiento y test (con estratificación en clasificación) y semilla aleatoria fija para que los resultados se puedan reproducir. El conjunto de test no se toca hasta la evaluación final.

### 4. Modelos base
Se empieza siempre por modelos sencillos que sirven de referencia (regresión lineal o logística, k-NN, árboles de decisión) y después se prueban modelos más potentes (random forest, gradient boosting, SVM). Así se puede medir si la complejidad adicional aporta mejora real.

### 5. Validación y ajuste de hiperparámetros
- Validación cruzada (k-fold, estratificada en clasificación) para estimar el rendimiento sin depender de una única partición.
- Búsqueda de hiperparámetros con `GridSearchCV` o `RandomizedSearchCV` sobre el conjunto de entrenamiento.

### 6. Evaluación
- **Clasificación**: accuracy, precision, recall, F1, matriz de confusión y curva ROC/AUC.
- **Regresión**: MAE, RMSE y R².

Las métricas se eligen según el problema (por ejemplo, F1 o recall cuando las clases están desbalanceadas) y se comparan entre modelos con los mismos datos y la misma validación.

### 7. Interpretación y conclusiones
Análisis de la importancia de las variables, revisión de errores y comparación final entre modelos, valorando rendimiento, complejidad y capacidad de generalización.

## Herramientas

- Python
- numpy y pandas para manipulación de datos
- scikit-learn para modelos, pipelines y validación
- matplotlib y seaborn para visualización
- Jupyter para los experimentos

## Reproducibilidad

- Semillas aleatorias fijas en las particiones y en los modelos.
- Preprocesado dentro de pipelines para que el mismo flujo se aplique igual en entrenamiento y test.
- Evaluación final única sobre el conjunto de test.

## Autor

**David Santiago Ruiz**
Estudiante del Grado en Ciencia de Datos e Inteligencia Artificial (ETSISI-UPM)
