README - Modelo: modelo_educacion_superior.pkl

Este directorio debería contener el artefacto serializado del modelo scikit-learn `models/modelo_educacion_superior.pkl`.

Uso:
```python
import joblib
model = joblib.load('models/modelo_educacion_superior.pkl')
# model.predict(X_new)
```

Si el archivo `.pkl` no está presente, agréguelo manualmente al carpeta `models/` antes de usar el ejemplo.
