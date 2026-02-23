"""
Ejemplo de carga del modelo guardado (si `models/modelo_educacion_superior.pkl` está presente).
"""
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'modelo_educacion_superior.pkl')

if not os.path.exists(MODEL_PATH):
    print('El archivo modelo no se encontró en:', MODEL_PATH)
    print('Coloca el archivo modelo_educacion_superior.pkl en la carpeta models/ y vuelve a ejecutar.')
else:
    model = joblib.load(MODEL_PATH)
    print('Modelo cargado. Ejemplo de uso:')
    print('  preds = model.predict(X_new)')
