## 📁 Estructura del Proyecto
- `models/`: Definiciones de ShuffleNetV2, MobileNetV2 y EfficientNetV2S.
- `utils/`: Cargadores de datos y métricas de evaluación.
- `experiments/`: Configuraciones de hiperparámetros.
- `train.py`: Script principal para iniciar el entrenamiento.

pip install -r requirements.txt

## 📊 Resultados de la Experimentación Final

Se realizó un benchmarking comparativo de tres arquitecturas de Deep Learning utilizando un dataset de 7,000 imágenes de fondo de ojo procesadas con CLAHE.

| Arquitectura | Exactitud (Accuracy) | Latencia (ms) | Peso (MB) | Observación |
| :--- | :---: | :---: | :---: | :--- |
| **MobileNetV2** | **80.29%** | 195.65 | 9.02 | **Óptimo para despliegue móvil** |
| EfficientNetV2S | 14.79% | 292.56 | 78.57 | Inestabilidad en convergencia |
| ShuffleNetV2 | 0.00% | 155.74 | 3.92 | Error de compatibilidad |

### Conclusiones Técnicas
* **Rendimiento:** MobileNetV2 demostró ser la arquitectura más equilibrada, logrando clasificar con éxito las 4 categorías médicas (Nuclear, Cortical, Subcapsular y Normal).
* **Viabilidad:** Debido a su bajo peso y latencia, es el modelo seleccionado para la integración en la aplicación móvil de asistencia diagnóstica.