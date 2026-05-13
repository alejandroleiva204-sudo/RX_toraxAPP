Aquí tienes el README:

---

# RX Tórax — Sistema de Análisis de Calidad

Sistema de análisis automático de calidad para radiografías de tórax, desarrollado en Python con Streamlit. Detecta y corrige artefactos de imagen que comprometen la aptitud diagnóstica, generando un reporte cuantitativo antes/después de la corrección.

---

## Características principales

- Detección independiente de sobreexposición y blur (una imagen puede presentar ambos simultáneamente)
- Corrección automática adaptada a la severidad detectada
- Dashboard interactivo con 8 KPIs, histograma comparativo y tabla de métricas
- Reporte exportable en texto plano
- Interfaz oscura optimizada para entornos clínicos

---

## Instalación

```bash
pip install streamlit opencv-python scikit-image scipy matplotlib Pillow numpy
```

```bash
streamlit run rx_torax.py
```

---

## Flujo de procesamiento

### 1. Preprocesamiento

La imagen cargada se convierte a escala de grises, se normaliza al rango 0–255 y se le aplica un filtro gaussiano configurable (σ por defecto: 1.0). Opcionalmente puede redimensionarse antes del análisis.

### 2. Cálculo de métricas

Se calculan 8 métricas cuantitativas sobre la imagen preprocesada:

| Métrica | Descripción |
|---|---|
| Varianza del Laplaciano | Nitidez global de la imagen |
| Contraste RMS | Desviación estándar de intensidades |
| SNR (dB) | Relación señal/ruido |
| Entropía de Shannon | Riqueza de información |
| Intensidad media | Brillo promedio |
| Saturación (%) | Porcentaje de píxeles >= 250 |
| Kurtosis | Forma de la distribución de intensidades |
| Píxeles oscuros (%) | Porcentaje de píxeles < 30 |

### 3. Clasificación dual e independiente

**Sobreexposición** — se confirma si al menos 2 de los siguientes criterios se activan:
- Intensidad media > 160
- Saturación > 2%
- Kurtosis < -0.5
- Píxeles oscuros > 25% (imagen "lavada")

La severidad se clasifica en leve, moderada o severa según la combinación de kurtosis, media y saturación.

**Blur** — se confirma únicamente si los 3 criterios se activan simultáneamente:
- Varianza del Laplaciano < 100
- SNR < 20 dB
- Entropía < 7.0

La severidad se clasifica en leve, moderada o severa según Laplaciano y SNR.

### 4. Corrección

Las correcciones se aplican en orden fijo: sobreexposición primero, blur después.

**Sobreexposición** — corrección por Gamma + CLAHE:
- Se aplica una curva gamma para oscurecer la imagen (γ entre 0.40 y 0.75 según severidad)
- Seguido de CLAHE (Contrast Limited Adaptive Histogram Equalization) para recuperar contraste local

**Blur** — pipeline sin estimación de PSF para evitar artefactos direccionales:
- Leve: Unsharp Mask (radio 2.0, cantidad 1.5)
- Moderada/severa: filtro bilateral + Unsharp Mask agresivo + realce suave de bordes con Laplaciano

Antes de aplicar la corrección de blur, se recalcula la varianza del Laplaciano sobre la imagen ya corregida de sobreexposición. Si supera 80, el blur era aparente (causado por la sobreexposición) y se aplica únicamente un unsharp suave en lugar del pipeline completo.

### 5. Evaluación post-corrección

Tras la corrección se recalculan todas las métricas y se reclasifica la imagen, determinando si quedó apta para diagnóstico.

---

## Decisión diagnóstica

| Resultado | Condición |
|---|---|
| **APTA PARA DIAGNÓSTICO** | Ningún artefacto confirmado |
| **DEGRADADA: SOBREEXPOSICIÓN (severidad)** | Solo sobreexposición confirmada |
| **DEGRADADA: BLUR (severidad)** | Solo blur confirmado |
| **DEGRADADA: SOBREEXPOSICIÓN + BLUR** | Ambos artefactos confirmados |

---

## Interfaz

- **Sidebar**: carga de imagen, parámetros de preprocesamiento, clasificación activa y exportación
- **Tab Comparación visual**: imagen original vs. corregida vs. diferencia amplificada ×5, con parámetros de corrección aplicados
- **Tab Histograma**: distribución de intensidades antes/después con criterios activos
- **Tab Métricas detalladas**: tabla comparativa con delta y estado por métrica
- **Tab Reporte**: reporte en texto plano descargable

---

## Formatos de imagen soportados

PNG, JPG, JPEG, BMP, TIFF

---

## Limitaciones

- El sistema evalúa calidad técnica de imagen, no contenido clínico ni patologías
- Imágenes con blur severo no pueden recuperar información perdida; la corrección reduce artefactos pero no restituye detalle original
- Los umbrales de clasificación están calibrados para radiografías de tórax estándar; pueden requerir ajuste para otras modalidades
