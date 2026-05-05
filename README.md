# 📊 Pulso de Calidad SPC

Aplicación desarrollada por **Jerson Andrés López Wilches** para el análisis completo de Control Estadístico de Procesos (SPC).

Permite evaluar estabilidad, capacidad y desempeño de procesos mediante herramientas estadísticas avanzadas y visualización interactiva.

---

## 🚀 Funcionalidades

### 📂 Gestión de datos
- Carga de archivos CSV y Excel
- Selección de variables de análisis
- Limpieza y conversión automática de datos

### 📈 Análisis de supuestos
- Normalidad (Shapiro-Wilk, gráficos Q-Q)
- Independencia (Durbin-Watson, autocorrelación)
- Homocedasticidad
- Detección de valores atípicos

### 📊 Gráficos de control
- I-MR (individuales)
- X̄ - R (subgrupos)
- p, np (proporciones)
- c, u (defectos)

### ⚙️ Análisis de capacidad
- Cp, Cpk, CPL, CPU
- Pp, Ppk, PPL, PPU
- Cpm
- Z hacia LIE y LSE
- Producto no conforme (PNC)
- Diagnóstico automático del proceso

### 📉 Producto no conforme
- Identificación de unidades fuera de especificación
- Clasificación por LIE y LSE
- Análisis tipo Pareto

### 📦 Muestreo de aceptación
- Curva OC
- Riesgo del productor (α)
- Riesgo del consumidor (β)
- Evaluación de planes de muestreo

### 🧠 Asistente de análisis
- Recomendación automática de herramientas SPC
- Guía para selección de gráficos y métodos

### ⏱ Monitoreo del proceso
- Simulación en tiempo real
- Ventana móvil de análisis
- Detección de tendencias y cambios

### 📑 Reporte automático
- Generación de reporte en Excel
- Inclusión de resultados e indicadores clave

---

## 🛠️ Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- SciPy
- Statsmodels
- Plotly
- OpenPyXL

---

## ▶️ Ejecución local

1. Instalar dependencias:

```bash
python -m pip install -r requirements.txt
