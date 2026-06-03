import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("       FASE 1: ANÁLISIS EXPLORATORIO (EDA) - DATASET WINE")
print("="*60)

# 1. Definimos los encabezados porque el archivo .data viene ciego
columnas_wine = [
    'Clase', 'Alcohol', 'Acido_Malico', 'Ceniza', 'Alcalinidad_Ceniza', 
    'Magnesio', 'Fenoles_Totales', 'Flavanoides', 'Fenoles_No_Flavanoides', 
    'Proantocianinas', 'Intensidad_Color', 'Matiz', 'OD280_OD315', 'Proline'
]

# Cargar el dataset
df_wine = pd.read_csv('wine.data', names=columnas_wine)

#2. PROBABILIDADES A PRIORI
print("\n[PUNTO 2] Probabilidades A Priori de cada Clase:")
total_muestras = len(df_wine)
clases = np.unique(df_wine['Clase'])

for c in clases:
    muestras_clase = len(df_wine[df_wine['Clase'] == c])
    prob_a_priori = muestras_clase / total_muestras
    print(f"  * Clase {c}: Muestras = {muestras_clase} | P(Clase {c}) = {prob_a_priori:.4f}")

#3. MEDIA (μ) Y DESVIACIÓN ESTÁNDAR (σ) POR CLASE
print("\n[PUNTO 3] Media (μ) y Desviación Estándar (σ) por característica:")
for c in clases:
    print(f"\n" + "-"*15 + f" ESTADÍSTICAS PARA LA CLASE {c} " + "-"*15)
    df_clase = df_wine[df_wine['Clase'] == c].drop('Clase', axis=1)
    resumen = pd.DataFrame({
        'Media (μ)': df_clase.mean(),
        'Desviación Estándar (σ)': df_clase.std()
    })
    print(resumen.round(4))

#4,5- GRÁFICAS DE DENSIDAD KDE (Ejemplo con dos variables clave)
print("\n[PUNTOS 4 Y 5] Generando gráficas de densidad (KDE)...")
# Graficaremos 'Alcohol' e 'Intensidad_Color' para verificar el ajuste Gaussiano
variables_kdes = ['Alcohol', 'Intensidad_Color']

for var in variables_kdes:
    plt.figure(figsize=(7, 4))
    for c in clases:
        sns.kdeplot(data=df_wine[df_wine['Clase'] == c], x=var, label=f'Clase {c}', fill=True, alpha=0.3)
    plt.title(f'Distribución KDE de la característica: {var}')
    plt.xlabel(var)
    plt.ylabel('Densidad')
    plt.legend()
    plt.tight_layout()
    plt.show()

# [PUNTO 6] MATRIZ DE CORRELACIÓN POR CLASE
print("\n[PUNTO 6] Generando Matrices de Correlación por Clase...")
for c in clases:
    plt.figure(figsize=(9, 7))
    # Calculamos la matriz de correlación de Pearson para esta clase
    matriz_corr = df_wine[df_wine['Clase'] == c].drop('Clase', axis=1).corr()
    
    # Dibujamos el mapa de calor (Heatmap)
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
    plt.title(f'Matriz de Correlación - Clase {c} (Wine)')
    plt.tight_layout()
    plt.show()

print("\n[PROCESO TERMINADO] Revisa las gráficas que se abrieron y copia los datos de la terminal.")
