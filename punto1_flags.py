import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("       FASE 1: ANÁLISIS EXPLORATORIO (EDA) - DATASET FLAGS")
print("="*60)

# Se definen las 30 columnas originales del repositorio
columnas_flags = [
    'name', 'landmass', 'zone', 'area', 'population', 'language', 'religion',
    'bars', 'stripes', 'colours', 'red', 'green', 'blue', 'gold', 'white', 
    'black', 'orange', 'mainhue', 'circles', 'crosses', 'saltires', 'quarters', 
    'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft', 'botright'
]

#Cargamos el archivo .data ciego
df_flags_raw = pd.read_csv('flag.data', names=columnas_flags)

#FILTRO: Solo variables continuas/conteo + nuestra etiqueta (religion)
cols_numericas = ['religion', 'area', 'population', 'bars', 'stripes', 'colours', 'circles', 'crosses', 'saltires', 'quarters', 'sunstars']
df_flags = df_flags_raw[cols_numericas]

print(f"Total de registros filtrados y listos para Gauss: {len(df_flags)}")

#2. PROBABILIDADES A PRIORI
print("\n[PUNTO 2] Probabilidades A Priori de cada Religión:")
total_muestras = len(df_flags)
clases_religiones = np.sort(np.unique(df_flags['religion']))

for c in clases_religiones:
    muestras_clase = len(df_flags[df_flags['religion'] == c])
    prob_a_priori = muestras_clase / total_muestras
    print(f"  * Religión {c}: Muestras = {muestras_clase} | P(Rel {c}) = {prob_a_priori:.4f}")

#3. MEDIA (μ) Y DESVIACIÓN ESTÁNDAR (σ) POR CLASE
print("\n[PUNTO 3] Media (μ) y Desviación Estándar (σ) por característica:")
for c in clases_religiones:
    df_clase = df_flags[df_flags['religion'] == c].drop('religion', axis=1)
    muestras = len(df_clase)
    
    # Si hay muy poquitos países en una religión, advertimos que la desviación puede fallar
    if muestras > 1:
        print(f"\n--- ESTADÍSTICAS RELIGIÓN {c} (Países: {muestras}) ---")
        resumen = pd.DataFrame({'Media (μ)': df_clase.mean(), 'Desviación (σ)': df_clase.std()})
        print(resumen.round(3))
    else:
        print(f"\n--- ESTADÍSTICAS RELIGIÓN {c} (Países: {muestras}) ---")
        print("  [!] Advertencia: Muy pocos datos para calcular desviación estándar real.")

#4,5- GRÁFICAS DE DENSIDAD KDE
print("\n[PUNTOS 4 Y 5] Generando gráficas de densidad (KDE)...")
# Graficaremos Área y Número de Colores
variables_kdes = ['colours', 'area']

for var in variables_kdes:
    plt.figure(figsize=(8, 4))
    for c in clases_religiones:
        subset = df_flags[df_flags['religion'] == c]
        if len(subset) > 1: # KDE necesita al menos 2 datos para trazar la curva
            sns.kdeplot(data=subset, x=var, label=f'Religión {c}', fill=True, alpha=0.2, warn_singular=False)
    plt.title(f'Distribución KDE de: {var} (Flags)')
    plt.xlabel(var)
    plt.ylabel('Densidad')
    plt.legend()
    plt.tight_layout()
    plt.show()

#6. MATRIZ DE CORRELACIÓN POR CLASE
print("\n[PUNTO 6] Generando Matrices de Correlación...")
# Para no saturarte la pantalla con 8 gráficas, solo mostraremos las de las 3 religiones más grandes (0, 1 y 2)
religiones_principales = [0, 1, 2] 
print(f"Mostrando matrices para las religiones mayoritarias: {religiones_principales}")

for c in religiones_principales:
    plt.figure(figsize=(8, 6))
    subset = df_flags[df_flags['religion'] == c].drop('religion', axis=1)
    matriz_corr = subset.corr()
    
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Matriz de Correlación - Religión {c} (Flags)')
    plt.tight_layout()
    plt.show()

print("\n[PROCESO TERMINADO]")
