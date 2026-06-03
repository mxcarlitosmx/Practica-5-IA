import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# 1. CARGA DE DATOS WINE
columnas_wine = ['Clase', 'Alcohol', 'Acido_Malico', 'Ceniza', 'Alcalinidad_Ceniza', 
                 'Magnesio', 'Fenoles_Totales', 'Flavanoides', 'Fenoles_No_Flavanoides', 
                 'Proantocianinas', 'Intensidad_Color', 'Matiz', 'OD280_OD315', 'Proline']
df_wine = pd.read_csv('wine.data', names=columnas_wine)
X = df_wine.drop('Clase', axis=1).values
y = df_wine['Clase'].values

# [PUNTO 8] MOTOR MATEMÁTICO DEL CLASIFICADOR (DESDE CERO)
class NaiveBayesGaussiano:
    def fit(self, X_train, y_train):
        self.clases = np.unique(y_train)
        self.parametros = {}
        # Calculamos Probabilidad a priori, Media (μ) y Desviación (σ)
        for c in self.clases:
            X_c = X_train[y_train == c]
            self.parametros[c] = {
                'a_priori': len(X_c) / len(X_train),
                'media': np.mean(X_c, axis=0),
                'std': np.std(X_c, axis=0) + 1e-9 # Epsilon para evitar división por 0
            }
            
    def verosimilitud_gaussiana(self, x, mean, std):
        # Ecuación de la campana de Gauss (Función de Densidad de Probabilidad)
        exponente = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponente

    def predict(self, X_test):
        predicciones = []
        for x in X_test:
            probs_posteriores = []
            for c in self.clases:
                # Regla de decisión de los apuntes: ln(P(Wk)) + sum(ln(P(Xi|Wk)))
                prior = np.log(self.parametros[c]['a_priori'])
                verosimilitud = np.sum(np.log(self.verosimilitud_gaussiana(x, self.parametros[c]['media'], self.parametros[c]['std'])))
                posterior = prior + verosimilitud
                probs_posteriores.append(posterior)
            # Gana la clase con la probabilidad logarítmica más alta
            predicciones.append(self.clases[np.argmax(probs_posteriores)])
        return np.array(predicciones)

def calcular_accuracy(y_real, y_pred):
    return np.mean(y_real == y_pred)

# [PUNTOS 9 Y 10] VALIDACIÓN Y COMPARACIÓN CON SCIKIT-LEARN
np.random.seed(42)
indices = np.random.permutation(len(X))
X_rev, y_rev = X[indices], y[indices]

modelo_propio = NaiveBayesGaussiano()
modelo_sk = GaussianNB()

print("="*60)
print("     RESULTADOS DE VALIDACIÓN NAIVE BAYES - WINE")
print("="*60)
print(f"{'Método':<20} | {'Motor Propio':<15} | {'Scikit-Learn':<15}")
print("-" * 60)

# --- HOLD-OUT 80/20 ---
corte = int(len(X) * 0.8)
X_train_h, X_test_h = X_rev[:corte], X_rev[corte:]
y_train_h, y_test_h = y_rev[:corte], y_rev[corte:]

modelo_propio.fit(X_train_h, y_train_h)
modelo_sk.fit(X_train_h, y_train_h)
acc_ho_p = calcular_accuracy(y_test_h, modelo_propio.predict(X_test_h))
acc_ho_sk = calcular_accuracy(y_test_h, modelo_sk.predict(X_test_h))
print(f"{'Hold-Out 80/20':<20} | {acc_ho_p:.4f}          | {acc_ho_sk:.4f}")

# --- 10-FOLD CROSS-VALIDATION ---
folds = 10
tam = len(X) // folds
acc_k_p, acc_k_sk = [], []

for i in range(folds):
    ini, fin = i * tam, (i + 1) * tam if i != folds - 1 else len(X)
    X_test_k, y_test_k = X_rev[ini:fin], y_rev[ini:fin]
    X_train_k = np.concatenate([X_rev[:ini], X_rev[fin:]])
    y_train_k = np.concatenate([y_rev[:ini], y_rev[fin:]])
    
    modelo_propio.fit(X_train_k, y_train_k)
    modelo_sk.fit(X_train_k, y_train_k)
    acc_k_p.append(calcular_accuracy(y_test_k, modelo_propio.predict(X_test_k)))
    acc_k_sk.append(calcular_accuracy(y_test_k, modelo_sk.predict(X_test_k)))
print(f"{'10-Fold CV':<20} | {np.mean(acc_k_p):.4f}          | {np.mean(acc_k_sk):.4f}")

# --- LEAVE-ONE-OUT ---
acc_loo_p, acc_loo_sk = [], []
for i in range(len(X)):
    X_test_l, y_test_l = X_rev[i:i+1], y_rev[i:i+1]
    X_train_l = np.concatenate([X_rev[:i], X_rev[i+1:]])
    y_train_l = np.concatenate([y_rev[:i], y_rev[i+1:]])
    
    modelo_propio.fit(X_train_l, y_train_l)
    modelo_sk.fit(X_train_l, y_train_l)
    acc_loo_p.append(calcular_accuracy(y_test_l, modelo_propio.predict(X_test_l)))
    acc_loo_sk.append(calcular_accuracy(y_test_l, modelo_sk.predict(X_test_l)))
print(f"{'Leave-One-Out':<20} | {np.mean(acc_loo_p):.4f}          | {np.mean(acc_loo_sk):.4f}")
print("="*60)
