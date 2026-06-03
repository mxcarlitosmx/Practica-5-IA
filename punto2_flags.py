import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# 1. CARGA DE DATOS FLAGS (Aplicando el filtro del EDA)
columnas_flags = ['name', 'landmass', 'zone', 'area', 'population', 'language', 'religion',
                  'bars', 'stripes', 'colours', 'red', 'green', 'blue', 'gold', 'white', 
                  'black', 'orange', 'mainhue', 'circles', 'crosses', 'saltires', 'quarters', 
                  'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft', 'botright']
df_flags_raw = pd.read_csv('flag.data', names=columnas_flags)
cols_num = ['religion', 'area', 'population', 'bars', 'stripes', 'colours', 'circles', 'crosses', 'saltires', 'quarters', 'sunstars']
df_flags = df_flags_raw[cols_num]

X = df_flags.drop('religion', axis=1).values
y = df_flags['religion'].values


# [PUNTO 8] MOTOR MATEMÁTICO DEL CLASIFICADOR (REUTILIZACIÓN)
class NaiveBayesGaussiano:
    def fit(self, X_train, y_train):
        self.clases = np.unique(y_train)
        self.parametros = {}
        for c in self.clases:
            X_c = X_train[y_train == c]
            # Si una clase tiene 1 elemento o 0 durante un Fold, forzamos valores seguros
            if len(X_c) == 0: continue
            self.parametros[c] = {
                'a_priori': len(X_c) / len(X_train),
                'media': np.mean(X_c, axis=0),
                'std': np.std(X_c, axis=0) + 1e-9
            }
            
    def verosimilitud_gaussiana(self, x, mean, std):
        exponente = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponente

    def predict(self, X_test):
        predicciones = []
        for x in X_test:
            probs_posteriores = []
            clases_validas = list(self.parametros.keys())
            for c in clases_validas:
                prior = np.log(self.parametros[c]['a_priori'])
                verosimilitud = np.sum(np.log(self.verosimilitud_gaussiana(x, self.parametros[c]['media'], self.parametros[c]['std'])))
                probs_posteriores.append(prior + verosimilitud)
            predicciones.append(clases_validas[np.argmax(probs_posteriores)])
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
print("     RESULTADOS DE VALIDACIÓN NAIVE BAYES - FLAGS")
print("="*60)
print(f"{'Método':<20} | {'Motor Propio':<15} | {'Scikit-Learn':<15}")
print("-" * 60)

# --- HOLD-OUT 80/20 ---
corte = int(len(X) * 0.8)
X_train_h, X_test_h = X_rev[:corte], X_rev[corte:]
y_train_h, y_test_h = y_rev[:corte], y_rev[corte:]

modelo_propio.fit(X_train_h, y_train_h)
modelo_sk.fit(X_train_h, y_train_h)
print(f"{'Hold-Out 80/20':<20} | {calcular_accuracy(y_test_h, modelo_propio.predict(X_test_h)):.4f}          | {calcular_accuracy(y_test_h, modelo_sk.predict(X_test_h)):.4f}")

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
