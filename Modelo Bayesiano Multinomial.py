import pandas as pd
import numpy as np

# Crear DataFrame
data = {
    'Encuestadora': ['Rubrum', 'Electoralia', 'Massive Caller'],
    'Muestra': [1000, 2500, 600],
    'Nivel de confianza': [0.95, 0.95, 0.95],
    'Margen de error': [0.038, 0.0196, 0.043],
    'PAN-PRI-PRD': [47.3, 36, 43],
    'MORENA-PVEM-PT': [32.5, 51, 35.8],
    'MC': [8, 3, 7.2],
    'Indecisos': [12.2, 10, 12.2]
}

df = pd.DataFrame(data)
print(df)

from scipy.stats import dirichlet, multinomial

# Definir el número de categorías (partidos + indecisos)
num_categories = 4

# Datos de las encuestas en forma de proporciones
proportions = np.array([
    [0.473, 0.325, 0.08, 0.122],
    [0.36, 0.51, 0.03, 0.1],
    [0.43, 0.358, 0.072, 0.122]
])

# Asegurarse de que las dimensiones coincidan correctamente
counts = (proportions * df['Muestra'].values[:, None]).astype(int)

# Priori Dirichlet
alpha = np.ones(num_categories)

# Distribución Dirichlet
theta = dirichlet(alpha).rvs(size=1)

# Distribución Multinomial para cada encuesta
multinom_samples = np.array([multinomial.rvs(n, theta[0]) for n in df['Muestra']])

# Imprimir los resultados
print("Probabilidades iniciales (theta):", theta)
print("Muestras multinomiales:", multinom_samples)

# Calcular el alfa actualizando con los datos observados
alpha_post = alpha + counts.sum(axis=0)

# Nueva distribución Dirichlet posterior
theta_posterior = dirichlet(alpha_post).rvs(size=1000)

# Calcular el promedio de las muestras de la posterior
theta_mean = theta_posterior.mean(axis=0)

# Imprimir las probabilidades posteriores
print("Probabilidades posteriores (media):", theta_mean)



