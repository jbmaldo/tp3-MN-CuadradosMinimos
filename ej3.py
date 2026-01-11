from cml_SVD import cml, cml_reg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm

# Traemos los datos
datos_ajuste = np.loadtxt("./datos/ajuste.txt")   # Matriz de nx2 donde n es el tamaño de la muestra. La primera columna corresponde a los x y la segunda a los y
datos_validacion = np.loadtxt("./datos/validacion.txt")

X_ajuste = datos_ajuste[:,0]
X_validacion = datos_validacion[:,0]

y_ajuste = datos_ajuste[:,1]
y_val = datos_validacion[:,1]

len_muestra = len(X_ajuste)

GRADO_MIN = 1
GRADO_MAX = 2 * len_muestra


X_legendre_validacion = np.polynomial.legendre.legvander(X_validacion, GRADO_MAX)
X_legendre_ajuste = np.polynomial.legendre.legvander(X_ajuste, GRADO_MAX)


# # Ejercicio 3a) -------------------------------------------------------------------------
errores_ajuste = []
errores_validacion = []


for i in range(1, GRADO_MAX + 1):
    X_slice_ajuste = X_legendre_ajuste[:,:i]
    X_slice_val = X_legendre_validacion[:,:i]

    beta_pred_ajuste = cml(X_slice_ajuste, y_ajuste)    # Obtenemos el beta que explica a los datos de ajuste

    y_pred_ajuste = X_slice_ajuste@beta_pred_ajuste
    ECM_ajuste = np.mean(np.square(y_ajuste - y_pred_ajuste))   # Vemos qué tanto nos acercamos a los datos de ajuste.
    errores_ajuste.append(ECM_ajuste)

    y_pred_val = X_slice_val@beta_pred_ajuste    # Con el mismo beta del ajuste vemos cuánto nos acercamos a los nuevos datos, que suponemos tienen la misma distribucion
    ECM_val = np.mean(np.square(y_val - y_pred_val))
    errores_validacion.append(ECM_val)

plt.figure(0)
plt.grid()
plt.plot(np.arange(GRADO_MIN,GRADO_MAX+1), errores_ajuste, label="Ajuste")
plt.plot(np.arange(GRADO_MIN,GRADO_MAX+1), errores_validacion, label="Validacion")
plt.ylabel("Error cuadratico medio")
plt.xlabel("Grado")
plt.yscale("log")
plt.title("Errores de validacion vs ajuste")
plt.legend()
plt.savefig("./graficos/ej3a.png")


# Ejercicio 3b) -------------------------------------------------------------------------
# La resolución es similar, solo que esta vez por cada slice vamos a probar con un rango de lambdas. Usamos el mapa de calor, cuanto mas "caliente" un punto, mayor el error cuadrático medio.
 # A partir de acá se puede mostrar cómo el error se dispara
GRADO_MAX_B = 50
GRADO_MIN_B = 10
lambdas = np.linspace(0.01, 0.1, 100)


error_regularizacion = np.empty(shape=(GRADO_MAX_B - GRADO_MIN_B + 1, len(lambdas)))  # error[i][j] -> error tomando polinomio desde grado 1 hasta i con lambda = j


min_ecm = np.inf

for i in range(GRADO_MIN_B, GRADO_MAX_B + 1):
    X_slice_ajuste = X_legendre_ajuste[:,:i]
    X_slice_val = X_legendre_validacion[:,:i]
    

    for j,e in enumerate(lambdas):
        beta_pred_ajuste = cml_reg(X_slice_ajuste, y_ajuste, e)

        y_pred_val = X_slice_val@beta_pred_ajuste
        ECM_val = np.mean(np.square(y_val - y_pred_val))
        if ECM_val < min_ecm:
            
            min_ecm = ECM_val
            min_axis = (i,e)
        error_regularizacion[i - GRADO_MIN_B][j] = ECM_val    # Para este inciso solo nos importa el error de validación


print("El mínimo error obtenido fue " + str(min_ecm) + " con el polinomio de Legendre de grado " + str(min_axis[0]) + " y lambda " + str(min_axis[1]))

# Debido a la cantidad de datos, en las abscisas hay una representación aproximada de las posiciones de lambda en el eje.
plt.figure(1)
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 7
plt.pcolor(error_regularizacion, norm=LogNorm())
indices = np.arange(0, len(lambdas), 3)
plt.xticks(indices, [lambdas[i] for i in indices], rotation=45)
plt.yticks(np.arange(len(error_regularizacion)),np.arange(GRADO_MIN_B, GRADO_MAX_B + 1))
ax = plt.gca()
xfmt = ScalarFormatter()
xfmt.set_powerlimits((0,0))
ax.xaxis.set_major_formatter(xfmt)
cbar = plt.colorbar()
cbar.set_label("ECM")
plt.xlabel(r'$\lambda$')
plt.ylabel("Grado de polinomio de Legendre")
plt.title("Error de validación con regularización")
plt.savefig("./graficos/ej3b.png")