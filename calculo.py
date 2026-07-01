"""
calculo.py — Motor de cálculo simbólico y numérico.

Todas las derivadas se obtienen automáticamente mediante
diferenciación simbólica con SymPy (NO se escriben a mano).
Luego se convierten a funciones NumPy rápidas con lambdify.
"""

import numpy as np
import sympy as sp
from scipy.optimize import minimize

# ============================================================
# CÁLCULO SIMBÓLICO CON SYMPY (Sección 5 y 6 del PDF)
# ============================================================

x_sp, y_sp, xi_sp, yi_sp, wi_sp = sp.symbols('x y x_i y_i w_i')

# Definición simbólica de un término de Q (sección 5)
fi_sp = wi_sp / (1 + (x_sp - xi_sp)**2 + (y_sp - yi_sp)**2)

# Derivadas parciales de primer orden (sección 6.2)
dfi_dx_sp = sp.diff(fi_sp, x_sp)
dfi_dy_sp = sp.diff(fi_sp, y_sp)

# Derivadas parciales de segundo orden para la Hessiana (sección 6.5)
d2fi_dx2_sp  = sp.diff(fi_sp, x_sp, 2)
d2fi_dy2_sp  = sp.diff(fi_sp, y_sp, 2)
d2fi_dxdy_sp = sp.diff(fi_sp, x_sp, y_sp)

# Convertir expresiones simbólicas a funciones numéricas rápidas
_vars = (x_sp, y_sp, xi_sp, yi_sp, wi_sp)
_fi_num     = sp.lambdify(_vars, fi_sp,         modules='numpy')
_dfi_dx_num = sp.lambdify(_vars, dfi_dx_sp,     modules='numpy')
_dfi_dy_num = sp.lambdify(_vars, dfi_dy_sp,     modules='numpy')
_d2fi_dx2   = sp.lambdify(_vars, d2fi_dx2_sp,   modules='numpy')
_d2fi_dy2   = sp.lambdify(_vars, d2fi_dy2_sp,   modules='numpy')
_d2fi_dxdy  = sp.lambdify(_vars, d2fi_dxdy_sp,  modules='numpy')


# ============================================================
# FUNCIONES NUMÉRICAS (sumatorias sobre todos los sectores)
# ============================================================

def Q(x, y, xi, yi, wi):
    """Q(x,y) = Σ wᵢ / (1 + (x-xᵢ)² + (y-yᵢ)²)"""
    return np.sum(_fi_num(x, y, xi, yi, wi))


def Q_grid(X, Y, xi, yi, wi):
    """Evalúa Q en una grilla 2D para gráficos de superficie."""
    Z = np.zeros_like(X)
    for i in range(len(xi)):
        Z += _fi_num(X, Y, xi[i], yi[i], wi[i])
    return Z


def gradiente_Q(x, y, xi, yi, wi):
    """∇Q = (∂Q/∂x, ∂Q/∂y) — derivadas calculadas por SymPy."""
    dQdx = np.sum(_dfi_dx_num(x, y, xi, yi, wi))
    dQdy = np.sum(_dfi_dy_num(x, y, xi, yi, wi))
    return np.array([dQdx, dQdy])


def hessiana_Q(x, y, xi, yi, wi):
    """Matriz Hessiana — segundas derivadas calculadas por SymPy."""
    Qxx = np.sum(_d2fi_dx2(x, y, xi, yi, wi))
    Qyy = np.sum(_d2fi_dy2(x, y, xi, yi, wi))
    Qxy = np.sum(_d2fi_dxdy(x, y, xi, yi, wi))
    return np.array([[Qxx, Qxy], [Qxy, Qyy]])


def optimizar_antena(xi, yi, wi):
    """
    Encuentra la ubicación óptima de la antena maximizando Q(x,y).
    Usa el método BFGS con gradiente analítico (calculado por SymPy).
    Retorna un dict con todos los resultados del análisis.
    """
    x0_init = np.average(xi, weights=wi)
    y0_init = np.average(yi, weights=wi)

    resultado = minimize(
        fun=lambda p: -Q(p[0], p[1], xi, yi, wi),
        x0=[x0_init, y0_init],
        jac=lambda p: -gradiente_Q(p[0], p[1], xi, yi, wi),
        method='BFGS'
    )

    x_opt, y_opt = resultado.x
    Q_opt = Q(x_opt, y_opt, xi, yi, wi)
    grad_opt = gradiente_Q(x_opt, y_opt, xi, yi, wi)
    H_opt = hessiana_Q(x_opt, y_opt, xi, yi, wi)
    eigenvalues = np.linalg.eigvals(H_opt)
    det_H = np.linalg.det(H_opt)

    return {
        "x_opt": x_opt,
        "y_opt": y_opt,
        "Q_opt": Q_opt,
        "grad": grad_opt,
        "hessiana": H_opt,
        "eigenvalues": eigenvalues,
        "det_H": det_H,
        "x0_init": x0_init,
        "y0_init": y0_init,
    }


def generar_grilla(xi, yi, wi, margen=2.0, resolucion=100):
    """Genera la grilla 2D para gráficos de superficie y contorno."""
    x_min, x_max = xi.min() - margen, xi.max() + margen
    y_min, y_max = yi.min() - margen, yi.max() + margen
    x_grid = np.linspace(x_min, x_max, resolucion)
    y_grid = np.linspace(y_min, y_max, resolucion)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = Q_grid(X, Y, xi, yi, wi)
    return x_grid, y_grid, X, Y, Z
