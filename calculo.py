"""
calculo.py — Motor de cálculo simbólico y numérico.

Todas las derivadas se obtienen automáticamente mediante
diferenciación simbólica con SymPy (NO se escriben a mano).
Luego se convierten a funciones NumPy rápidas con lambdify.
"""

import numpy as np
import sympy as sp
from scipy.optimize import minimize, fsolve

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


def clasificar_punto(H):
    """Clasifica un punto crítico usando la Hessiana."""
    det = np.linalg.det(H)
    if det > 0 and H[0, 0] < 0:
        return "Máximo local"
    elif det > 0 and H[0, 0] > 0:
        return "Mínimo local"
    elif det < 0:
        return "Punto de silla"
    return "Inconcluso"


def encontrar_puntos_criticos(xi, yi, wi, margen=3.0, n_grid=15,
                               tol_grad=1e-8, tol_dedup=1e-3):
    """
    Encuentra TODOS los puntos críticos de Q(x,y) usando multi-start con fsolve.
    Genera n_grid×n_grid puntos iniciales, resuelve ∇Q=0 desde cada uno,
    filtra los no convergidos y deduplica.
    """
    x_min, x_max = xi.min() - margen, xi.max() + margen
    y_min, y_max = yi.min() - margen, yi.max() + margen

    x_starts = np.linspace(x_min, x_max, n_grid)
    y_starts = np.linspace(y_min, y_max, n_grid)

    candidatos = []

    for xs in x_starts:
        for ys in y_starts:
            try:
                sol, info, ier, _ = fsolve(
                    lambda p: gradiente_Q(p[0], p[1], xi, yi, wi),
                    [xs, ys], full_output=True
                )
                if ier != 1:
                    continue
                xc, yc = sol
                grad = gradiente_Q(xc, yc, xi, yi, wi)
                if np.sqrt(grad[0]**2 + grad[1]**2) > tol_grad:
                    continue
                if xc < x_min or xc > x_max or yc < y_min or yc > y_max:
                    continue
                candidatos.append((xc, yc))
            except Exception:
                continue

    if not candidatos:
        return []

    unicos = [candidatos[0]]
    for xc, yc in candidatos[1:]:
        es_duplicado = False
        for xu, yu in unicos:
            if np.sqrt((xc - xu)**2 + (yc - yu)**2) < tol_dedup:
                es_duplicado = True
                break
        if not es_duplicado:
            unicos.append((xc, yc))

    puntos = []
    for xc, yc in unicos:
        q_val = Q(xc, yc, xi, yi, wi)
        H = hessiana_Q(xc, yc, xi, yi, wi)
        det_h = np.linalg.det(H)
        eigs = np.linalg.eigvals(H)
        tipo = clasificar_punto(H)
        puntos.append({
            "x": xc, "y": yc, "Q": q_val,
            "det_H": det_h, "Qxx": H[0, 0], "Qyy": H[1, 1], "Qxy": H[0, 1],
            "eigenvalues": eigs, "tipo": tipo,
        })

    puntos.sort(key=lambda p: p["Q"], reverse=True)
    return puntos


def optimizar_antena(xi, yi, wi):
    """
    Encuentra la ubicación óptima (MÁXIMO GLOBAL) de la antena.
    Usa multi-start para encontrar todos los puntos críticos y selecciona
    el máximo con mayor Q(x,y).
    """
    x0_init = np.average(xi, weights=wi)
    y0_init = np.average(yi, weights=wi)

    todos_criticos = encontrar_puntos_criticos(xi, yi, wi)

    maximos = [p for p in todos_criticos if p["tipo"] == "Máximo local"]

    if maximos:
        mejor = maximos[0]
        x_opt, y_opt = mejor["x"], mejor["y"]
    else:
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
        "todos_los_puntos_criticos": todos_criticos,
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
