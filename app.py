"""
app.py — Interfaz principal de Streamlit.

Proyecto Final MATE1189 - Cálculo Avanzado
Optimización de la ubicación de una antena de comunicaciones en Temuco.
Universidad Católica de Temuco — 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
from streamlit_folium import st_folium

from datos import (
    TEMUCO_LAT_REF, TEMUCO_LON_REF, BARRIOS_TEMUCO,
    latlon_a_km, km_a_latlon, detectar_barrio,
)
from calculo import (
    Q, gradiente_Q, optimizar_antena, generar_grilla,
    fi_sp, dfi_dx_sp, dfi_dy_sp,
    d2fi_dx2_sp, d2fi_dy2_sp, d2fi_dxdy_sp,
)
from graficos import (
    crear_mapa_input, crear_mapa_resultado,
    fig_superficie_3d, fig_mapa_calor,
    fig_campo_gradiente, fig_trayectoria_gradiente,
    fig_hessiana_heatmap, fig_cobertura_barras,
)

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="Optimización Antena - Temuco",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Optimización de la Ubicación de una Antena de Comunicaciones")
st.markdown("**MATE1189 - Cálculo Avanzado | Universidad Católica de Temuco**")
st.markdown("---")

# ============================================================
# SESSION STATE
# ============================================================

if "sectores" not in st.session_state:
    st.session_state.sectores = []

# ============================================================
# SIDEBAR: Controles de entrada
# ============================================================

st.sidebar.header("⚙️ Configuración")

modo = st.sidebar.radio(
    "Modo de entrada",
    ["🖱️ Clic en el mapa", "📋 Sectores predeterminados"],
    index=0,
)

if modo == "📋 Sectores predeterminados":
    st.sidebar.markdown("*Datos: Censo INE — Barrios de Temuco (coords. OpenStreetMap)*")
    if st.sidebar.button("Cargar sectores de Temuco"):
        st.session_state.sectores = [dict(s) for s in BARRIOS_TEMUCO]
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Borrar todos los puntos", type="secondary"):
    st.session_state.sectores = []
    st.rerun()

# Lista editable de sectores en sidebar
if len(st.session_state.sectores) > 0:
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📍 Sectores marcados ({len(st.session_state.sectores)})")
    indices_a_borrar = []
    for idx, s in enumerate(st.session_state.sectores):
        col_name, col_hab, col_del = st.sidebar.columns([3, 2, 1])
        with col_name:
            nuevo_nombre = st.text_input(
                "n", value=s["nombre"], key=f"nombre_{idx}",
                label_visibility="collapsed",
            )
            st.session_state.sectores[idx]["nombre"] = nuevo_nombre
        with col_hab:
            nuevo_hab = st.number_input(
                "h", value=s["hab"], min_value=100, step=1000,
                key=f"hab_{idx}", label_visibility="collapsed",
            )
            st.session_state.sectores[idx]["hab"] = nuevo_hab
        with col_del:
            if st.button("❌", key=f"del_{idx}"):
                indices_a_borrar.append(idx)

    if indices_a_borrar:
        for idx in sorted(indices_a_borrar, reverse=True):
            st.session_state.sectores.pop(idx)
        st.rerun()

    total_hab = sum(s["hab"] for s in st.session_state.sectores)
    st.sidebar.markdown(f"**Total usuarios: {total_hab:,}**")

# ============================================================
# MAPA INTERACTIVO: Marcar sectores con clic
# ============================================================

st.subheader("🗺️ Haz clic en el mapa para agregar sectores")
st.markdown(
    "Haz clic en cualquier zona de Temuco. La app **detecta automáticamente** "
    "el barrio más cercano y asigna su nombre y habitantes (Censo INE)."
)

mapa_input = crear_mapa_input(
    st.session_state.sectores, TEMUCO_LAT_REF, TEMUCO_LON_REF
)
map_data = st_folium(mapa_input, width=None, height=500, key="mapa_input")

if map_data and map_data.get("last_clicked"):
    click_lat = map_data["last_clicked"]["lat"]
    click_lon = map_data["last_clicked"]["lng"]

    ya_existe = any(
        abs(s["lat"] - click_lat) < 0.0005 and abs(s["lon"] - click_lon) < 0.0005
        for s in st.session_state.sectores
    )
    if not ya_existe:
        nombre_det, hab_det, _ = detectar_barrio(click_lat, click_lon)
        if nombre_det:
            if any(s["nombre"] == nombre_det for s in st.session_state.sectores):
                nombre_det = f"{nombre_det} (2)"
            st.session_state.sectores.append({
                "nombre": nombre_det, "lat": click_lat,
                "lon": click_lon, "hab": hab_det,
            })
        else:
            st.session_state.sectores.append({
                "nombre": f"Punto {len(st.session_state.sectores) + 1}",
                "lat": click_lat, "lon": click_lon, "hab": 5000,
            })
        st.rerun()

# ============================================================
# CÁLCULOS (requiere >= 2 sectores)
# ============================================================

if len(st.session_state.sectores) < 2:
    st.info(
        "📌 Marca al menos **2 sectores** en el mapa para ejecutar la optimización."
    )
    st.stop()

nombres = [s["nombre"] for s in st.session_state.sectores]
lats = np.array([s["lat"] for s in st.session_state.sectores])
lons = np.array([s["lon"] for s in st.session_state.sectores])
wi = np.array([s["hab"] for s in st.session_state.sectores], dtype=float)

coords_km = np.array([latlon_a_km(lat, lon) for lat, lon in zip(lats, lons)])
xi, yi = coords_km[:, 0], coords_km[:, 1]
n_sectores = len(xi)

res = optimizar_antena(xi, yi, wi)
x_opt, y_opt = res["x_opt"], res["y_opt"]
Q_opt = res["Q_opt"]
grad_opt = res["grad"]
H_opt = res["hessiana"]
eigenvalues_opt = res["eigenvalues"]
det_H = res["det_H"]
lat_opt, lon_opt = km_a_latlon(x_opt, y_opt)

x_grid, y_grid, X, Y, Z = generar_grilla(xi, yi, wi)

# ============================================================
# RESULTADO: Mapa con antena óptima
# ============================================================

st.markdown("---")
st.header("📡 Resultado: Ubicación Óptima de la Antena")

col_map, col_info = st.columns([2, 1])

with col_map:
    mapa_res = crear_mapa_resultado(
        st.session_state.sectores, lat_opt, lon_opt, Q_opt,
        TEMUCO_LAT_REF, TEMUCO_LON_REF,
    )
    st_folium(mapa_res, width=None, height=450, key="mapa_resultado")

with col_info:
    st.metric("Latitud óptima", f"{lat_opt:.6f}")
    st.metric("Longitud óptima", f"{lon_opt:.6f}")
    st.metric("Q(x*, y*)", f"{Q_opt:,.2f}")
    st.metric("Sectores", n_sectores)
    st.metric("Total usuarios", f"{int(wi.sum()):,}")
    st.markdown("---")
    for i, nombre in enumerate(nombres):
        dist = np.sqrt((xi[i] - x_opt)**2 + (yi[i] - y_opt)**2)
        cob_i = wi[i] / (1 + dist**2)
        st.markdown(f"**{nombre}**: {dist:.2f} km → Cob: {cob_i:,.1f}")

# ============================================================
# TABS DE ANÁLISIS
# ============================================================

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📐 Modelo Matemático",
    "📊 Superficie y Mapa de Calor",
    "🔍 Gradiente y Puntos Críticos",
    "🧮 Matriz Hessiana",
    "📈 Análisis de Resultados",
])

# --- TAB 1: MODELO MATEMÁTICO ---
with tab1:
    st.header("Modelo Matemático")

    st.subheader("5. Función de Calidad de Cobertura")
    st.markdown("Cada término de la suma se define simbólicamente en SymPy:")
    st.code(
        "x, y, x_i, y_i, w_i = sp.symbols('x y x_i y_i w_i')\n"
        "f_i = w_i / (1 + (x - x_i)**2 + (y - y_i)**2)",
        language="python",
    )
    st.latex(
        r"Q(x, y) = \sum_{i=1}^{n} \frac{w_i}"
        r"{1 + (x - x_i)^2 + (y - y_i)^2}"
    )
    st.markdown("""
    Donde:
    - $(x, y)$: ubicación de la antena
    - $(x_i, y_i)$: ubicación del sector $i$
    - $w_i$: cantidad de usuarios del sector $i$
    - $Q(x,y)$: calidad total de cobertura
    """)

    st.markdown("---")
    st.subheader("6.2. Derivadas Parciales (calculadas por SymPy)")
    st.markdown(
        "Las derivadas parciales se obtienen **automáticamente** "
        "mediante diferenciación simbólica:"
    )
    st.code(
        "dfi_dx = sp.diff(f_i, x)   # SymPy calcula ∂fᵢ/∂x\n"
        "dfi_dy = sp.diff(f_i, y)   # SymPy calcula ∂fᵢ/∂y",
        language="python",
    )
    st.markdown("**Resultado de SymPy para** $\\partial f_i / \\partial x$**:**")
    st.latex(sp.latex(sp.simplify(dfi_dx_sp)))
    st.markdown("**Resultado de SymPy para** $\\partial f_i / \\partial y$**:**")
    st.latex(sp.latex(sp.simplify(dfi_dy_sp)))

    st.latex(
        r"\frac{\partial Q}{\partial x} = \sum_{i=1}^{n} "
        r"\frac{\partial f_i}{\partial x}, \qquad "
        r"\frac{\partial Q}{\partial y} = \sum_{i=1}^{n} "
        r"\frac{\partial f_i}{\partial y}"
    )

    st.markdown("---")
    st.subheader("6.3. Gradiente")
    st.latex(
        r"\nabla Q(x,y) = \left(\frac{\partial Q}{\partial x},\, "
        r"\frac{\partial Q}{\partial y}\right)"
    )
    st.markdown(
        "El gradiente apunta en la dirección de **máximo crecimiento** de $Q$. "
        "En el punto óptimo, $\\nabla Q = \\mathbf{0}$."
    )

    st.markdown("---")
    st.subheader("6.4. Puntos Críticos")
    st.markdown(
        "Se resuelve el sistema $\\nabla Q(x,y) = \\mathbf{0}$ "
        "numéricamente con el método BFGS (scipy.optimize):"
    )
    st.code(
        "resultado = minimize(\n"
        "    fun  = -Q,           # negativo porque minimize busca mínimos\n"
        "    x0   = centroide,    # punto inicial\n"
        "    jac  = -∇Q,          # gradiente (calculado por SymPy)\n"
        "    method = 'BFGS'\n"
        ")",
        language="python",
    )

    st.markdown("---")
    st.subheader("6.5. Matriz Hessiana (calculada por SymPy)")
    st.code(
        "d2fi_dx2  = sp.diff(f_i, x, 2)     # ∂²fᵢ/∂x²\n"
        "d2fi_dy2  = sp.diff(f_i, y, 2)     # ∂²fᵢ/∂y²\n"
        "d2fi_dxdy = sp.diff(f_i, x, y)     # ∂²fᵢ/∂x∂y",
        language="python",
    )
    st.latex(
        r"H_Q = \begin{pmatrix} \dfrac{\partial^2 Q}{\partial x^2} & "
        r"\dfrac{\partial^2 Q}{\partial x \partial y} \\[8pt] "
        r"\dfrac{\partial^2 Q}{\partial y \partial x} & "
        r"\dfrac{\partial^2 Q}{\partial y^2} \end{pmatrix}"
    )
    st.markdown("**Resultado de SymPy para** $\\partial^2 f_i / \\partial x^2$**:**")
    st.latex(sp.latex(sp.simplify(d2fi_dx2_sp)))
    st.markdown("**Resultado de SymPy para** $\\partial^2 f_i / \\partial y^2$**:**")
    st.latex(sp.latex(sp.simplify(d2fi_dy2_sp)))
    st.markdown(
        "**Resultado de SymPy para** $\\partial^2 f_i / \\partial x \\partial y$**:**"
    )
    st.latex(sp.latex(sp.simplify(d2fi_dxdy_sp)))

    st.markdown("---")
    st.subheader("Conversión a funciones numéricas")
    st.markdown(
        "Las expresiones simbólicas se convierten a funciones NumPy "
        "rápidas con `sp.lambdify`:"
    )
    st.code(
        "# SymPy → NumPy (vectorizado, rápido)\n"
        "_fi_num     = sp.lambdify((x, y, x_i, y_i, w_i), f_i,    modules='numpy')\n"
        "_dfi_dx_num = sp.lambdify((x, y, x_i, y_i, w_i), dfi_dx, modules='numpy')\n"
        "# ... etc.\n\n"
        "# Luego se suman sobre todos los sectores:\n"
        "def Q(x, y, xi, yi, wi):\n"
        "    return np.sum(_fi_num(x, y, xi, yi, wi))",
        language="python",
    )

# --- TAB 2: SUPERFICIE 3D Y MAPA DE CALOR ---
with tab2:
    st.header("Superficie de Cobertura Q(x, y)")
    col_3d, col_heat = st.columns(2)
    with col_3d:
        st.subheader("Superficie 3D")
        st.plotly_chart(
            fig_superficie_3d(x_grid, y_grid, Z, xi, yi, wi,
                              nombres, x_opt, y_opt, Q_opt, n_sectores)
        )
    with col_heat:
        st.subheader("Mapa de Calor (Curvas de Nivel)")
        st.plotly_chart(
            fig_mapa_calor(x_grid, y_grid, Z, xi, yi, nombres, x_opt, y_opt)
        )

# --- TAB 3: GRADIENTE ---
with tab3:
    st.header("Gradiente y Puntos Críticos")
    col_grad, col_crit = st.columns(2)

    with col_grad:
        st.subheader("Campo Vectorial del Gradiente")
        st.plotly_chart(
            fig_campo_gradiente(x_grid, y_grid, Z, xi, yi, wi, x_opt, y_opt)
        )

    with col_crit:
        st.subheader("Análisis del Punto Crítico")
        st.markdown("**Condición:** $\\nabla Q(x^*, y^*) = \\mathbf{0}$")
        st.markdown(
            f"**Ubicación:** $x^* = {x_opt:.6f}$ km, $y^* = {y_opt:.6f}$ km"
        )
        st.markdown(f"**Coordenadas geográficas:** ({lat_opt:.6f}, {lon_opt:.6f})")
        st.markdown("---")
        st.latex(
            rf"\nabla Q(x^*, y^*) = ({grad_opt[0]:.2e},\; {grad_opt[1]:.2e})"
        )
        norma_grad = np.sqrt(grad_opt[0]**2 + grad_opt[1]**2)
        st.markdown(f"$\\|\\nabla Q\\| = {norma_grad:.2e}$")
        if norma_grad < 1e-4:
            st.success("El gradiente es prácticamente cero → punto crítico confirmado.")
        else:
            st.warning("El gradiente no es exactamente cero.")
        st.metric("Q(x*, y*)", f"{Q_opt:,.2f}")

        st.markdown("---")
        st.subheader("Trayectoria de Ascenso por Gradiente")
        st.plotly_chart(
            fig_trayectoria_gradiente(
                x_grid, y_grid, Z, xi, yi, wi,
                res["x0_init"], res["y0_init"],
            )
        )

# --- TAB 4: HESSIANA ---
with tab4:
    st.header("Matriz Hessiana y Clasificación")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.subheader("Hessiana en el Punto Óptimo")
        st.latex(
            rf"H_Q = \begin{{pmatrix}} {H_opt[0,0]:.4f} & {H_opt[0,1]:.4f} \\"
            rf" {H_opt[1,0]:.4f} & {H_opt[1,1]:.4f} \end{{pmatrix}}"
        )
        st.markdown("---")
        st.markdown(r"""
        **Clasificación:**
        - $\det(H) > 0$ y $Q_{xx} < 0$ → **Máximo local**
        - $\det(H) > 0$ y $Q_{xx} > 0$ → **Mínimo local**
        - $\det(H) < 0$ → **Punto de silla**
        """)

    with col_h2:
        traza = H_opt[0, 0] + H_opt[1, 1]
        st.metric("det(H)", f"{det_H:,.4f}")
        st.metric("tr(H)", f"{traza:,.4f}")
        st.metric("λ₁", f"{eigenvalues_opt[0].real:,.4f}")
        st.metric("λ₂", f"{eigenvalues_opt[1].real:,.4f}")
        st.metric("Q_xx", f"{H_opt[0, 0]:,.4f}")
        st.markdown("---")
        if det_H > 0 and H_opt[0, 0] < 0:
            st.success("**MÁXIMO LOCAL** confirmado: det(H) > 0 y Q_xx < 0")
        elif det_H > 0 and H_opt[0, 0] > 0:
            st.error("Mínimo local detectado")
        elif det_H < 0:
            st.warning("Punto de silla detectado")
        else:
            st.info("Caso inconcluso")
        st.plotly_chart(fig_hessiana_heatmap(H_opt))

# --- TAB 5: ANÁLISIS ---
with tab5:
    st.header("Análisis de Resultados")
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.subheader("Óptimo vs. Centroide Ponderado")
        x_cent = np.average(xi, weights=wi)
        y_cent = np.average(yi, weights=wi)
        Q_cent = Q(x_cent, y_cent, xi, yi, wi)
        lat_cent, lon_cent = km_a_latlon(x_cent, y_cent)

        comparacion = pd.DataFrame({
            "Métrica": ["Latitud", "Longitud", "Q(x,y)"],
            "Óptimo (BFGS)": [
                f"{lat_opt:.6f}", f"{lon_opt:.6f}", f"{Q_opt:,.2f}",
            ],
            "Centroide Ponderado": [
                f"{lat_cent:.6f}", f"{lon_cent:.6f}", f"{Q_cent:,.2f}",
            ],
        })
        st.dataframe(comparacion, hide_index=True)
        mejora = ((Q_opt - Q_cent) / Q_cent) * 100
        st.metric("Mejora respecto al centroide", f"{mejora:+.2f}%")
        st.markdown(
            "La ubicación óptima **no coincide** con el centroide ponderado "
            "porque la función $Q(x,y)$ no es lineal: la señal decae con el "
            "**cuadrado de la distancia**."
        )

    with col_a2:
        st.subheader("Cobertura por Sector")
        cob_data = []
        for i in range(n_sectores):
            dist = np.sqrt((xi[i] - x_opt)**2 + (yi[i] - y_opt)**2)
            cob = wi[i] / (1 + dist**2)
            cob_data.append({
                "Sector": nombres[i],
                "Habitantes": int(wi[i]),
                "Distancia (km)": round(dist, 3),
                "Cobertura": round(cob, 2),
                "% del Total": round(cob / Q_opt * 100, 2),
            })
        df_cob = pd.DataFrame(cob_data).sort_values("% del Total", ascending=False)
        st.dataframe(df_cob, hide_index=True)
        st.plotly_chart(fig_cobertura_barras(df_cob))

    st.markdown("---")
    st.subheader("Interpretación Práctica")
    st.markdown(f"""
    **Resultado:** La antena debe instalarse en **({lat_opt:.6f}, {lon_opt:.6f})**
    para maximizar la cobertura de {n_sectores} sectores con **{int(wi.sum()):,} usuarios**.

    **Q óptimo = {Q_opt:,.2f}**

    1. La ubicación se desplaza hacia sectores con **mayor densidad poblacional**.
    2. Sectores lejanos reciben menor cobertura por el decaimiento cuadrático.
    3. El modelo asume propagación isotrópica (sin obstáculos).
    """)

st.markdown("---")
st.markdown(
    "**MATE1189 - Cálculo Avanzado** | Proyecto Final | "
    "Universidad Católica de Temuco | 2026"
)
