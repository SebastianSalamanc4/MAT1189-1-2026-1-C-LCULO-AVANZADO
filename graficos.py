"""
graficos.py — Todas las visualizaciones (Plotly y Folium).

Funciones puras que reciben datos y retornan figuras,
sin lógica de Streamlit (excepto st_folium para renderizar mapas).
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium

from calculo import Q, gradiente_Q


# ============================================================
# MAPAS FOLIUM
# ============================================================

def crear_mapa_input(sectores, lat_ref, lon_ref):
    """Crea el mapa interactivo para marcar sectores."""
    m = folium.Map(
        location=[lat_ref, lon_ref],
        zoom_start=13,
        tiles="OpenStreetMap"
    )
    for idx, s in enumerate(sectores):
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=max(5, s["hab"] / 2000),
            color='blue', fill=True, fill_color='blue', fill_opacity=0.6,
            popup=f"<b>{s['nombre']}</b><br>{s['hab']:,} hab.",
            tooltip=f"{s['nombre']} ({s['hab']:,})"
        ).add_to(m)
        folium.Marker(
            location=[s["lat"], s["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;color:#1a1a2e;'
                     f'background:rgba(255,255,255,0.8);padding:1px 4px;'
                     f'border-radius:3px;white-space:nowrap;">'
                     f'{idx+1}. {s["nombre"]}</div>'
            )
        ).add_to(m)
    return m


def crear_mapa_resultado(sectores, lat_opt, lon_opt, Q_opt, lat_ref, lon_ref):
    """Crea el mapa con la ubicación óptima de la antena."""
    m = folium.Map(
        location=[lat_ref, lon_ref],
        zoom_start=13,
        tiles="OpenStreetMap"
    )
    for s in sectores:
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=max(5, s["hab"] / 2000),
            color='blue', fill=True, fill_color='blue', fill_opacity=0.6,
            popup=f"<b>{s['nombre']}</b><br>{s['hab']:,} hab.",
            tooltip=s['nombre']
        ).add_to(m)

    folium.Marker(
        location=[lat_opt, lon_opt],
        popup=f"<b>ANTENA ÓPTIMA</b><br>Q = {Q_opt:,.2f}<br>({lat_opt:.4f}, {lon_opt:.4f})",
        tooltip="Ubicación Óptima",
        icon=folium.Icon(color='red', icon='signal', prefix='fa')
    ).add_to(m)

    for s in sectores:
        folium.PolyLine(
            locations=[[s["lat"], s["lon"]], [lat_opt, lon_opt]],
            color='red', weight=1, opacity=0.3, dash_array='5'
        ).add_to(m)
    return m


# ============================================================
# GRÁFICOS PLOTLY
# ============================================================

def fig_superficie_3d(x_grid, y_grid, Z, xi, yi, wi, nombres, x_opt, y_opt, Q_opt, n):
    """Superficie 3D de Q(x,y)."""
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x_grid, y=y_grid, z=Z, colorscale='Viridis',
        colorbar=dict(title="Q(x,y)"), opacity=0.9
    ))
    fig.add_trace(go.Scatter3d(
        x=xi, y=yi,
        z=[Q(xi[i], yi[i], xi, yi, wi) for i in range(n)],
        mode='markers+text', marker=dict(size=5, color='blue'),
        text=nombres, textposition='top center',
        textfont=dict(size=8), name="Sectores"
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_opt], y=[y_opt], z=[Q_opt],
        mode='markers+text', marker=dict(size=8, color='red', symbol='diamond'),
        text=["ÓPTIMO"], textposition='top center', name="Antena Óptima"
    ))
    fig.update_layout(
        scene=dict(xaxis_title="x (km)", yaxis_title="y (km)", zaxis_title="Q(x,y)"),
        height=550, margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def fig_mapa_calor(x_grid, y_grid, Z, xi, yi, nombres, x_opt, y_opt):
    """Mapa de calor con curvas de nivel."""
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x_grid, y=y_grid, z=Z, colorscale='Viridis',
        colorbar=dict(title="Q(x,y)"),
        contours=dict(showlabels=True, labelfont=dict(size=10, color='white'))
    ))
    fig.add_trace(go.Scatter(
        x=xi, y=yi, mode='markers+text',
        marker=dict(size=10, color='blue', line=dict(width=1, color='white')),
        text=nombres, textposition='top center',
        textfont=dict(size=9, color='white'), name="Sectores"
    ))
    fig.add_trace(go.Scatter(
        x=[x_opt], y=[y_opt], mode='markers+text',
        marker=dict(size=15, color='red', symbol='star',
                    line=dict(width=2, color='white')),
        text=["ÓPTIMO"], textposition='top center',
        textfont=dict(size=11, color='red'), name="Antena Óptima"
    ))
    fig.update_layout(
        xaxis_title="x (km)", yaxis_title="y (km)",
        height=550, margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def fig_campo_gradiente(x_grid, y_grid, Z, xi, yi, wi, x_opt, y_opt):
    """Campo vectorial del gradiente con contorno de fondo."""
    x_min, x_max = x_grid.min(), x_grid.max()
    y_min, y_max = y_grid.min(), y_grid.max()
    res_grad = 15
    x_g = np.linspace(x_min, x_max, res_grad)
    y_g = np.linspace(y_min, y_max, res_grad)
    Xg, Yg = np.meshgrid(x_g, y_g)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x_grid, y=y_grid, z=Z, colorscale='Viridis',
        opacity=0.5, showscale=False, contours=dict(showlabels=False)
    ))

    scale = (x_max - x_min) / res_grad * 0.4
    for ii in range(Xg.shape[0]):
        for jj in range(Xg.shape[1]):
            grad = gradiente_Q(Xg[ii, jj], Yg[ii, jj], xi, yi, wi)
            norm = np.sqrt(grad[0]**2 + grad[1]**2)
            if norm > 1e-10:
                ux, uy = grad[0]/norm, grad[1]/norm
                fig.add_annotation(
                    x=Xg[ii,jj] + ux*scale, y=Yg[ii,jj] + uy*scale,
                    ax=Xg[ii,jj], ay=Yg[ii,jj],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                    arrowcolor="rgba(255,100,0,0.7)"
                )

    fig.add_trace(go.Scatter(
        x=[x_opt], y=[y_opt], mode='markers',
        marker=dict(size=12, color='red', symbol='star'), name="Punto Crítico"
    ))
    fig.update_layout(
        xaxis_title="x (km)", yaxis_title="y (km)",
        height=500, margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def fig_trayectoria_gradiente(x_grid, y_grid, Z, xi, yi, wi, x0, y0):
    """Trayectoria de ascenso por gradiente desde el centroide."""
    tray_x, tray_y = [x0], [y0]
    px_c, py_c = x0, y0
    for _ in range(500):
        grad = gradiente_Q(px_c, py_c, xi, yi, wi)
        px_c += 0.01 * grad[0]
        py_c += 0.01 * grad[1]
        tray_x.append(px_c)
        tray_y.append(py_c)
        if np.sqrt(grad[0]**2 + grad[1]**2) < 1e-6:
            break

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x_grid, y=y_grid, z=Z, colorscale='Viridis',
        opacity=0.5, showscale=False
    ))
    fig.add_trace(go.Scatter(
        x=tray_x, y=tray_y, mode='lines+markers',
        marker=dict(size=3, color='orange'),
        line=dict(color='orange', width=2), name="Trayectoria"
    ))
    fig.add_trace(go.Scatter(
        x=[tray_x[0]], y=[tray_y[0]], mode='markers',
        marker=dict(size=10, color='green'), name="Inicio"
    ))
    fig.add_trace(go.Scatter(
        x=[tray_x[-1]], y=[tray_y[-1]], mode='markers',
        marker=dict(size=12, color='red', symbol='star'), name="Óptimo"
    ))
    fig.update_layout(
        xaxis_title="x (km)", yaxis_title="y (km)",
        height=350, margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def fig_hessiana_heatmap(H_opt):
    """Visualización de la matriz Hessiana como heatmap."""
    fig = px.imshow(
        H_opt, labels=dict(x="", y="", color="Valor"),
        x=["∂²Q/∂x²", "∂²Q/∂x∂y"],
        y=["∂²Q/∂x²", "∂²Q/∂x∂y"],
        color_continuous_scale="RdBu_r", text_auto=".4f",
        title="Matriz Hessiana"
    )
    fig.update_layout(height=300)
    return fig


def fig_cobertura_barras(df_cob):
    """Gráfico de barras de contribución por sector."""
    fig = px.bar(
        df_cob, x="Sector", y="% del Total",
        color="Distancia (km)", color_continuous_scale="RdYlGn_r",
        title="Contribución por sector"
    )
    fig.update_layout(height=400)
    return fig
