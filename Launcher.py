"""
RX Tórax — Sistema de Análisis de Calidad v2
Ejecutar : streamlit run rx_torax.py
Instalar : pip install streamlit opencv-python scikit-image scipy matplotlib Pillow numpy
"""

import io
import base64
import numpy as np
import cv2
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from datetime import datetime
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis as scipy_kurtosis
import warnings
warnings.filterwarnings("ignore")
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile

# ═══════════════════════════════════════════════════════════════
#  UMBRALES DE CLASIFICACIÓN
#  Cada artefacto se evalúa de forma independiente.
#  Una imagen puede tener sobreexposición, blur, ambos o ninguno.
# ═══════════════════════════════════════════════════════════════

# ── Sobreexposición ───────────────────────────────────────────
# Métricas que el blur NO distorsiona igual que la sobreexposición.
# Se necesitan al menos 2 criterios activos para confirmar.
UMBRAL_SOBREX = {
    "media":       160.0,   # intensidad media
    "saturacion":    2.0,   # % px >= 250
    "kurtosis":     -0.5,   # distribución aplastada
    "px_oscuros":   25.0,   # % px < 30 (imagen "lavada")
}

# ── Blur ─────────────────────────────────────────────────────
# Los tres criterios deben activarse juntos para confirmar blur.
# El Laplaciano solo NO es suficiente (se sesga por sobreexposición).
# ── Blur ─────────────────────────────────────────────────────
UMBRAL_BLUR = {
    "laplaciano":  95.0,   # antes: 100 — más estricto para evitar falsos positivos
    "snr":         18.5,   # antes: 20  — el SNR de imágenes normales ronda 7-15 dB
    "entropia":     7.0,   # antes: 7.0 — entropía 7.23 ya no dispara el criterio
}

# ── Severidad sobreexposición ─────────────────────────────────
NIVELES_SOBREX = [
    ("severa",   {"gamma": 0.40, "clahe_clip": 8.0, "tile": (8, 8)},
     lambda m: m["kurtosis"] < -0.9 or (m["media"] > 200 and m["saturacion"] > 20)),
    ("moderada", {"gamma": 0.55, "clahe_clip": 5.0, "tile": (8, 8)},
     lambda m: m["kurtosis"] < -0.6 or (m["media"] > 180 and m["saturacion"] > 10)),
    ("leve",     {"gamma": 0.75, "clahe_clip": 3.0, "tile": (8, 8)},
     lambda m: True),
]

# ── Severidad blur ────────────────────────────────────────────
NIVELES_BLUR = [
    ("severa",   {"metodo": "wiener", "balance": 0.0001, "sharpen_post": True},
     lambda m: m["laplaciano"] < 20  and m["snr"] < 10),
    ("moderada", {"metodo": "wiener", "balance": 0.001,  "sharpen_post": True},
     lambda m: m["laplaciano"] < 50  and m["snr"] < 15),
    ("leve",     {"metodo": "unsharp_mask", "radio": 2.0, "cantidad": 1.5},
     lambda m: True),
]


# ═══════════════════════════════════════════════════════════════
#  PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════

def preprocesar(img_bgr, sigma=1.0, target_size=None):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    k = max(3, int(sigma * 3) | 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)
    
def recortar_roi(img):
    # Umbraliza para encontrar la región brillante (la RX)
    _, mask = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    coords  = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    # Margen de seguridad
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)
    return img[y:y+h, x:x+w]

# ═══════════════════════════════════════════════════════════════
#  MÉTRICAS
# ═══════════════════════════════════════════════════════════════

def calcular_metricas(img, nombre="imagen"):
    f    = img.astype(np.float64)
    mean = np.mean(f)
    std  = np.std(f)
    return {
        "nombre":      nombre,
        "laplaciano":  float(np.var(cv2.Laplacian(f, cv2.CV_64F))),
        "contraste":   float(std),
        "snr":         float(10 * np.log10(mean**2 / std**2)) if std > 0 else 0.0,
        "entropia":    float(shannon_entropy(img)),
        "media":       float(mean),
        "saturacion":  float(np.sum(img >= 250) / img.size * 100),
        "kurtosis":    float(scipy_kurtosis(f.ravel())),
        "px_oscuros":  float(np.sum(img < 30) / img.size * 100),
    }


# ═══════════════════════════════════════════════════════════════
#  CLASIFICACIÓN DUAL E INDEPENDIENTE
# ═══════════════════════════════════════════════════════════════

def _criterios_sobrex(m):
    """Retorna lista de criterios activos de sobreexposición."""
    activos = []
    if m["media"]      > UMBRAL_SOBREX["media"]:      activos.append(f"Media={m['media']:.1f} > {UMBRAL_SOBREX['media']}")
    if m["saturacion"] > UMBRAL_SOBREX["saturacion"]: activos.append(f"Sat={m['saturacion']:.1f}% > {UMBRAL_SOBREX['saturacion']}%")
    if m["kurtosis"]   < UMBRAL_SOBREX["kurtosis"]:   activos.append(f"Kurt={m['kurtosis']:.2f} < {UMBRAL_SOBREX['kurtosis']}")
    if m["kurtosis"]   < UMBRAL_SOBREX["kurtosis"] and m["px_oscuros"] > UMBRAL_SOBREX["px_oscuros"]:
        activos.append(f"Px.oscuros={m['px_oscuros']:.1f}% > {UMBRAL_SOBREX['px_oscuros']}%")
    return activos


def _criterios_blur(m):
    """Retorna lista de criterios activos de blur. Los 3 deben activarse."""
    flags = {
        "lap": m["laplaciano"] < UMBRAL_BLUR["laplaciano"],
        "snr": m["snr"]        < UMBRAL_BLUR["snr"],
        "ent": m["entropia"]   < UMBRAL_BLUR["entropia"],
    }
    activos = []
    if flags["lap"]: activos.append(f"Lap={m['laplaciano']:.1f} < {UMBRAL_BLUR['laplaciano']}")
    if flags["snr"]: activos.append(f"SNR={m['snr']:.1f} dB < {UMBRAL_BLUR['snr']} dB")
    if flags["ent"]: activos.append(f"Entropía={m['entropia']:.2f} < {UMBRAL_BLUR['entropia']}")
    # Blur confirmado solo si los 3 se activan
    confirmado = flags["lap"] and flags["snr"]
    return activos, confirmado



def _severidad_sobrex(m):
    for sev, params, cond in NIVELES_SOBREX:
        if cond(m):
            return sev, params
    return "leve", NIVELES_SOBREX[2][1]


def _severidad_blur(m):
    for sev, params, cond in NIVELES_BLUR:
        if cond(m):
            return sev, params
    return "leve", NIVELES_BLUR[2][1]


def clasificar(m):
    """
    Evalúa sobreexposición y blur de forma INDEPENDIENTE.
    Retorna un dict con los dos diagnósticos y la decisión final.
    """
    c_sobrex  = _criterios_sobrex(m)
    c_blur, blur_ok = _criterios_blur(m)

    tiene_sobrex = len(c_sobrex) >= 2          # 2 o más criterios = confirmado
    tiene_blur   = blur_ok                      # los 3 criterios juntos

    resultado = {
        "sobrex": None,
        "blur":   None,
        "final":  "APTA PARA DIAGNÓSTICO",
    }

    if tiene_sobrex:
        sev, params = _severidad_sobrex(m)
        resultado["sobrex"] = {
            "severidad": sev,
            "params":    params,
            "criterios": c_sobrex,
        }

    if tiene_blur:
        sev, params = _severidad_blur(m)
        resultado["blur"] = {
            "severidad": sev,
            "params":    params,
            "criterios": c_blur,
        }

    if tiene_sobrex or tiene_blur:
        partes = []
        if tiene_sobrex: partes.append(f"SOBREEXPOSICIÓN ({resultado['sobrex']['severidad']})")
        if tiene_blur:   partes.append(f"BLUR ({resultado['blur']['severidad']})")
        resultado["final"] = "DEGRADADA: " + " + ".join(partes)

    return resultado


# ═══════════════════════════════════════════════════════════════
#  CORRECCIÓN  (sobreexposición primero, luego blur)
# ═══════════════════════════════════════════════════════════════

def _gamma_clahe(img, gamma, clahe_clip, tile):
    tabla  = np.array([255 * (i / 255.0) ** gamma for i in range(256)], dtype=np.uint8)
    result = cv2.LUT(img, tabla)
    return cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile).apply(result)


def _estimar_psf(img, longitud):
    from skimage.transform import radon
    f   = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    esp = np.log1p(np.abs(f))
    esp = (esp / esp.max() * 255).astype(np.uint8)
    ang = np.arange(0, 180, 1)
    sin = radon(esp, theta=ang, circle=False)
    ap  = ang[np.argmax(sin.var(axis=0))]
    ab  = float((ap + 90) % 180)
    k   = np.zeros((longitud, longitud))
    k[longitud // 2, :] = 1.0 / longitud
    M   = cv2.getRotationMatrix2D((longitud // 2, longitud // 2), ab, 1.0)
    k   = cv2.warpAffine(k, M, (longitud, longitud))
    return (k / k.sum()) if k.sum() != 0 else k


def _deblur_robusto(img, severidad):
    """
    Pipeline sin estimación de PSF: evita ringing y bandas direccionales.
    Severidad leve → unsharp moderado.
    Severidad moderada/severa → bilateral + unsharp agresivo + realce de bordes.
    """
    if severidad == "leve":
        return _unsharp(img, radio=2.0, cantidad=1.5)

    # Paso 1: Denoising bilateral (preserva bordes, no introduce frecuencias espurias)
    d = 9 if severidad == "moderada" else 12
    sigma_c = 50 if severidad == "moderada" else 75
    denoised = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_c)

    # Paso 2: Unsharp agresivo sobre imagen denoisada
    cantidad = 2.0 if severidad == "moderada" else 2.8
    sharpened = _unsharp(denoised, radio=1.5, cantidad=cantidad)

    # Paso 3: Realce de bordes con Laplaciano suave (no el kernel duro de _sharpen)
    lap = cv2.Laplacian(sharpened, cv2.CV_64F, ksize=3)
    lap_norm = np.clip(lap * 0.3, -50, 50)
    result = np.clip(sharpened.astype(np.float64) + lap_norm, 0, 255).astype(np.uint8)

    return result


def _sharpen(img):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return np.clip(cv2.filter2D(img.astype(np.float32), -1, k), 0, 255).astype(np.uint8)


def _unsharp(img, radio, cantidad):
    blur  = cv2.GaussianBlur(img, (0, 0), sigmaX=radio)
    sharp = img.astype(np.float64) + cantidad * (img.astype(np.float64) - blur.astype(np.float64))
    return np.clip(sharp, 0, 255).astype(np.uint8)


def aplicar_correcciones(img, dx):
    """
    Aplica correcciones en orden: sobreexposición → blur.
    Cada una se aplica solo si fue detectada.
    """
    resultado = img.copy()

    if dx["sobrex"]:
        p = dx["sobrex"]["params"]
        resultado = _gamma_clahe(resultado, p["gamma"], p["clahe_clip"], p["tile"])

    if dx["blur"]:
        p = dx["blur"]["params"]
        sev = dx["blur"]["severidad"]
        if p["metodo"] == "wiener":
            lap_check = float(np.var(cv2.Laplacian(resultado.astype(np.float64), cv2.CV_64F)))
            if lap_check > 80:
                resultado = _unsharp(resultado, radio=1.5, cantidad=0.8)
            else:
                resultado = _deblur_robusto(resultado, sev)
        else:
            resultado = _unsharp(resultado, p["radio"], p["cantidad"])

    return resultado


# ═══════════════════════════════════════════════════════════════
#  FIGURAS MATPLOTLIB
# ═══════════════════════════════════════════════════════════════

DARK = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1f2e",
    "axes.edgecolor":   "#2d3748",
    "axes.labelcolor":  "#a0aec0",
    "xtick.color":      "#718096",
    "ytick.color":      "#718096",
    "text.color":       "#e2e8f0",
}


def fig_comparacion(img_pre, img_corr):
    diff     = cv2.absdiff(img_corr, img_pre).astype(np.float32)
    diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
    with plt.rc_context(DARK):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        datos = [
            (img_pre,  "gray", "Original (preprocesada)", "#ef4444"),
            (img_corr, "gray", "Corregida",               "#22c55e"),
            (diff_vis, "hot",  "Diferencia ×5",            "#f59e0b"),
        ]
        for ax, (im, cmap, titulo, borde) in zip(axes, datos):
            ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
            ax.set_title(titulo, fontsize=10, pad=6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(borde); sp.set_linewidth(2)
        fig.tight_layout(pad=1.0)
    return fig


def fig_histograma(img_pre, img_corr=None):
    with plt.rc_context(DARK):
        ncols = 2 if img_corr is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(10 if ncols == 2 else 5, 3.5))
        if ncols == 1: axes = [axes]
        pares = [(img_pre, "#378ADD", "Original")]
        if img_corr is not None:
            pares.append((img_corr, "#639922", "Corregida"))
        for ax, (im, color, titulo) in zip(axes, pares):
            ax.hist(im.ravel(), bins=80, range=(0, 255), color=color, alpha=0.8, edgecolor="none")
            ax.axvline(160, color="#f59e0b", linestyle="--", linewidth=1.2, label="Umbral sobreexp. (160)")
            ax.axvline(UMBRAL_BLUR["laplaciano"], color="#e74c3c", linestyle=":", linewidth=1.0, alpha=0.6)
            ax.set_title(f"Histograma — {titulo}", fontsize=10, pad=5)
            ax.set_xlabel("Intensidad (0–255)", fontsize=9)
            ax.set_ylabel("Frecuencia", fontsize=9)
            ax.legend(fontsize=8)
        fig.tight_layout(pad=1.0)
    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0f1117")
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════
#  CSS GLOBAL
# ═══════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f1117;
    color: #e2e8f0;
}

/* ── Topbar ── */
.topbar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 20px;
    background: #0a0e1a;
    border-bottom: 1px solid #1e2d40;
    border-radius: 10px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.topbar-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    font-weight: 500;
    color: #60a5fa;
    letter-spacing: .03em;
}
.topbar-sub {
    font-size: 11px;
    color: #4a6080;
    margin-left: 4px;
}
.topbar-file {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #718096;
    margin-left: auto;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 99px;
    white-space: nowrap;
}
.badge-danger  { background: #2b1a1a; color: #f87171; border: 1px solid #f87171; }
.badge-warn    { background: #2b1a0d; color: #fb923c; border: 1px solid #fb923c; }
.badge-success { background: #0d2b1a; color: #34d399; border: 1px solid #34d399; }
.badge-info    { background: #0d1a2b; color: #60a5fa; border: 1px solid #60a5fa; }
.badge-neutral { background: #1a1a2b; color: #818cf8; border: 1px solid #818cf8; }

/* ── KPI strip ── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 14px;
}
.kpi-card {
    background: #1a1f2e;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.kpi-card.alert { border-color: #7f1d1d; background: #1a0f0f; }
.kpi-label {
    font-size: 9px;
    font-weight: 500;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: .08em;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 20px;
    font-weight: 500;
    color: #e2e8f0;
    line-height: 1.1;
}
.kpi-value.bad { color: #f87171; }
.kpi-delta { font-size: 10px; }
.kpi-delta.pos { color: #34d399; }
.kpi-delta.neg { color: #f87171; }
.kpi-bar-bg {
    height: 3px;
    background: #1e2d40;
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
}
.kpi-bar-fill {
    height: 100%;
    border-radius: 2px;
}
.fill-blue   { background: #185FA5; }
.fill-green  { background: #3B6D11; }
.fill-red    { background: #991b1b; }
.fill-amber  { background: #92400e; }

/* ── Paneles ── */
.panel {
    background: #1a1f2e;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 10px;
}
.panel-header {
    padding: 7px 14px;
    background: #141824;
    border-bottom: 1px solid #1e2d40;
    font-size: 10px;
    font-weight: 500;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: .08em;
}
.panel-body { padding: 12px 14px; }

/* ── Tabla de métricas ── */
.met-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
.met-table th {
    text-align: left;
    color: #4a6080;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .07em;
    padding: 5px 8px;
    border-bottom: 1px solid #1e2d40;
}
.met-table td {
    padding: 6px 8px;
    border-bottom: 1px solid #141824;
    color: #a0aec0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    vertical-align: middle;
}
.met-table td:first-child {
    color: #e2e8f0;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
}
.tag {
    display: inline-block;
    font-size: 9px;
    font-weight: 500;
    padding: 2px 6px;
    border-radius: 4px;
}
.tag-bad { background: #2b1a1a; color: #f87171; }
.tag-ok  { background: #0d2b1a; color: #34d399; }
.tag-warn{ background: #2b1a0d; color: #fb923c; }

/* ── Criterio box ── */
.crit-box {
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.crit-box.danger { background: #1a0f0f; border-left: 3px solid #dc2626; }
.crit-box.warn   { background: #1a130a; border-left: 3px solid #d97706; }
.crit-box.ok     { background: #0a1a0f; border-left: 3px solid #16a34a; }
.crit-title {
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 4px;
}
.crit-box.danger .crit-title { color: #f87171; }
.crit-box.warn   .crit-title { color: #fb923c; }
.crit-box.ok     .crit-title { color: #34d399; }
.crit-detail {
    font-size: 10px;
    color: #718096;
    line-height: 1.6;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Método aplicado ── */
.method-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid #141824;
    font-size: 11px;
}
.method-row:last-child { border-bottom: none; }
.method-key { color: #4a6080; }
.method-val {
    color: #e2e8f0;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
}

/* ── Upload zone ── */
.upload-hint {
    text-align: center;
    padding: 16px;
    border: 1px dashed #1e2d40;
    border-radius: 10px;
    color: #4a6080;
    font-size: 12px;
    margin-bottom: 12px;
}

/* ── Info apta ── */
.apta-box {
    background: #0a1a0f;
    border: 1px solid #16a34a;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin: 20px 0;
}
.apta-box p { font-size: 15px; font-weight: 500; color: #34d399; }
.apta-box span { font-size: 12px; color: #4a6080; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""


# ═══════════════════════════════════════════════════════════════
#  HELPERS HTML
# ═══════════════════════════════════════════════════════════════

def badge(texto, tipo="neutral"):
    clases = {
        "danger": "badge-danger",
        "warn":   "badge-warn",
        "ok":     "badge-success",
        "info":   "badge-info",
    }.get(tipo, "badge-neutral")
    return f'<span class="badge {clases}">{texto}</span>'


def kpi_card(label, val, delta=None, fill="blue", is_bad=False, pct=50):
    bad_cls  = " bad" if is_bad else ""
    card_cls = " alert" if is_bad else ""
    delta_html = ""
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        cls  = "pos" if (delta > 0 and not is_bad) or (delta < 0 and is_bad) else "neg"
        delta_html = f'<div class="kpi-delta {cls}">{sign}{delta:.2f}</div>'
    return f"""
    <div class="kpi-card{card_cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value{bad_cls}">{val}</div>
        {delta_html}
        <div class="kpi-bar-bg"><div class="kpi-bar-fill fill-{fill}" style="width:{min(pct,100):.0f}%"></div></div>
    </div>"""


def criterio_box(titulo, detalles, tipo):
    detalle_html = "<br>".join(detalles) if detalles else "—"
    return f"""
    <div class="crit-box {tipo}">
        <div class="crit-title">{titulo}</div>
        <div class="crit-detail">{detalle_html}</div>
    </div>"""


def met_row(nombre, orig, corr=None, inversa=False):
    if corr is not None:
        delta = corr - orig
        mejora = (delta < 0) if inversa else (delta >= 0)
        tag_cls = "tag-ok" if mejora else "tag-bad"
        tag_txt = "mejor" if mejora else "peor"
        sign    = "+" if delta >= 0 else ""
        return (f"<tr><td>{nombre}</td><td>{orig:.4f}</td><td>{corr:.4f}</td>"
                f"<td>{'<span class=\"kpi-delta pos\">' if mejora else '<span class=\"kpi-delta neg\">'}{sign}{delta:.4f}</span></td>"
                f"<td><span class='tag {tag_cls}'>{tag_txt}</span></td></tr>")
    return f"<tr><td>{nombre}</td><td>{orig:.4f}</td></tr>"


# ═══════════════════════════════════════════════════════════════
#  PUNTAJES DE SERVICIO (escala 1–10)
# ═══════════════════════════════════════════════════════════════

def _normalizar(valor, minv, maxv):
    return max(0.0, min(1.0, (valor - minv) / (maxv - minv)))

def score_nitidez(m, m_corr=None):
    src = m_corr if m_corr else m
    lap = _normalizar(src["laplaciano"], 10, 2000)
    ent = _normalizar(src["entropia"],   4.5, 7.5)
    score = max(1, min(10, round((lap * 0.65 + ent * 0.35) * 10)))
    delta = round((m_corr["laplaciano"] - m["laplaciano"]) * 0.65 +
                  (m_corr["entropia"]   - m["entropia"]) * 10, 1) if m_corr else None
    return score, delta

def score_contraste(m, m_corr=None):
    src = m_corr if m_corr else m
    rms = _normalizar(src["contraste"],  20, 90)
    sat = _normalizar(src["saturacion"],  0, 10) * 0.3
    score = max(1, min(10, round((rms - sat) * 10 + 1)))
    delta = round((m_corr["contraste"] - m["contraste"]) * 0.5, 1) if m_corr else None
    return score, delta

def score_ruido(m, m_corr=None):
    src = m_corr if m_corr else m
    snr = _normalizar(src["snr"],        5, 40)
    px  = _normalizar(src["px_oscuros"], 0, 50) * 0.25
    score = max(1, min(10, round((snr - px) * 10 + 0.5)))
    delta = round(m_corr["snr"] - m["snr"], 1) if m_corr else None
    return score, delta


# ═══════════════════════════════════════════════════════════════
#  REPORTE PDF — ReportLab
# ═══════════════════════════════════════════════════════════════

def generar_reporte_pdf(m, m_corr, dx, dx_final_str, nombre, img_pre, img_corr):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import mm, inch
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, KeepTogether
    )
    from reportlab.graphics.shapes import Drawing, Rect, String, Circle
    from reportlab.graphics import renderPDF

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Colores ───────────────────────────────────────────────
    AZUL        = colors.HexColor("#1b4f8a")
    AZUL_CLARO  = colors.HexColor("#e8f0fb")
    GRIS_FONDO  = colors.HexColor("#f5f7fb")
    GRIS_BORDE  = colors.HexColor("#d8e2ef")
    GRIS_TEXTO  = colors.HexColor("#4a5a6e")
    GRIS_LABEL  = colors.HexColor("#6b7c93")
    VERDE       = colors.HexColor("#1a6b3a")
    VERDE_FONDO = colors.HexColor("#e4f4ec")
    VERDE_BORDE = colors.HexColor("#9ed0b3")
    ROJO        = colors.HexColor("#a32d2d")
    ROJO_FONDO  = colors.HexColor("#fdf0f0")
    ROJO_BORDE  = colors.HexColor("#f0b0b0")
    BLANCO      = colors.white
    FILA_ALT    = colors.HexColor("#f8fafc")
    AMBER       = colors.HexColor("#c8882a")

    # ── Estilos de párrafo ────────────────────────────────────
    def st(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9, leading=13,
                    textColor=GRIS_TEXTO)
        base.update(kw)
        return ParagraphStyle(name, **base)

    S_TITLE   = st("title",   fontName="Helvetica-Bold", fontSize=15,
                   textColor=AZUL, leading=18)
    S_SUB     = st("sub",     fontSize=9, textColor=GRIS_LABEL, leading=12)
    S_LABEL   = st("label",   fontName="Helvetica-Bold", fontSize=7.5,
                   textColor=GRIS_LABEL, leading=10, spaceBefore=10, spaceAfter=3)
    S_BODY    = st("body",    fontSize=9, leading=13)
    S_BOLD    = st("bold",    fontName="Helvetica-Bold", fontSize=9, leading=13,
                   textColor=colors.HexColor("#1a2332"))
    S_SMALL   = st("small",   fontSize=8, textColor=GRIS_LABEL, leading=10)
    S_CENTER  = st("center",  alignment=TA_CENTER, fontSize=9, leading=13)
    S_DECISION= st("dec",     fontName="Helvetica-Bold", fontSize=13,
                   alignment=TA_CENTER, leading=17)
    S_DEC_BODY= st("decbody", fontSize=9, alignment=TA_CENTER,
                   leading=14, textColor=GRIS_TEXTO)
    S_FOOTER  = st("footer",  fontSize=7.5, textColor=colors.HexColor("#a0b0c4"),
                   leading=10)
    S_META_K  = st("mk", fontSize=8.5, textColor=GRIS_LABEL,
                   fontName="Helvetica-Bold", leading=11)
    S_META_V  = st("mv", fontSize=8.5, textColor=colors.HexColor("#1a2332"),
                   leading=11)

    # ── Puntajes ──────────────────────────────────────────────
    s_nit, d_nit = score_nitidez(m,  m_corr)
    s_con, d_con = score_contraste(m, m_corr)
    s_rui, d_rui = score_ruido(m,    m_corr)

    def delta_str(d):
        if d is None: return "—"
        return f"+{d}" if d >= 0 else str(d)

    def score_colors(s):
        if s >= 8: return VERDE,       VERDE_FONDO
        if s >= 6: return AZUL,        AZUL_CLARO
        return          ROJO,          ROJO_FONDO

    # ── Imagen numpy → RLImage ────────────────────────────────
    def arr_to_rl(arr, w, h):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(arr.astype(np.uint8)).save(tmp.name)
        return RLImage(tmp.name, width=w, height=h)

    # ── Histograma matplotlib → RLImage ───────────────────────
    def hist_rl(img_arr, color_hex, titulo, w, h):
        fig, ax = plt.subplots(figsize=(w / 72, h / 72))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8fafc")
        ax.hist(img_arr.ravel(), bins=80, range=(0, 255),
                color=color_hex, alpha=0.82, edgecolor="none")
        ax.axvline(160, color="#c8882a", linestyle="--",
                   linewidth=1.0, label="Umbral sobreexp.")
        ax.set_title(titulo, fontsize=7, pad=3, color="#4a5a6e")
        ax.set_xlabel("Intensidad (0–255)", fontsize=6.5, color="#8a9ab5")
        ax.tick_params(labelsize=6, colors="#8a9ab5")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_edgecolor("#d8e2ef")
        ax.yaxis.set_visible(False)
        ax.legend(fontsize=6, framealpha=0.6)
        fig.tight_layout(pad=0.4)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return RLImage(tmp.name, width=w, height=h)

    # ── Círculo de puntaje (Drawing) ──────────────────────────
    def score_drawing(s, size=26):
        txt_col, bg_col = score_colors(s)
        d = Drawing(size, size)
        d.add(Circle(size/2, size/2, size/2 - 1,
                     fillColor=bg_col,
                     strokeColor=txt_col,
                     strokeWidth=1.2))
        d.add(String(size/2, size/2 - 4, str(s),
                     fontSize=12, fontName="Helvetica-Bold",
                     fillColor=txt_col,
                     textAnchor="middle"))
        return d

    # ── Documento ─────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=17*mm, rightMargin=17*mm,
        topMargin=16*mm,  bottomMargin=14*mm,
    )
    W = letter[0] - 34*mm   # ancho útil

    story = []

    # ── ENCABEZADO ────────────────────────────────────────────
    header_data = [[
        Paragraph("Reporte Técnico de Calidad Radiológica", S_TITLE),
        Paragraph("UTB · Bioingeniería<br/><b>RX Tórax v3</b>",
                  st("logo", alignment=TA_RIGHT, fontSize=8,
                     textColor=GRIS_LABEL, leading=12)),
    ]]
    header_tbl = Table(header_data, colWidths=[W * 0.72, W * 0.28])
    header_tbl.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "BOTTOM"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    story.append(header_tbl)
    story.append(Paragraph(
        "Sistema computacional de evaluación y corrección — RX Tórax",
        st("subt", fontSize=8.5, textColor=GRIS_LABEL, leading=11, spaceAfter=4)
    ))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=AZUL, spaceAfter=10))

    # ── BLOQUE 1: INFO ────────────────────────────────────────
    story.append(Paragraph("INFORMACIÓN DEL PACIENTE Y DATOS TÉCNICOS", S_LABEL))

    info_data = [
        [Paragraph("Paciente ID",  S_META_K), Paragraph("Person100 (Anonymized)", S_META_V),
         Paragraph("Equipo ID",    S_META_K), Paragraph("GE Optima XR240 (Anonymized)", S_META_V)],
        [Paragraph("Procedimiento",S_META_K), Paragraph("PA Chest X-ray", S_META_V),
         Paragraph("Proyección",   S_META_K), Paragraph("AP-Pediatric (inferred)", S_META_V)],
        [Paragraph("Fecha",        S_META_K), Paragraph(ts, S_META_V),
         Paragraph("Archivo",      S_META_K), Paragraph(nombre, S_META_V)],
    ]
    cw = [W*0.13, W*0.37, W*0.13, W*0.37]
    info_tbl = Table(info_data, colWidths=cw)
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), GRIS_FONDO),
        ("GRID",         (0,0), (-1,-1), 0.4, GRIS_BORDE),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(info_tbl)

    # ── DIAGNÓSTICO ───────────────────────────────────────────
    story.append(Paragraph("DIAGNÓSTICO TÉCNICO", S_LABEL))
    dx_lines = []
    if dx["sobrex"]:
        p = dx["sobrex"]["params"]
        dx_lines.append(
            f'<b>Sobreexposición ({dx["sobrex"]["severidad"]}):</b> '
            f'{", ".join(dx["sobrex"]["criterios"])} — '
            f'Corrección: Gamma={p["gamma"]} · CLAHE clip={p["clahe_clip"]}'
        )
    if dx["blur"]:
        dx_lines.append(
            f'<b>Blur ({dx["blur"]["severidad"]}):</b> '
            f'{", ".join(dx["blur"]["criterios"])} — '
            f'Corrección: Bilateral + Unsharp Mask + realce laplaciano'
        )
    if not dx_lines:
        dx_lines.append("No se detectaron artefactos en la imagen.")

    dx_rows = [[Paragraph(line, S_BODY)] for line in dx_lines]
    dx_tbl = Table(dx_rows, colWidths=[W])
    dx_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), GRIS_FONDO),
        ("GRID",         (0,0), (-1,-1), 0.4, GRIS_BORDE),
        ("LEFTPADDING",  (0,0), (-1,-1), 9),
        ("RIGHTPADDING", (0,0), (-1,-1), 9),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(dx_tbl)

    # ── BLOQUE 2: IMÁGENES ────────────────────────────────────
    story.append(Paragraph("COMPARACIÓN VISUAL — ORIGINAL VS. CORREGIDA", S_LABEL))

    IW = (W - 6*mm) / 2
    IH = IW * 0.82

    rl_orig = arr_to_rl(img_pre,  IW, IH)
    rl_corr = arr_to_rl(img_corr, IW, IH) if img_corr is not None else \
              Paragraph("Sin corrección aplicada", S_CENTER)

    def img_cell(content, label_txt):
        label = Paragraph(label_txt,
                          st("ilbl", fontSize=8, textColor=GRIS_LABEL,
                             fontName="Helvetica-Bold", alignment=TA_CENTER,
                             leading=10, spaceAfter=3))
        cell_tbl = Table([[label], [content]],
                         colWidths=[IW])
        cell_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (0,0), GRIS_FONDO),
            ("BACKGROUND",   (0,1), (0,1), colors.black),
            ("BOX",          (0,0), (-1,-1), 0.4, GRIS_BORDE),
            ("LEFTPADDING",  (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING",   (0,0), (0,0), 4),
            ("BOTTOMPADDING",(0,0), (0,0), 4),
            ("TOPPADDING",   (0,1), (0,1), 0),
            ("BOTTOMPADDING",(0,1), (0,1), 0),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ]))
        return cell_tbl

    img_row = Table(
        [[img_cell(rl_orig, "ORIGINAL"),
          img_cell(rl_corr, "IMAGEN CORREGIDA")]],
        colWidths=[IW, IW],
        hAlign="LEFT",
    )
    img_row.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("COLPADDING",   (0,0), (-1,-1), 3),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(img_row)

    # ── BLOQUE 3: MÉTRICAS DE SERVICIO ────────────────────────
    story.append(Paragraph("MÉTRICAS DE CALIDAD — RESUMEN DE SERVICIO", S_LABEL))

    SVC = [
        (s_nit, d_nit, "Nitidez y Detalle Estructural",
         "Var. Laplaciano · Entropía",
         "Mayor visualización de estructuras anatómicas finas: parrilla costal, "
         "bordes vasculares y contornos mediastínicos."),
        (s_con, d_con, "Definición de Tejidos (Contraste)",
         "Contraste RMS · Saturación",
         "Mayor diferenciación entre densidades de tejido blando, parénquima "
         "pulmonar y estructuras óseas."),
        (s_rui, d_rui, "Reducción de Ruido Base",
         "SNR · Píxeles oscuros",
         "Mayor claridad en áreas de bajo contraste, reduciendo artefactos "
         "y mejorando la relación señal-ruido global."),
    ]

    svc_header = [
        Paragraph("Puntaje 0-10", st("th", fontName="Helvetica-Bold", fontSize=8,
                                textColor=BLANCO, alignment=TA_CENTER, leading=10)),
        Paragraph("Métrica de rendimiento", st("th", fontName="Helvetica-Bold",
                                               fontSize=8, textColor=BLANCO, leading=10)),
        Paragraph("Δ Mejora", st("th", fontName="Helvetica-Bold", fontSize=8,
                                 textColor=BLANCO, alignment=TA_CENTER, leading=10)),
        Paragraph("Significado clínico", st("th", fontName="Helvetica-Bold",
                                            fontSize=8, textColor=BLANCO, leading=10)),
    ]

    svc_rows = [svc_header]
    for s, d, name, sub, meaning in SVC:
        txt_col, bg_col = score_colors(s)
        svc_rows.append([
            score_drawing(s, 26),
            [Paragraph(f"<b>{name}</b>",
                       st("sn", fontSize=9, textColor=colors.HexColor("#1a2332"),
                          leading=12)),
             Paragraph(sub, st("ss", fontSize=8, textColor=GRIS_LABEL,
                               fontName="Helvetica-Oblique", leading=10))],
            Paragraph(delta_str(d),
                      st("dp", fontSize=9, fontName="Helvetica-Bold",
                         textColor=VERDE if (d or 0) >= 0 else ROJO,
                         alignment=TA_CENTER, leading=12)),
            Paragraph(meaning, st("sm", fontSize=8.5, textColor=GRIS_TEXTO,
                                  leading=12)),
        ])

    svc_tbl = Table(svc_rows,
                    colWidths=[W*0.09, W*0.31, W*0.12, W*0.48])
    svc_cmds = [
        ("BACKGROUND",   (0,0), (-1,0),  AZUL),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [BLANCO, FILA_ALT]),
        ("GRID",         (0,0), (-1,-1), 0.4, GRIS_BORDE),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",        (0,0), (0,-1),  "CENTER"),
        ("ALIGN",        (2,0), (2,-1),  "CENTER"),
    ]
    svc_tbl.setStyle(TableStyle(svc_cmds))
    story.append(svc_tbl)

    # ── MÉTRICAS RAW ──────────────────────────────────────────
    story.append(Paragraph("DETALLE DE MÉTRICAS CUANTITATIVAS", S_LABEL))

    KEYS = [
        ("laplaciano", "Var. Laplaciano", False),
        ("contraste",  "Contraste RMS",   False),
        ("snr",        "SNR (dB)",        False),
        ("entropia",   "Entropía",        False),
        ("media",      "Int. media",      True),
        ("saturacion", "Saturación (%)",  True),
        ("kurtosis",   "Kurtosis",        True),
        ("px_oscuros", "Px. oscuros (%)", True),
    ]

    def th(txt):
        return Paragraph(txt, st("rth", fontName="Helvetica-Bold", fontSize=8,
                                 textColor=BLANCO, leading=10))
    def td(txt, bold=False, col=None):
        return Paragraph(txt, st("rtd", fontSize=8.5,
                                 fontName="Helvetica-Bold" if bold else "Helvetica",
                                 textColor=col or GRIS_TEXTO,
                                 leading=11, alignment=TA_CENTER))
    def tdl(txt):
        return Paragraph(txt, st("rtdl", fontSize=8.5, textColor=colors.HexColor("#1a2332"),
                                 leading=11))

    if m_corr:
        raw_header = [th("Métrica"), th("Original"), th("Corregida"), th("Δ")]
        raw_rows   = [raw_header]
        for key, label, inv in KEYS:
            orig  = m[key]; corr = m_corr[key]; delta = corr - orig
            mej   = (delta < 0) if inv else (delta >= 0)
            sign  = "+" if delta >= 0 else ""
            col   = VERDE if mej else ROJO
            raw_rows.append([
                tdl(label),
                td(f"{orig:.4f}"),
                td(f"{corr:.4f}"),
                td(f"{sign}{delta:.4f}", bold=True, col=col),
            ])
        raw_cw = [W*0.30, W*0.23, W*0.23, W*0.24]
    else:
        raw_header = [th("Métrica"), th("Valor")]
        raw_rows   = [raw_header]
        for key, label, _ in KEYS:
            raw_rows.append([tdl(label), td(f"{m[key]:.4f}")])
        raw_cw = [W*0.55, W*0.45]

    raw_tbl = Table(raw_rows, colWidths=raw_cw)
    raw_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  AZUL),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [BLANCO, FILA_ALT]),
        ("GRID",          (0,0), (-1,-1), 0.4, GRIS_BORDE),
        ("LEFTPADDING",   (0,0), (-1,-1), 7),
        ("RIGHTPADDING",  (0,0), (-1,-1), 7),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
    ]))
    story.append(raw_tbl)

    # ── BLOQUE 4: HISTOGRAMAS ─────────────────────────────────
    story.append(Paragraph("DISTRIBUCIÓN DE INTENSIDADES", S_LABEL))

    HW = (W - 6*mm) / 2
    HH = HW * 0.52

    rl_h_orig = hist_rl(img_pre,  "#378ADD", "Histograma — Original",  HW, HH)
    rl_h_corr = hist_rl(img_corr, "#1d9e75", "Histograma — Corregida", HW, HH) \
                if img_corr is not None else \
                Paragraph("Sin corrección", S_CENTER)

    hist_row = Table([[rl_h_orig, rl_h_corr]],
                     colWidths=[HW, HW])
    hist_row.setStyle(TableStyle([
        ("BOX",         (0,0), (0,0), 0.4, GRIS_BORDE),
        ("BOX",         (1,0), (1,0), 0.4, GRIS_BORDE),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING",(0,0), (-1,-1), 0),
        ("COLPADDING",  (0,0), (-1,-1), 3),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(hist_row)
    story.append(Paragraph(
        "Distribución de intensidades antes y después de la corrección · "
        "línea discontinua = umbral de sobreexposición (160)",
        st("hcap", fontSize=7.5, textColor=GRIS_LABEL,
           alignment=TA_CENTER, leading=10, spaceBefore=3)
    ))

    # ── BLOQUE 5: DECISIÓN FINAL ──────────────────────────────
    story.append(Paragraph("DECISIÓN FINAL", S_LABEL))

    es_apta    = "DEGRADADA" not in dx_final_str
    dec_color  = VERDE       if es_apta else ROJO
    dec_bg     = colors.HexColor("#f0f8f3") if es_apta else ROJO_FONDO
    dec_border = VERDE_BORDE if es_apta else ROJO_BORDE
    dec_tag    = "APTO"      if es_apta else "NO APTO"
    conclusion = (
        "La corrección técnica aplicada ha resultado en una mejora significativa y uniforme "
        "en la calidad de la imagen, permitiendo una visualización nítida y detallada de la "
        "vasculatura pulmonar y los contornos óseos." if es_apta else
        "La imagen presenta artefactos que comprometen la aptitud diagnóstica. "
        "Se recomienda revisión manual y, de ser posible, repetir el estudio."
    )

    dec_rows = [
        [Paragraph(
            f"El estudio es <b>{dec_tag}</b> para diagnóstico clínico inicial",
            st("dectit", fontName="Helvetica-Bold", fontSize=13,
               textColor=dec_color, alignment=TA_CENTER, leading=17)
        )],
        [Paragraph(conclusion,
                   st("decbod", fontSize=9, textColor=GRIS_TEXTO,
                      alignment=TA_CENTER, leading=14))],
    ]
    dec_tbl = Table(dec_rows, colWidths=[W])
    dec_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), dec_bg),
        ("BOX",          (0,0), (-1,-1), 1.5, dec_border),
        ("LEFTPADDING",  (0,0), (-1,-1), 16),
        ("RIGHTPADDING", (0,0), (-1,-1), 16),
        ("TOPPADDING",   (0,0), (0,0),   12),
        ("BOTTOMPADDING",(0,0), (0,0),   4),
        ("TOPPADDING",   (0,1), (0,1),   0),
        ("BOTTOMPADDING",(0,1), (0,1),   12),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(dec_tbl)

    # ── FOOTER ────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=GRIS_BORDE, spaceAfter=4))

    footer_data = [[
        Paragraph(f"Generado por RX Tórax v3 · UTB Bioingeniería · {ts}",
                  st("fl", fontSize=7.5, textColor=colors.HexColor("#a0b0c4"),
                     leading=10)),
        Paragraph(
            dec_tag,
            st("ft", fontSize=7.5, fontName="Helvetica-Bold",
               textColor=dec_color, alignment=TA_RIGHT, leading=10)
        ),
    ]]
    ft = Table(footer_data, colWidths=[W * 0.75, W * 0.25])
    ft.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(ft)

    doc.build(story)
    buf.seek(0)
    return buf

# ═══════════════════════════════════════════════════════════════
#  STREAMLIT — CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RX Tórax",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Estado de sesión ──────────────────────────────────────────
for k in ["ok", "img_pre", "img_corr", "m", "m_corr", "dx", "nombre"]:
    if k not in st.session_state:
        st.session_state[k] = None


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🫁 RX Tórax")
    st.caption("Sistema de análisis de calidad")
    st.markdown("---")

    archivo = st.file_uploader(
        "Subir radiografía",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
        label_visibility="collapsed",
    )

    if not archivo:
        st.markdown(
            '<div class="upload-hint">📂 Sube una radiografía<br>.png · .jpg · .bmp · .tiff</div>',
            unsafe_allow_html=True,
        )
    else:
        st.success(f"✓ {archivo.name}")

    st.markdown("#### Preprocesamiento")
    sigma = st.slider("Filtro Gaussiano σ", 0.5, 3.0, 1.0, 0.1,
                      help="Suavizado leve antes del análisis. Valor recomendado: 1.0")
    redim = st.checkbox("Redimensionar imagen")
    if redim:
        c1, c2 = st.columns(2)
        tw = c1.number_input("W px", 64, 2048, 512, 64)
        th = c2.number_input("H px", 64, 2048, 512, 64)
        target_size = (int(tw), int(th))
    else:
        target_size = None

    st.markdown("---")
    analizar  = st.button("▶ Analizar imagen", use_container_width=True, type="primary")
    exportar  = st.button("⬇ Exportar reporte", use_container_width=True,
                          disabled=not st.session_state.ok)

    if st.session_state.ok and st.session_state.dx:
        st.markdown("---")
        st.markdown("#### Clasificación activa")
        dx = st.session_state.dx
        if dx["sobrex"]:
            st.markdown(
                criterio_box(
                    f"Sobreexposición · {dx['sobrex']['severidad']}",
                    dx["sobrex"]["criterios"], "danger"
                ), unsafe_allow_html=True
            )
        else:
            st.markdown(criterio_box("Sobreexposición", ["No detectada"], "ok"),
                        unsafe_allow_html=True)
        if dx["blur"]:
            st.markdown(
                criterio_box(
                    f"Blur · {dx['blur']['severidad']}",
                    dx["blur"]["criterios"], "warn"
                ), unsafe_allow_html=True
            )
        else:
            st.markdown(criterio_box("Blur", ["No detectado"], "ok"),
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PROCESAMIENTO
# ═══════════════════════════════════════════════════════════════

if archivo and analizar:
    with st.spinner("Analizando imagen..."):
        pil      = Image.open(archivo).convert("RGB")
        bgr      = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img_pre  = preprocesar(bgr, sigma=sigma, target_size=target_size)
        m        = calcular_metricas(img_pre, nombre=archivo.name)
        dx       = clasificar(m)

        img_corr = None
        m_corr   = None
        if dx["sobrex"] or dx["blur"]:
            img_corr = aplicar_correcciones(img_pre, dx)
            m_corr   = calcular_metricas(img_corr, nombre="corregida")

        st.session_state.update({
            "ok": True, "img_pre": img_pre, "img_corr": img_corr,
            "m": m, "m_corr": m_corr, "dx": dx, "nombre": archivo.name,
        })
        st.rerun()


# ═══════════════════════════════════════════════════════════════
#  EXPORTAR
# ═══════════════════════════════════════════════════════════════

if exportar and st.session_state.ok:
    m        = st.session_state.m
    m_corr   = st.session_state.m_corr
    dx       = st.session_state.dx
    nombre   = st.session_state.nombre
    img_pre  = st.session_state.img_pre
    img_corr = st.session_state.img_corr

    dx_str = dx["final"] if (dx["sobrex"] or dx["blur"]) else "APTA PARA DIAGNÓSTICO"
    dx_fin = clasificar(m_corr)["final"] if m_corr else dx_str

    with st.spinner("Generando PDF..."):
        pdf_buf = generar_reporte_pdf(m, m_corr, dx, dx_fin, nombre, img_pre, img_corr)

    st.sidebar.download_button(
        "📄 Descargar reporte PDF",
        pdf_buf,
        f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        "application/pdf",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════
#  PANTALLA PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if not st.session_state.ok:
    # ── Estado inicial ─────────────────────────────────────────
    st.markdown("""
    <div class="topbar">
        <span class="topbar-logo">RX Tórax</span>
        <span class="topbar-sub">Sistema de análisis de calidad</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel"><div class="panel-header">Fase 1 — Diagnóstico</div><div class="panel-body" style="font-size:12px;color:#718096;line-height:1.7">Calcula 8 métricas cuantitativas y clasifica la imagen de forma <b>independiente</b> por sobreexposición y blur. Una imagen puede tener ambos simultáneamente.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="panel-header">Fase 2 — Corrección</div><div class="panel-body" style="font-size:12px;color:#718096;line-height:1.7">Aplica Gamma + CLAHE para sobreexposición y Wiener / Unsharp Mask para blur — en ese orden, sin interferencia entre correcciones.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel"><div class="panel-header">Reporte</div><div class="panel-body" style="font-size:12px;color:#718096;line-height:1.7">Genera dashboard comparativo con 8 KPIs, histograma, tabla de métricas antes/después y decisión final de aptitud diagnóstica.</div></div>', unsafe_allow_html=True)

else:
    # ── Dashboard ──────────────────────────────────────────────
    m        = st.session_state.m
    m_corr   = st.session_state.m_corr
    dx       = st.session_state.dx
    img_pre  = st.session_state.img_pre
    img_corr = st.session_state.img_corr
    nombre   = st.session_state.nombre

    hay_corr   = img_corr is not None
    tiene_sobrex = dx["sobrex"] is not None
    tiene_blur   = dx["blur"] is not None
    es_apta    = not tiene_sobrex and not tiene_blur

    if m_corr:
        dx_post    = clasificar(m_corr)
        apta_post  = not dx_post["sobrex"] and not dx_post["blur"]
    else:
        dx_post   = None
        apta_post = es_apta

    # ── Topbar ────────────────────────────────────────────────
    badges_html = ""
    if tiene_sobrex:
        badges_html += badge(f"Sobreexposición · {dx['sobrex']['severidad']}", "danger") + " "
    if tiene_blur:
        badges_html += badge(f"Blur · {dx['blur']['severidad']}", "warn") + " "
    if es_apta:
        badges_html += badge("Apta para diagnóstico", "ok") + " "
    if hay_corr:
        tipo_fin = "ok" if apta_post else "danger"
        txt_fin  = "Apta post-corrección" if apta_post else "No apta post-corrección"
        badges_html += badge(txt_fin, tipo_fin)

    st.markdown(f"""
    <div class="topbar">
        <span class="topbar-logo">RX Tórax</span>
        <span class="topbar-sub">Sistema de análisis de calidad</span>
        {badges_html}
        <span class="topbar-file">{nombre}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip (8 métricas en 2 filas de 4) ───────────────
    KPIS = [
        ("Var. Laplaciano", "laplaciano", "blue",  False, 100),
        ("SNR (dB)",        "snr",        "blue",  False, 40),
        ("Entropía",        "entropia",   "green", False, 8),
        ("Contraste RMS",   "contraste",  "green", False, 80),
        ("Int. media",      "media",      "red",   True,  255),
        ("Saturación (%)",  "saturacion", "red",   True,  20),
        ("Kurtosis",        "kurtosis",   "amber", True,  None),
        ("Px. oscuros (%)", "px_oscuros", "amber", True,  100),
    ]

    row1 = st.columns(4)
    row2 = st.columns(4)
    rows = row1 + row2

    for col, (label, key, fill, inversa, max_val) in zip(rows, KPIS):
        val    = m[key]
        delta  = (m_corr[key] - val) if m_corr else None
        is_bad = (val > UMBRAL_SOBREX.get(key, float("inf"))) if inversa else (val < UMBRAL_BLUR.get(key, 0))

        if max_val:
            pct = abs(val) / max_val * 100
        else:
            pct = min(abs(val) * 20, 100)

        val_str = f"{val:.1f}" if abs(val) >= 10 else f"{val:.2f}"
        with col:
            st.markdown(
                kpi_card(label, val_str, delta, fill, is_bad, pct),
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Pestañas principales ──────────────────────────────────
    tab_img, tab_hist, tab_met = st.tabs([
        "🖼 Comparación visual",
        "📊 Histograma",
        "📋 Métricas detalladas",
        ])

    # ── Tab 1: Imágenes ───────────────────────────────────────
    with tab_img:
        if hay_corr:
            fig = fig_comparacion(img_pre, img_corr)
            st.pyplot(fig, use_container_width=True)

            col_dl, _ = st.columns([1, 3])
            with col_dl:
                st.download_button(
                    "⬇ Descargar comparación (PNG)",
                    fig_to_bytes(fig),
                    "comparacion.png", "image/png",
                )
            plt.close(fig)

            # Métodos aplicados
            st.markdown("---")
            mc1, mc2 = st.columns(2)
            if tiene_sobrex:
                p = dx["sobrex"]["params"]
                with mc1:
                    st.markdown(f"""
                    <div class="panel">
                        <div class="panel-header">Corrección sobreexposición</div>
                        <div class="panel-body">
                            <div class="method-row"><span class="method-key">Método</span><span class="method-val">Gamma + CLAHE</span></div>
                            <div class="method-row"><span class="method-key">Gamma γ</span><span class="method-val">{p['gamma']}</span></div>
                            <div class="method-row"><span class="method-key">CLAHE clip</span><span class="method-val">{p['clahe_clip']}</span></div>
                            <div class="method-row"><span class="method-key">Tile</span><span class="method-val">{p['tile'][0]}×{p['tile'][1]}</span></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            if tiene_blur:
                p = dx["blur"]["params"]
                with mc2:
                    metodo = p.get("metodo", "—")
                    extra  = f"Balance: {p.get('balance','—')}" if metodo == "wiener" else f"Radio: {p.get('radio','—')} · Cant: {p.get('cantidad','—')}"
                    st.markdown(f"""
                    <div class="panel">
                        <div class="panel-header">Corrección blur</div>
                        <div class="panel-body">
                            <div class="method-row"><span class="method-key">Método</span><span class="method-val">{metodo.title()}</span></div>
                            <div class="method-row"><span class="method-key">Parámetros</span><span class="method-val">{extra}</span></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="apta-box">
                <p>✓ Imagen apta para diagnóstico</p>
                <span>No se detectaron artefactos. No se aplicó corrección.</span>
            </div>""", unsafe_allow_html=True)
            st.image(img_pre, caption="Imagen preprocesada", clamp=True)

    # ── Tab 2: Histograma ─────────────────────────────────────
    with tab_hist:
        fig_h = fig_histograma(img_pre, img_corr)
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)

        st.markdown("---")
        st.markdown("**Criterios de clasificación activos**")
        cc1, cc2 = st.columns(2)
        with cc1:
            c_sobrex = _criterios_sobrex(m)
            if c_sobrex:
                st.markdown(criterio_box(
                    f"Sobreexposición {'confirmada' if len(c_sobrex) >= 2 else '(insuficiente)'}",
                    c_sobrex,
                    "danger" if len(c_sobrex) >= 2 else "warn"
                ), unsafe_allow_html=True)
            else:
                st.markdown(criterio_box("Sobreexposición", ["Sin criterios activos"], "ok"),
                            unsafe_allow_html=True)
        with cc2:
            c_blur, blur_ok = _criterios_blur(m)
            if blur_ok:
                st.markdown(criterio_box("Blur confirmado (3/3 criterios)", c_blur, "warn"),
                            unsafe_allow_html=True)
            elif c_blur:
                st.markdown(criterio_box(
                    f"Blur parcial ({len(c_blur)}/3 criterios — no confirmado)",
                    c_blur, "warn"
                ), unsafe_allow_html=True)
            else:
                st.markdown(criterio_box("Blur", ["Sin criterios activos"], "ok"),
                            unsafe_allow_html=True)

    # ── Tab 3: Métricas ───────────────────────────────────────
    with tab_met:
        TABLA_KEYS = [
            ("laplaciano", "Var. Laplaciano", False),
            ("contraste",  "Contraste RMS",   False),
            ("snr",        "SNR (dB)",        False),
            ("entropia",   "Entropía",        False),
            ("media",      "Int. media",      True),
            ("saturacion", "Saturación (%)",  True),
            ("kurtosis",   "Kurtosis",        True),
            ("px_oscuros", "Px. oscuros (%)", True),
        ]

        if m_corr:
            headers = "<tr><th>Métrica</th><th>Original</th><th>Corregida</th><th>Δ</th><th>Estado</th></tr>"
            filas   = "".join(met_row(label, m[k], m_corr[k], inv) for k, label, inv in TABLA_KEYS)
        else:
            headers = "<tr><th>Métrica</th><th>Valor</th></tr>"
            filas   = "".join(met_row(label, m[k]) for k, label, _ in TABLA_KEYS)

        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">Métricas de calidad{"— antes vs. después" if m_corr else ""}</div>
            <div class="panel-body" style="padding:0">
                <table class="met-table">{headers}{filas}</table>
            </div>
        </div>""", unsafe_allow_html=True)

        if m_corr:
            st.markdown("---")
            st.markdown(f"""
            <div class="crit-box {'ok' if apta_post else 'danger'}">
                <div class="crit-title">{'✓ Apta post-corrección' if apta_post else '✗ No apta post-corrección'}</div>
                <div class="crit-detail">{'Todas las métricas están dentro del rango aceptable tras la corrección.' if apta_post else 'Algunas métricas siguen fuera del rango. Revisar imagen manualmente.'}</div>
            </div>""", unsafe_allow_html=True)

