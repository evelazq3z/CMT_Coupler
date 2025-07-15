

"""
CMT – dos guías rectangulares paralelas
=======================================
• Usando el modelo de Marcatili en https://github.com/evelazq3z/Marcatili, se obtienen los valores de neff y β (rad/µm) para c-Si, a-Si:H para el modo TE00 .
• Si no se ha analizado el codigo anterior, se obtiene β, h, q con el Effective-Index-Method (EIM, 2-pasos).

Coeficiente de acoplo (Ida & back):
    κ(s) = 2 h² q e^(-q s) / ( β W (q² + h²) )

Requisitos: instalar las librerias numpy, scipy, matplotlib
          pip install numpy scipy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


# ────────────────────────────────────────────────────────────────
# 1. Solver TE0 para un slab simétrico (altura ó ancho)
# ────────────────────────────────────────────────────────────────
def slab_TE0(n_core, n_clad, thickness, lam):
    """Devuelve β, h y q para el modo TE0 de un slab."""
    k0 = 2 * np.pi / lam

    def F(beta):
        h = np.sqrt(beta**2 - (k0 * n_clad) ** 2)
        q = np.sqrt((k0 * n_core) ** 2 - beta ** 2)
        return np.tan(h * thickness / 2) - q / h

    beta = root_scalar(F,
                       bracket=[k0 * n_clad * 1.001,
                                k0 * n_core * 0.999],
                       xtol=1e-10).root
    h = np.sqrt(beta**2 - (k0 * n_clad) ** 2)
    q = np.sqrt((k0 * n_core) ** 2 - beta ** 2)
    return beta, h, q


# ────────────────────────────────────────────────────────────────
# 2. κ vía Effective-Index-Method (EIM)
# ────────────────────────────────────────────────────────────────
def kappa_rect_EIM(n_core, n_clad, W, H, s, lam):
    """κ, β, h, q obtenidos con el método de índice efectivo."""
    beta_v, _, _ = slab_TE0(n_core, n_clad, H, lam)        # paso vertical
    n_eff_v = beta_v * lam / (2 * np.pi)

    try:
        beta, h, q = slab_TE0(n_eff_v, n_clad, W, lam)     # paso horizontal
    except ValueError:
        return 0.0, np.nan, np.nan, np.nan                 # sin modo guiado

    kappa = 2 * h ** 2 * q * np.exp(-q * s) / (beta * W * (q ** 2 + h ** 2))
    return kappa, beta, h, q


# ────────────────────────────────────────────────────────────────
# 3. κ cuando el usuario aporta β (h y q se deducen)
# ────────────────────────────────────────────────────────────────
def kappa_from_beta(beta, n_core, n_clad, W, s, lam):
    k0 = 2 * np.pi / lam
    h = np.sqrt(beta**2 - (k0 * n_clad) ** 2)
    q = np.sqrt((k0 * n_core) ** 2 - beta ** 2)
    kappa = 2 * h ** 2 * q * np.exp(-q * s) / (beta * W * (q ** 2 + h ** 2))
    return kappa, h, q


# ────────────────────────────────────────────────────────────────
# 4. Potencias a una longitud L
# ────────────────────────────────────────────────────────────────
def powers_at_L(kappa, L, Pin=1.0):
    return Pin * np.cos(kappa * L) ** 2, Pin * np.sin(kappa * L) ** 2


# ────────────────────────────────────────────────────────────────
# 5. Helper de entrada con valor por defecto
# ────────────────────────────────────────────────────────────────
def ask(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    return float(txt) if txt else default


# ────────────────────────────────────────────────────────────────
# 6. Programa principal
# ────────────────────────────────────────────────────────────────
def main():
    print("\n=== CMT – dos guías rectangulares paralelas ===")

    # ── ¿β proporcionado? ──────────────────────────────────────
    beta_known = input("¿Tienes β (rad/µm) ya calculado? (s/N) "
                       ).lower().startswith("s")

    lam = ask("λ (µm)", 1.55)

    if beta_known:
        beta = ask("β (rad/µm)", 13.0)
        n_core = ask("Índice núcleo  n_core", 2.4)   # para q
        n_clad = ask("Índice cubierta n_clad", 1.444)
    else:
        n_core = ask("Índice núcleo  n_core", 2.4)
        n_clad = ask("Índice cubierta n_clad", 1.444)

    W = ask("Ancho  W (µm)", 1.0)
    H = ask("Altura H (µm)", 0.8)
    

    s_max = ask("Separación máxima s_max  (µm)", 1.0)
    ds = ask("Paso Δs (µm)", 0.02)

    L_user = ask("Longitud L (µm) para evaluar potencias", 500)
    Pin = ask("Pin (potencia normalizada)", 1.0)

    # auto_Lc = input("¿Usar L = Lc(s) automática? (s/N) ").lower().startswith("s")  / Evalua al usuario si desea analizar para una lognitud donde se produce el acoplo Potencia máxima transferida;
    # se ve crecer Lc​ al separar las guías para (Lc​(s)=π/(2κ)) evalua True. False por defecto como esta en la linea de abajo evalua las Oscilaciones de potencia según la fase κ·L.
    auto_Lc = False 

    # ── Barrido de separaciones ────────────────────────────────
    s_vals = np.arange(0, s_max + ds / 2, ds)
    κ_vals, L_effs, Pt_vals, Pc_vals = [], [], [], []

    for s in s_vals:
        if beta_known:
            κ, h, q = kappa_from_beta(beta, n_core, n_clad, W, s, lam)
        else:
            κ, β, h, q = kappa_rect_EIM(n_core, n_clad, W, H, s, lam)

        if np.isnan(κ):      # sin modo → κ=0, no acoplo
            κ = 0.0

        L_eff = np.pi / (2 * κ) if auto_Lc and κ > 0 else L_user
        Pt, Pc = powers_at_L(κ, L_eff, Pin)

        κ_vals.append(κ)
        L_effs.append(L_eff)
        Pt_vals.append(Pt)
        Pc_vals.append(Pc)

    # ── Convertir a array para graficar ────────────────────────
    s_vals = np.array(s_vals)
    Pt_vals = np.array(Pt_vals)
    Pc_vals = np.array(Pc_vals)

    # ── Tabla resumen ──────────────────────────────────────────
    print("\n  s (µm)   κ(µm⁻¹)    L_eff(µm)   P_T   P_C")
    for s, κ, Le, pt, pc in zip(s_vals, κ_vals, L_effs, Pt_vals, Pc_vals):
        print(f" {s:6.2f}  {κ:8.2e}   {Le:9.1f}  {pt:5.2f} {pc:5.2f}")

    # ── Gráfica ────────────────────────────────────────────────
    plt.figure(figsize=(7, 4))
    plt.plot(s_vals, Pt_vals, 'b-', label="Pthrough")
    plt.plot(s_vals, Pc_vals, 'r-', label="Pcross")
    titulo_L = "L = Lc(s)" if auto_Lc else f"L = {L_user} µm"
    plt.title(f"{titulo_L}\n(W={W} µm, H={H} µm, λ={lam} µm)")
    plt.xlabel("Separación s [µm]")
    plt.ylabel("Potencia")
    plt.ylim(0, Pin * 1.05)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Info de referencia para s = 0
    κ0 = κ_vals[0]
    if κ0 > 0:
        print(f"\nPara s = 0 µm → κ = {κ0:.3e} µm⁻¹,  "
              f"Lc = π/(2κ) ≈ {np.pi/(2*κ0):.1f} µm")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
