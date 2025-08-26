"""
CMT – dos guías rectangulares paralelas)
================================================================
Opción 1: Potencias a L fijo para un s dado.
Opción 2: Barrido en s (manteniendo L fijo o igualando L = Lc(s)).

Modelo:
  - Aproximación de índice efectivo (EIM, 2 pasos) para un canal rectangular.
  - κ(s) ≈ 2 h^2 q exp(-q s) / [ β W (q^2 + h^2) ], con h y q del slab horizontal.
  - Potencias (guías idénticas, Δβ≈0): P_through = cos^2(κ L), P_cross = sin^2(κ L).

Unidades:
  - λ, W, H, s, L en µm; β, h, q, κ en 1/µm .

Requisitos: numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


# ────────────────────────────────────────────────────────────────
# 1) TE0 en slab simétrico (robusto en forma normalizada)
# ────────────────────────────────────────────────────────────────
def slab_TE0(n_core, n_clad, thickness, lam):
    """
    Devuelve beta, h, q para el modo TE0 de un slab simétrico.
    Convención:
      beta [rad/µm], h y q [1/µm]; k0 = 2π/λ [rad/µm]
      h = sqrt((k0 n_core)^2 - beta^2)
      q = sqrt(beta^2 - (k0 n_clad)^2)
    """
    k0 = 2*np.pi/lam
    V = 0.5*thickness*np.sqrt((k0**2)*(n_core**2 - n_clad**2))  # adimensional
    if not np.isfinite(V) or V <= 0:
        raise ValueError("No hay guiado (V<=0) para el slab especificado.")

    def G(u):
        # Ecuación TE0: tan(u) = w/u, con u=h*t/2, w=q*t/2 y u^2 + w^2 = V^2
        w_sq = V*V - u*u
        w = np.sqrt(w_sq) if w_sq > 0 else 0.0
        return np.tan(u) - (w/u if u != 0 else np.inf)

    a = 1e-9
    b = np.pi/2 - 1e-9
    sol = root_scalar(G, bracket=[a, b], method='bisect', xtol=1e-12, rtol=1e-12, maxiter=1000)
    u = sol.root
    h = 2*u/thickness
    beta = np.sqrt((k0*n_core)**2 - h**2)
    q = np.sqrt(max(beta**2 - (k0*n_clad)**2, 0.0))
    return beta, h, q


# ────────────────────────────────────────────────────────────────
# 2) κ(s) vía EIM (vertical → horizontal)
# ────────────────────────────────────────────────────────────────
def kappa_rect_EIM(n_core, n_clad, W, H, s, lam):
    """
    Calcula κ(s) para dos guías idénticas rectangulares.
    Paso 1: slab vertical → n_eff_v
    Paso 2: slab horizontal con n_core = n_eff_v
    """
    beta_v, _, _ = slab_TE0(n_core, n_clad, H, lam)
    n_eff_v = beta_v / (2*np.pi/lam)  # = beta_v/k0

    try:
        beta, h, q = slab_TE0(n_eff_v, n_clad, W, lam)
    except Exception:
        return 0.0, np.nan, np.nan, np.nan  # sin modo en horizontal

    kappa = 2*h*h * q * np.exp(-q*s) / (beta * W * (q*q + h*h))
    return kappa, beta, h, q


# ────────────────────────────────────────────────────────────────
# 3) κ(s) cuando el usuario ya conoce beta del slab horizontal
# ────────────────────────────────────────────────────────────────
def kappa_from_beta(beta, n_core_eff, n_clad, W, s, lam):
    k0 = 2*np.pi/lam
    h = np.sqrt((k0*n_core_eff)**2 - beta**2)
    q = np.sqrt(max(beta**2 - (k0*n_clad)**2, 0.0))
    kappa = 2*h*h * q * np.exp(-q*s) / (beta * W * (q*q + h*h))
    return kappa, h, q


# ────────────────────────────────────────────────────────────────
# 4) Potencias en una longitud L (Δβ≈0)
# ────────────────────────────────────────────────────────────────
def powers_at_L(kappa, L, Pin=1.0):
    Pt = Pin * np.cos(kappa*L)**2
    Pc = Pin * np.sin(kappa*L)**2
    return Pt, Pc


# ────────────────────────────────────────────────────────────────
# 5) Utilidades de entrada
# ────────────────────────────────────────────────────────────────
def ask_float(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    return float(txt) if txt else default

def ask_bool(prompt, default=False):
    d = "s" if default else "n"
    txt = input(f"{prompt} (s/n) [{d}]: ").strip().lower()
    if not txt:
        return default
    return txt.startswith("s")


# ────────────────────────────────────────────────────────────────
# 6) Programa principal 
# ────────────────────────────────────────────────────────────────
def main():
    print("\n=== CMT – dos guías paralelas (EIM + κ(s)) ===")
    print("  [1] Potencias a L fijo para un s dado")
    print("  [2] Barrido en separación s")
    choice = input("Elige opción [1/2]: ").strip() or "1"

    # Parámetros comunes
    lam = ask_float("Longitud de onda λ (µm)", 1.55)
    beta_known = ask_bool("¿Ya conoces β del slab horizontal?")
    if beta_known:
        beta = ask_float("β (rad/µm)", 10.0)
        n_core_eff = ask_float("Índice efectivo del 'core' horizontal (n_core_eff)", 2.4)
        n_core = ask_float("Índice núcleo vertical  n_core (para referencia)", 2.4)
        n_clad = ask_float("Índice cubierta       n_clad", 1.444)
    else:
        n_core = ask_float("Índice núcleo  n_core", 2.4)
        n_clad = ask_float("Índice cubierta n_clad", 1.444)

    W = ask_float("Ancho  W (µm)", 1.0)
    H = ask_float("Altura H (µm)", 1.0)

    if choice == "1":
        # ── Opción 1: Potencias para s y L dados ─────────────────
        s = ask_float("Separación s (µm)", 0.5)
        L = ask_float("Longitud L (µm)", 1000)
        Pin = ask_float("Potencia de entrada Pin (normalizada)", 1.0)

        if beta_known:
            kappa, h, q = kappa_from_beta(beta, n_core_eff, n_clad, W, s, lam)
        else:
            kappa, beta_h, h, q = kappa_rect_EIM(n_core, n_clad, W, H, s, lam)

        Pt, Pc = powers_at_L(kappa, L, Pin)
        Lc = np.pi/(2*kappa) if kappa > 0 else np.inf

        print("\n— Resultados (Opción 1) —")
        print(f"  κ = {kappa:.4e} µm⁻¹   Lc = {Lc:.2f} µm")
        print(f"  P_through(L) = {Pt:.4f}   P_cross(L) = {Pc:.4f}")

        if ask_bool("¿Graficar P_through y P_cross vs z?", True) and np.isfinite(Lc):
            z = np.linspace(0, L, 500)
            Pt_z = np.cos(kappa*z)**2
            Pc_z = np.sin(kappa*z)**2

            plt.figure(figsize=(8,4))
            plt.plot(z, Pt_z, label="P_through")
            plt.plot(z, Pc_z, label="P_cross")
            plt.xlabel("z [µm]")
            plt.ylabel("Potencia óptica (normalizada)")
            plt.title(f"Acoplador: s={s} µm, W={W} µm, H={H} µm, λ={lam} µm")
            plt.grid(alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()

    else:
        # ── Opción 2: Barrido en s ───────────────────────────────
        s_min = ask_float("s_min (µm)", 0.0)
        s_max = ask_float("s_max (µm)", 1.0)
        ds    = ask_float("Paso Δs (µm)", 0.02)
        L     = ask_float("Longitud L (µm)", 1000.0)
        Pin   = ask_float("Potencia de entrada Pin (normalizada)", 1.0)
        auto_Lc = ask_bool("¿Usar L = Lc(s) automática?", False)

        s_vals, k_vals, Leffs, Pts, Pcs = [], [], [], [], []

        s = s_min
        while s <= s_max + 1e-12:
            if beta_known:
                kappa, h, q = kappa_from_beta(beta, n_core_eff, n_clad, W, s, lam)
            else:
                kappa, beta_h, h, q = kappa_rect_EIM(n_core, n_clad, W, H, s, lam)

            if not np.isfinite(kappa) or kappa <= 0:
                Le = L
                Pt, Pc = Pin, 0.0
                kappa = 0.0
            else:
                Le = (np.pi/(2*kappa)) if auto_Lc else L
                Pt, Pc = powers_at_L(kappa, Le, Pin)

            s_vals.append(s); k_vals.append(kappa); Leffs.append(Le); Pts.append(Pt); Pcs.append(Pc)
            s += ds

        s_vals = np.array(s_vals); Pts = np.array(Pts); Pcs = np.array(Pcs)

        print("\n— Tabla (Opción 2) —")
        print("  s (µm)   κ(µm⁻¹)     L_evaluado(µm)   P_th   P_cr")
        for s, k, Le, pt, pc in zip(s_vals, k_vals, Leffs, Pts, Pcs):
            print(f" {s:6.3f}  {k:9.3e}   {Le:12.2f}     {pt:5.3f} {pc:5.3f}")

        # Gráfica (una figura, dos curvas)
        titulo_L = "L = Lc(s)" if auto_Lc else f"L = {L} µm"
        plt.figure(figsize=(8,4))
        plt.plot(s_vals, Pts, label="P_through")
        plt.plot(s_vals, Pcs, label="P_cross")
        plt.xlabel("Separación s [µm]")
        plt.ylabel("Potencia óptica (normalizada)")
        plt.title(f"Barrido en s — {titulo_L}\n(W={W} µm, H={H} µm, λ={lam} µm)")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

        # Info en s=0 (si aplica)
        if len(k_vals) > 0 and k_vals[0] > 0:
            print(f"\nPara s = {s_vals[0]:.3f} µm → κ = {k_vals[0]:.3e} µm⁻¹,  "
                  f"Lc = π/(2κ) ≈ {np.pi/(2*k_vals[0]):.2f} µm")


if __name__ == "__main__":
    main()
