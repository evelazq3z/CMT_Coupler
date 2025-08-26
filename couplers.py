"""
CMT de dos guías (con pérdidas): P0(z)=Pin cos^2(k z) e^{-α z}, P1(z)=Pin sin^2(k z) e^{-α z}
κ(s) ≈ [2 h^2 p e^{-p s}] / [β W (p^2 + h^2)]  (h: oscilación en núcleo, p: decaimiento lateral en la cubierta)

Opción 1: Potencias a L fijo para un s dado (z ∈ [0, L]).
Opción 2: Barrido en s (L fijo o L = Lc(s) = π/(2κ)).

Requisitos: numpy, scipy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ---------- utilidades ----------
def dBcm_to_Np_per_um(alpha_dB_cm):
    # Potencia: α[Np/µm] = (ln(10)/10)*α[dB/µm]
    return (np.log(10)/10.0) * (alpha_dB_cm/1e4)

def ask_float(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    return float(txt) if txt else default

def ask_bool(prompt, default=False):
    d = "s" if default else "n"
    txt = input(f"{prompt} (s/n) [{d}]: ").strip().lower()
    return (txt.startswith("s") if txt else default)

# ---------- slab TE0 simétrico (forma normalizada: u,w) ----------
def slab_TE0(n_core, n_clad, thickness, lam):
    """
    Devuelve: beta [rad/µm], h_core [1/µm], p_clad [1/µm]
    u = h_core*t/2, w = p_clad*t/2,   u^2 + w^2 = V^2,   tan(u)=w/u
    """
    k0 = 2*np.pi/lam
    V = 0.5*thickness*np.sqrt((k0**2)*(n_core**2 - n_clad**2))
    if not np.isfinite(V) or V <= 0:
        raise ValueError("Sin guiado (V≤0).")
    def G(u):
        w2 = V*V - u*u
        if w2 <= 0: return np.inf*np.sign(u)
        w = np.sqrt(w2)
        return np.tan(u) - (w/u if u != 0 else np.inf)
    sol = root_scalar(G, bracket=[1e-9, np.pi/2-1e-9], method='bisect', xtol=1e-12, rtol=1e-12)
    u = sol.root
    w = np.sqrt(max(V*V - u*u, 0.0))
    h_core = 2*u/thickness
    p_clad = 2*w/thickness
    beta   = np.sqrt((k0*n_core)**2 - h_core**2)
    return beta, h_core, p_clad

# ---------- κ(s) por EIM (vertical→horizontal) ----------
def kappa_rect_EIM(n_core, n_clad, W, H, s, lam):
    # Paso vertical → n_eff_v
    beta_v, _, _ = slab_TE0(n_core, n_clad, H, lam)
    n_eff_v = beta_v/(2*np.pi/lam)         # = beta_v/k0
    # Paso horizontal (núcleo = n_eff_v)
    beta_h, h, p = slab_TE0(n_eff_v, n_clad, W, lam)
    kappa = 2*h*h * p * np.exp(-p*s) / (beta_h * W * (p*p + h*h))
    return kappa, beta_h, h, p

# ---------- potencias con pérdidas ----------
def powers_with_loss(kappa, alpha_np_um, z, Pin=1.0):
    # P0(z) = Pin cos^2(k z) e^{-α z}, P1(z) = Pin sin^2(k z) e^{-α z}
    decay = np.exp(-alpha_np_um * z)
    return Pin * (np.cos(kappa*z)**2) * decay, Pin * (np.sin(kappa*z)**2) * decay

# ===================== main =====================
def main():
    print("\n=== CMT – dos guías paralelas (con pérdidas) ===")
    print("  [1] Potencias a L,s fijo   (z ∈ [0, L])")
    print("  [2] Barrido en separación s          (L fijo o L = Lc(s))")
    choice = (input("Seleccione una opción [1] [2]: ").strip() or "1")

    lam = ask_float("Longitud de onda λ (µm)", 1.55)
    n_core = ask_float("Índice del núcleo n_core a-Si:H", 3.4918)
    n_clad = ask_float("Índice de la cubierta n_clad SiO2" , 1.42)
    W      = ask_float("Ancho  W (µm)", 1.0)
    H      = ask_float("Altura H (µm)", 1.0)
    alpha_dB_cm = ask_float("Pérdidas α (dB/cm)", 0.0)
    alpha_np_um = dBcm_to_Np_per_um(alpha_dB_cm)

    if choice == "1":
        s = ask_float("Separación s (µm)", 0.25)
        L = ask_float("Longitud L (µm)", 1000.0)
        Pin = ask_float("Potencia de entrada Pin (normalizada)", 1.0)

        try:
            kappa, beta_h, h, p = kappa_rect_EIM(n_core, n_clad, W, H, s, lam)
        except Exception as e:
            print("No hay modo guiado en los pasos EIM:", e)
            return

        Lc = (np.pi/(2*kappa)) if (kappa > 0) else np.inf
        PtL, PcL = powers_with_loss(kappa, alpha_np_um, L, Pin)

        print("\n— Resultados —")
        print(f"  κ = {kappa:.4e} µm⁻¹   Lc ≈ {Lc:.2f} µm   (p = {p:.3e} 1/µm)")
        print(f"  t^2=cos^2(κL)={(np.cos(kappa*L)**2):.4f},  c^2=sin^2(κL)={(np.sin(kappa*L)**2):.4f}")
        print(f"  P_through(L) = {PtL:.4f}   P_cross(L) = {PcL:.4f}   (con e^(-αL))")

        # ---------- Gráfica -------------------------------------------
        z = np.linspace(0, L, 600)
        Pt, Pc = powers_with_loss(kappa, alpha_np_um, z, Pin)
        plt.figure(figsize=(8,4))
        plt.plot(z, Pt, label="P_through")
        plt.plot(z, Pc, label="P_cross")
        plt.axvline(Lc, ls="--", lw=.8, label="L_c = π/(2κ)")
        plt.xlabel("z [µm]")
        plt.ylabel(" Optical Power Pin (u.a)")
        plt.title(f"Directional Coupler at (W={W} µm, H={H} µm, λ={lam} µm, s={s} µm), L={L} µm")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    else:
        s_min = ask_float("s_min (µm)", 0.0)
        s_max = ask_float("s_max (µm)", 1.0)
        ds    = ask_float("Paso Δs (µm)", 0.01)
        L     = ask_float("Longitud L (µm)", 1000.0)
        Pin   = ask_float("Pin (normalizada)", 1.0)
        auto_Lc = False

        s_vals=[]; Pth=[]; Pcr=[]
        s = s_min
        while s <= s_max + 1e-12:
            try:
                kappa, beta_h, h, p = kappa_rect_EIM(n_core, n_clad, W, H, s, lam)
            except Exception:
                kappa = 0.0
            L_eval = (np.pi/(2*kappa)) if (auto_Lc and kappa>0) else L
            Pt, Pc = powers_with_loss(kappa, alpha_np_um, L_eval, Pin) if kappa>0 else (Pin*np.exp(-alpha_np_um*L_eval), 0.0)
            s_vals.append(s); Pth.append(Pt); Pcr.append(Pc)
            s += ds

        s_vals = np.array(s_vals); Pth=np.array(Pth); Pcr=np.array(Pcr)
        titulo_L = "L = Lc(s)" if auto_Lc else f"L = {L} µm"
        plt.figure(figsize=(8,4))
        plt.plot(s_vals, Pth, label="P_through @L")
        plt.plot(s_vals, Pcr, label="P_cross @L")
        plt.xlabel("Separation s [µm]"); plt.ylabel("Optical Power (u.a)")
        plt.title(f"Sweep in s — {titulo_L}\nλ={lam} µm, α={alpha_dB_cm} dB/cm, W×H={W}×{H} µm²")
        plt.grid(alpha=0.3); plt.legend(loc="best"); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
