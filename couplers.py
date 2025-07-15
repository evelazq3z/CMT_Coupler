"""
CMT simplificado: Pthrough y Pcross en función de la longitud L
================================================================
Parámetros solicitados al usuario:
    β  : constante de propagación (rad/µm)
    W  : ancho del núcleo   (µm)
    s  : separación entre guías (µm)
    L  : longitud total que se desea mostrar (µm)
Opcional:
    Pin: potencia de entrada (1 por defecto)

Modelo sin pérdidas (α = 0):
    P_T(z) = Pin · cos²(κ z)
    P_C(z) = Pin · sin²(κ z)
κ calculado con la ec. 3.2.15 usando h ≈ q ≈ π / W.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
def kappa(beta: float, width: float, separation: float) -> float:
    """Ecuación 3.2.15 con h ≈ q ≈ π/W."""
    h = q = np.pi / width           # µm⁻¹
    return 2 * h**2 * q * np.exp(-q * separation) / (beta * width * (q**2 + h**2))

def powers(z: np.ndarray, kappa_: float, pin: float = 1.0):
    """Potencias Through (P0) y Cross (P1) sin pérdidas."""
    P_through = pin * np.cos(kappa_ * z) ** 2
    P_cross   = pin * np.sin(kappa_ * z) ** 2
    return P_through, P_cross
# ----------------------------------------------------------------------

def pedir_float(txt, default):
    dato = input(f"{txt} [{default}]: ").strip()
    return float(dato) if dato else default

def main():
    print("\n=== Coupled-Mode Theory (simplificado) ===\n"
          "Pulsa Enter para aceptar los valores por defecto.\n")

    beta = pedir_float("β  (rad/µm)",        12.1)
    W    = pedir_float("Ancho W  (µm)",       2.0)
    s    = pedir_float("Separación s (µm)",   2.0)
    L    = pedir_float("Longitud L a mostrar (µm)", 500)
    Pin  = pedir_float("Pin (potencia de entrada)", 1.0)

    κ  = kappa(beta, W, s)
    print(f"\nκ calculado = {κ:.4e} µm⁻¹  →  Lc₁ = π/(2κ) ≈ {np.pi/(2*κ):.1f} µm\n")

    z = np.linspace(0, L, 1500)
    P_t, P_c = powers(z, κ, Pin)

    # ---------- Gráfica -------------------------------------------
    plt.figure(figsize=(8,4))
    plt.plot(z, P_t, label="Pthrough (guía 0)")
    plt.plot(z, P_c, label="Pcross   (guía 1)")
    plt.axvline(np.pi/(2*κ), ls="--", c="grey", lw=.8,
                label=r"1ᵉʳ acoplo 100 % ($\pi/2κ$)")
    plt.xlabel("z  [µm]")
    plt.ylabel("Potencia")
    plt.title(f"Intercambio de potencia (W={W} µm, s={s} µm)")
    plt.ylim(0, Pin*1.05)
    plt.xlim(0, L)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
