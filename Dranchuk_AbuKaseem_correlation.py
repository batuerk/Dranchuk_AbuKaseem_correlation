import math
import matplotlib.pyplot as plt

# Constant
A = {
    1: 0.3265,
    2: -1.07,
    3: -0.5339,
    4: 0.01569,
    5: -0.05165,
    6: 0.5475,
    7: -0.7361,
    8: 0.1844,
    9: 0.1056,
    10: 0.6134,
    11: 0.7210
}

spes_grav = 0.65  # Specific gravity
T_F = 150  # Fahrenheit
T_R = T_F + 459.67  # Rankine

# Kritik özellikler
Ppc = -3.6 * spes_grav**2 - 131 * spes_grav + 756.8  # psia
Tpc = -74 * spes_grav**2 + 349.5 * spes_grav + 169.2  # Rankine

zc = 0.27
epsilon = 1e-8

def f_rho(rho_r, T_r, P_r, zc, A):
    term1 = rho_r * T_r
    term2 = (A[1] + A[2]/T_r + A[3]/T_r**3 + A[4]/T_r**4 + A[5]/T_r**5) * rho_r**2
    term3 = (A[6] + A[7]/T_r + A[8]/T_r**2) * rho_r**3
    term4 = -A[9] * (A[7]/T_r + A[8]/T_r**2) * rho_r**6
    term5 = A[10] * (1 + A[11]*rho_r**2) * (rho_r**3 / T_r**3) * math.exp(-A[11]*rho_r**2)
    term6 = -zc * P_r
    return term1 + term2 + term3 + term4 + term5 + term6

def f_rho_prime(rho_r, T_r, A):
    term1 = T_r
    term2 = 2 * (A[1] + A[2]/T_r + A[3]/T_r**3 + A[4]/T_r**4 + A[5]/T_r**5) * rho_r
    term3 = 3 * (A[6] + A[7]/T_r + A[8]/T_r**2) * rho_r**2
    term4 = -6 * A[9] * (A[7]/T_r + A[8]/T_r**2) * rho_r**5
    term5 = A[10] * (rho_r**2 / T_r**3) * (3 + A[11]*rho_r**2*(3 - 2*A[11]*rho_r**2)) * math.exp(-A[11]*rho_r**2)
    return term1 + term2 + term3 + term4 + term5

pressures = range(15, 6000, 1)
z_values = []

for P in pressures:
    Ppr = P / Ppc
    Tpr = T_R / Tpc

    z = 1.0
    rho_r = (zc * Ppr) / (z * Tpr)
    
    for _ in range(1000):
        f_val = f_rho(rho_r, Tpr, Ppr, zc, A)
        f_prime_val = f_rho_prime(rho_r, Tpr, A)
        
        rho_new = rho_r - f_val / f_prime_val
        if abs(rho_new - rho_r) < epsilon:
            rho_r = rho_new
            break
        rho_r = rho_new
    
    z = (zc * Ppr) / (rho_r * Tpr)
    z_values.append(z)


plt.figure(figsize=(8,6))
plt.plot(pressures, z_values, color='b', linewidth=2)
plt.title("Dranchuk-AbuKassem Correlation")
plt.xlabel("Pressure (psia)")
plt.ylabel("Z-Factor")
plt.grid(True)
plt.show()
