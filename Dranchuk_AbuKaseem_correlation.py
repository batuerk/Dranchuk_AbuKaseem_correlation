import numpy as np
import matplotlib.pyplot as plt

A = {
    1: 0.3265, 2: -1.07, 3: -0.5339, 4: 0.01569, 5: -0.05165,
    6: 0.5475, 7: -0.7361, 8: 0.1844, 9: 0.1056, 10: 0.6134, 11: 0.7210
}

spes_grav = 0.65
T_F = 150.0 # F
T_R = T_F + 459.67 # R

z_c = 0.27
epsilon_rho = 1e-8
epsilon_p = 1e-5

P_pc = -3.6 * (spes_grav**2) - 131 * spes_grav + 756.8 #psia
T_pc = -74 * (spes_grav**2) + 349.5 * spes_grav + 169.2 # R

T_pr = T_R / T_pc

def f_rho(rho_r, P_pr, T_pr, z_c, A):
    T_r = T_pr
    
    term1 = rho_r * T_r
    term2 = (A[1]*T_r + A[2] + A[3]/T_r**2 + A[4]/T_r**3 + A[5]/T_r**4) * rho_r**2
    term3 = (A[6]*T_r + A[7] + A[8]/T_r) * rho_r**3
    term4 = -A[9] * (A[7] + A[8]/T_r) * rho_r**6
    
    exp_term = np.exp(-A[11] * rho_r**2)
    term5 = A[10] * (1 + A[11] * rho_r**2) * (rho_r**3 / T_r**2) * exp_term
    
    term_pr = -z_c * P_pr
    
    return term1 + term2 + term3 + term4 + term5 + term_pr

def f_prime_rho(rho_r, T_pr, A):
    T_r = T_pr

    term1 = T_r
    term2 = 2 * (A[1]*T_r + A[2] + A[3]/T_r**2 + A[4]/T_r**3 + A[5]/T_r**4) * rho_r
    term3 = 3 * (A[6]*T_r + A[7] + A[8]/T_r) * rho_r**2
    term4 = -6 * A[9] * (A[7] + A[8]/T_r) * rho_r**5
    
    exp_term = np.exp(-A[11] * rho_r**2)
    term5_factor1 = A[10] * (rho_r**2 / T_r**2)
    term5_factor2 = (3 + A[11] * rho_r**2 * (3 - 2 * A[11] * rho_r**2))
    term5 = term5_factor1 * term5_factor2 * exp_term
    
    return term1 + term2 + term3 + term4 + term5

def calculate_z(p, T_pr, P_pc, z_c, A, epsilon_rho):    
    P_pr = p / P_pc
    
    # Başlangıç tahmini yap
    z_initial = 1.0
    rho_r = (z_c * P_pr) / (z_initial * T_pr)
    
    # Newton-Raphson iterasyonu
    for _ in range(100): # Olası bir sonsuz döngüye karşı korumak için
        rho_r_old = rho_r
        
        # Fonksiyonu ve türevini hesapla
        f_val = f_rho(rho_r, P_pr, T_pr, z_c, A)
        f_prime_val = f_prime_rho(rho_r, T_pr, A)
                    
        # Yeni rho_r'yi hesapla
        rho_r = rho_r_old - f_val / f_prime_val
        
        # Yakınsama kriterini kontrol et
        if abs(rho_r - rho_r_old) <= epsilon_rho:
            break
    
    # Yakınsayan rho_r değerini kullanarak z'yi hesapla
    z = (z_c * P_pr) / (rho_r * T_pr)
    return z

def solve_part1(T_pr, P_pc, z_c, A, epsilon_rho):
    
    pressures = np.arange(15, 6001, 1)
    z_factors = []
        
    for p in pressures:
        z = calculate_z(p, T_pr, P_pc, z_c, A, epsilon_rho)
        z_factors.append(z)
                 
    plt.figure(figsize=(10, 6))
    plt.plot(pressures, z_factors, label=f'T = {T_F}°F, spes_grav = {spes_grav}')
    plt.xlabel('Pressure (psia)')
    plt.ylabel('Gas Deviation Factor (z)')
    plt.title('Dranchuk-AbuKassem Correlation')
    plt.grid(True)
    plt.legend()
    plt.show()

# --- Part 2 ---

def find_pressure_for_pz(pz_target, T_pr, P_pc, z_c, A, epsilon_rho, epsilon_p):    
    p_k = pz_target 
    
    delta_p = 0.1
    
    # Newton-Raphson iterasyonu
    for _ in range(200):
        p_k_old = p_k
        
        # Mevcut p_k için z'yi belirle
        z_k = calculate_z(p_k, T_pr, P_pc, z_c, A, epsilon_rho)
        
        # F(p) = p/z(p) - (p/z)_target
        F_p = p_k / z_k - pz_target
        
        # F'(p) türevini nümerik olarak hesapla
        # F'(p) = ( F(p + dp) - F(p) ) / dp
        p_k_plus_dp = p_k + delta_p
        z_k_plus_dp = calculate_z(p_k_plus_dp, T_pr, P_pc, z_c, A, epsilon_rho)

        F_p_plus_dp = p_k_plus_dp / z_k_plus_dp - pz_target
        
        F_prime_p = (F_p_plus_dp - F_p) / delta_p
            
        p_k = p_k_old - F_p / F_prime_p
        
        if abs(p_k - p_k_old) <= epsilon_p:
            break 
            
    return p_k

def solve_part2(T_pr, P_pc, z_c, A, epsilon_rho, epsilon_p):
    Gp_array = np.arange(0, 200, 1)

    pz_array = 5035 - 25.175 * Gp_array
    
    pressures_part2 = []
    
    for pz_val in pz_array:        
        p_calc = find_pressure_for_pz(
            pz_val, T_pr, P_pc, z_c, A, epsilon_rho, epsilon_p
        )
        pressures_part2.append(p_calc)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel('Gp (Bcf)')
    
    # Sol Y ekseni (p/z)
    color = 'tab:blue'
    ax1.set_ylabel('p/z (psia)', color=color)
    ax1.plot(Gp_array, pz_array, color=color, label='p/z vs. Gp')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Sağ Y ekseni (p)
    ax2 = ax1.twinx() # Aynı X eksenini paylaşan ikinci Y ekseni
    color = 'tab:red'
    ax2.set_ylabel('Pressure (p, psia)', color=color)
    ax2.plot(Gp_array, pressures_part2, color=color, label='p vs. Gp')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('p/z vs Gp & p vs. Gp')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":    
    solve_part1(T_pr, P_pc, z_c, A, epsilon_rho)
    
    solve_part2(T_pr, P_pc, z_c, A, epsilon_rho, epsilon_p)
