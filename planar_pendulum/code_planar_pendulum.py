#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 23:49:39 2026

@author: rogerio
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.integrate import solve_ivp
from scipy.special import ellipj, ellipkinc

# ======================
#
# 1. PHYSICAL PARAMETERS
# ======================

g = 9.81      # gravity [m/s²]
L = 1.0       # pendulum length [m]

omega0 = np.sqrt(g/L)  # natural frequency (mass-independent!)


# Instead of setting dot_theta0 directly, let's specify the dimensionless energy ε = E/ mgL
# This is more fundamental and shows the physics better

# Choose the motion regime:
#  ε < 2: Oscillatory -Bound motion (oscilatory)
#  ε = 2: Separatrix  -Bound motion (with unstable critical point)
#  ε > 2: complete circulation -Unbound motion.

#Set the initial position
theta_0 = - np.pi / 2

# Choose dimensionless energy: ε = E/(mgL)
# oscillation with amplitude θ_max = arccos(1-ε)
# To start from rest simply set: ε = 1 - np.cos(theta_0)
ε =  1.5 # 1 - np.cos(theta_0)  # remember: ε = E/(mgL)


# Checking if there is any motion
if ε < (1 - np.cos(theta_0)):
    raise ValueError(f"Energy ε={ε} too low; for θ₀={theta_0} we need ε ≥ {1 - np.cos(theta_0):.3f}")


# Compute dθ̇0 from energy conservation: ε = (1/2)(dθ̇²/ω₀²) + (1 - cosθ)
# Rearranging: dθ̇ = ω₀ * xi * sqrt(2[ε - (1 - cosθ)])
xi = + 1

dot_theta_0 = omega0 * xi * np.sqrt(2 * (ε - (1 - np.cos(theta_0))))

# Choosing the time interval:
t0 = 0.0
T_small = 2 * np.pi / omega0 #Small angle period for comparison

t_max = 8*T_small # Max time for numerical integration / analytical plots.

t_eval = np.linspace(0, t_max, 2000)


print(f"  1. Parameters: ")
print(f"  g = {g:.3f} [m/s²], L = {L:.3f} [m] → ω₀ = √(g/L) = {omega0:.3f} [rad/s] ")
print(f"  Initial: θ₀ = {theta_0:.3f} [rad], θ̇'₀ = {dot_theta_0:.3f} [rad/s]") 
print()
print(f"  Small-angle period T_small = 2π/ω₀ = {T_small:.6f} [s] ")

print()


# ======================
# 2. NUMERICAL INTEGRATION
# ======================
print("2. Numerical integration...")

def pendulum_ode(t, y):
    """System: dθ/dt = ω, dω/dt = -(g/L)sinθ"""
    theta, omega = y
    return [omega, -(g/L) * np.sin(theta)]

ICs = [theta_0, dot_theta_0]

sol = solve_ivp(pendulum_ode, [0, t_max], ICs, t_eval=t_eval, 
                rtol=1e-9, atol=1e-12, method='DOP853')

theta_num = sol.y[0]
dot_theta_num = sol.y[1]

print("  ODE: θ̈  + (g/L)sinθ = 0 (mass-independent)")
print("  Did the numerical integration reach the max value?", sol.success)

# ======================
# 2.1 NUMERICAL PERIOD CALCULATION
# ======================

# Method 1: For oscillatory motion (ε < 2), find time between successive maxima
# Method 2: For rotational motion (ε ≥ 2), find time for θ to change by 2π

if ε < 2.0:
    # OSCILLATORY MOTION: Find period from zero crossings with positive slope
    # Find indices where theta crosses 0 with positive derivative
    zero_crossings = []
    for i in range(1, len(theta_num)):
        if theta_num[i-1] * theta_num[i] < 0 and dot_theta_num[i] > 0:
            t1, t2 = t_eval[i-1], t_eval[i]
            y1, y2 = theta_num[i-1], theta_num[i]
            t_cross = t1 - y1 * (t2 - t1) / (y2 - y1)
            zero_crossings.append(t_cross)
    
    if len(zero_crossings) >= 2:
        T_num = zero_crossings[1] - zero_crossings[0]
        print(f"  Numerical period T_num = {T_num:.6f} s")
    else:
        print("  WARNING: Could not find enough zero crossings for period calculation")
        print("  Try increasing t_max or adjusting initial conditions")
        T_num = None

else:
    # ROTATIONAL MOTION (ε ≥ 2): Find period from full 2π rotations
    # Track when theta crosses multiples of 2π with consistent direction
    crossings = []
    
    # Unwrap theta to get continuous angle
    theta_unwrapped = np.unwrap(theta_num)
    
    # Find when theta_unwrapped crosses multiples of 2π
    for i in range(1, len(theta_unwrapped)):
        # Check for crossing of 2π*n (where n is integer)
        prev = theta_unwrapped[i-1] / (2*np.pi)
        curr = theta_unwrapped[i] / (2*np.pi)
        
        # If integer part changed, we crossed a multiple of 2π
        if np.floor(prev) != np.floor(curr):
            # Linear interpolation for exact crossing time
            t1, t2 = t_eval[i-1], t_eval[i]
            y1, y2 = prev - np.floor(prev), curr - np.floor(curr)
            # We want time when fractional part = 0 (or 1)
            # Since we crossed from <1 to >1 or vice versa
            if y1 < y2:  # Crossing upward (0 to 1 equivalent)
                t_cross = t1 + (0 - y1) * (t2 - t1) / (y2 - y1)
            else:  # Crossing downward (1 to 0 equivalent)
                t_cross = t1 + (1 - y1) * (t2 - t1) / (y2 - y1)
            crossings.append(t_cross)
    
    if len(crossings) >= 2:
        # Period is time between successive crossings in same direction
        # For consistent rotation, all crossings should be periodic
        periods = np.diff(crossings)
        if len(periods) > 0:
            T_num = np.mean(periods)
            T_num_std = np.std(periods)
            print(f"  Found {len(crossings)} 2π crossings")
            print(f"  Numerical period T_num = {T_num:.6f} s")
            print(f"  Standard deviation of periods: {T_num_std:.6f} s")
            
            # Also compute average angular velocity
            avg_omega = 2 * np.pi / T_num
            print(f"  Average angular velocity: {avg_omega:.6f} rad/s")
    else:
        print("  WARNING: Could not find complete rotations for period calculation")
        print("  Try increasing t_max for rotational motion")
        T_num = None

print()

T_exact = None

# ======================
#
# 3. ANALYTICAL SOLUTION
# ======================

print("3. Analytical solution")

# Compute exact solutions
theta_exact = np.zeros_like(t_eval)
dot_theta_exact = np.zeros_like(t_eval)

##################################
# BOUND MOTION --> ε < 2:
# Solution is: θ(t) = 2* arcsin( k* sn( ξ*ω₀*(t - t₀) + F(ψ₀, k) ) ) 
##################################    
    
if ε < 2.0:
    # Setting the parameters for the elliptic functions:
    theta_max = np.arccos( 1- ε )
    k = np.sin(theta_max /2)
    
    if k > 1e-12:  # Avoid division by zero for very small k
        psi_0 = np.arcsin(np.sin(theta_0/2) / k)
        # Ensure ψ₀ is in the correct branch (-π/2 ≤ ψ₀ ≤ π/2)
        psi_0 = np.clip(psi_0, -np.pi/2, np.pi/2)
    else:
        # Small-angle limit: k ≈ θ_max/2 ≈ 0
        psi_0 = theta_0/2
        
    # Incomplete elliptic integral F(ψ₀, k) 
    # scipy.special.ellipkinc uses parameter m = k²
    F_psi_0 = ellipkinc(psi_0, k**2)
    
    # Define the exact solution function
    # Note that in scipy.special ellipj automatically obtain all the four Jacobi elliptic functions,
    # we only need the sn(u,m=k**2), which is ellipj(u,m)[0]
    def exact_state_bound(t):
        u = xi * omega0 * (t - t0) + F_psi_0
        # ellipj returns (sn, cn, dn, amplitude)
        sn_u, cn_u, dn_u, amplitude_u = ellipj(u, k**2)
        
        # Position: θ(t) = 2 arcsin(k * sn(u, k))
        theta = 2 * np.arcsin(k * sn_u)
        
        # Velocity: θ̇'(t) = 2kω₀ cn(u,k) * ξ
        dot_theta = xi * omega0 * 2 * k * cn_u
        
        return theta, dot_theta
    
    for i, t in enumerate(t_eval):
        theta, dot_theta = exact_state_bound(t)
        theta_exact[i] = theta
        dot_theta_exact[i] = dot_theta
    
    # Period from complete elliptic integral K(k)
    K_k = ellipkinc(np.pi/2, k**2)  # K(k) = F(π/2, k)
    T_exact = 4 * K_k / omega0
    
    print(f"  (Dimensionless) Energy ε = {ε:.3f} --BOUND MOTION--")
    print(f"  Exact period T = 4K(k)/ω₀ = {T_exact:.6f} s")
    print(f"  Maximum angular displacement θ_max = {theta_max:.6f} [rad] " )
    print() 

############################    
# SEPARATRIX SOLUTION --> ε = 2, the analytical solution is:
# θ(t) = 4 * arctan[ tan(θ₀/4 + π/4) * exp(ξ*ω₀*(t - t₀)) ] - π
###########################

elif np.isclose(ε, 2.0):

    def exact_state_sep(t):
        A0 = np.tan(theta_0 / 4 + np.pi / 4)
        theta = 4 * np.arctan(A0 * np.exp(xi * omega0 * (t - t0))) - np.pi
        dot_theta = 2 * xi * omega0 * np.cos(theta / 2)
        return theta, dot_theta
    
    
    for i, t in enumerate(t_eval):
        theta_exact[i], dot_theta_exact[i] = exact_state_sep(t)
        
    print(f"  (Dimensionless) Energy ε = {ε:.3f} --SEPARATRIX--")
    
#################################
# UNBOUND MOTION --> ε > 2:
# Solution is: θ(t) = 2* am( ξ*ω₀/j*(t - t₀) + F(ψ2, k) | j )  
##################################    

else:
    # Setting the parameters for elliptic integrals. Note that ψ2 is just the initial condition.
    j = np.sqrt(2 / ε)
    
    psi_2 = theta_0 / 2
    
    F_psi_2 = ellipkinc(psi_2, j**2)
    
    def exact_state_unbound(t):
        u = xi * omega0 * np.sqrt(ε / 2) * (t - t0) + F_psi_2
        
        # ellipj returns (sn, cn, dn, amplitude)
        sn_u, cn_u, dn_u, amplitude_u = ellipj(u, j**2)
        
        theta = 2 * amplitude_u
        
        dot_theta = xi * omega0 * np.sqrt(2 * ε) * dn_u
        
        return theta, dot_theta


    for i, t in enumerate(t_eval):
        theta_exact[i], dot_theta_exact[i] = exact_state_unbound(t)

    print(f"  (Dimensionless) Energy ε = {ε:.3f} --UNBOUND MOTION--")
    

# %%
# ======================
#
# 4. PLOTS!
# ======================

print("4. Let's now see things!")

# First, let's compute the small-angle approximation for comparison
theta_small = theta_0 * np.cos(omega0 * (t_eval - t0)) + (dot_theta_0/omega0) * np.sin(omega0 * (t_eval - t0))
dot_theta_small = -omega0 * theta_0 * np.sin(omega0 * (t_eval - t0)) + dot_theta_0 * np.cos(omega0 * (t_eval - t0))

# Note that, since we set the energy, the small angle approximation only works well when we the pendulum starts from rest,
# otherwise, there will be a a difference in amplitude and phase.

# Create the 2x3 figure grid
fig = plt.figure(figsize=(15, 10))

# ======================
# SUBPLOT 1: θ(t) vs time
# ======================
ax1 = plt.subplot(2, 3, 1)

line_num, = ax1.plot(t_eval, theta_num, 'k--', linewidth=2.5, label='Numerical (ODE)')
line_exact, = ax1.plot(t_eval, theta_exact, 'b-', linewidth=2, alpha=0.8, label='Analytical (elliptic)')
line_small, = ax1.plot(t_eval, theta_small, 'g-', linewidth=1.5, alpha=0.7, label='Small-angle approx.')

curves_legend = ax1.legend(handles=[line_num, line_exact, line_small], 
                         loc='lower right', fontsize=9, framealpha=0.9)

period_handles = [
    mlines.Line2D([], [], color='g', linestyle=':', alpha=0,
                  label=f'T_small = {T_small:.4f} s'),
]

if T_exact is not None:
    period_handles.append(
        mlines.Line2D([], [], color='b', linestyle=':', alpha=0,
                      label=f'T_exact = {T_exact:.4f} s')
    )

if T_num is not None:
    period_handles.append(
        mlines.Line2D([], [], color='k', linestyle=':', alpha=0,
                      label=f'T_num = {T_num:.4f} s')
    )


periods_text = f'Periods:\nT_small = {T_small:.4f} s'

if T_exact is not None:
    periods_text += f'\nT_exact = {T_exact:.4f} s'

if T_num is not None:
    periods_text += f'\nT_num   = {T_num:.4f} s'


ax1.text(0.02, 0.02, periods_text,
         transform=ax1.transAxes, fontsize=8, 
         va='bottom', ha='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


ax1.set_xlabel('Time [s]', fontsize=11)
ax1.set_ylabel(r'$\theta(t)$ [rad]', fontsize=11)
ax1.set_title('Angular Position vs Time', fontsize=12)
ax1.grid(True, alpha=0.3)



# ======================
# SUBPLOT 2: θ relative error
# ======================

ax2 = plt.subplot(2, 3, 2)

# Compute absolute errors

error_analytical = np.abs((theta_num - theta_exact))
error_small = np.abs((theta_num - theta_small))

ax2.semilogy(t_eval, error_analytical, 'b-', linewidth=1.5, 
             alpha=0.8, label='Absolute Error (Analytic)')
ax2.semilogy(t_eval, error_small, 'g-', linewidth=1.5, 
             alpha=0.8, label='Absolute Error (Small)')

ax2.set_xlabel('Time [s]', fontsize=11)
ax2.set_ylabel(r'Absolute error in $\theta$ [rad/s]' , fontsize=11)
ax2.set_title('Absolute Angular Position Errors', fontsize=12)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

# Update text box for relative errors
max_err_analytic = np.max(error_analytical)
max_err_small = np.max(error_small)

ax2.text(0.02, 0.02, 
         f'Max Abs Errors:\nAnalytic: {max_err_analytic:.2e}\nSmall: {max_err_small:.2e}',
         transform=ax2.transAxes, fontsize=8, va='bottom', ha='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))



# ======================
# SUBPLOT 3: Phase space plot
# ======================
ax3 = plt.subplot(2, 3, 3)

# Create grid for phase space contours
theta_range = np.linspace(-2*np.pi, 2*np.pi, 2000)
dot_theta_range = np.linspace(-4*omega0, 4*omega0, 2000)
Theta_grid, DotTheta_grid = np.meshgrid(theta_range, dot_theta_range)

# Compute dimensionless energy on grid: ε = (1/2)(θ̇²/ω₀²) + (1 - cosθ)
E_grid = 0.5 * (DotTheta_grid**2) / omega0**2 + (1 - np.cos(Theta_grid))

# Different levels are shown, the selected orbit will be highlighted in blue
levels = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
contour_gray = ax3.contour(Theta_grid, DotTheta_grid, E_grid, levels=levels, 
                          colors='gray', alpha=0.5, linewidths=0.5)
ax3.clabel(contour_gray, inline=1, fontsize=7, fmt='%.1f')

# Highlight the chosen energy contour in blue
if ε < 3.0:  # Only plot if within our range
    contour_blue = ax3.contour(Theta_grid, DotTheta_grid, E_grid, levels=[ε], 
                              colors='blue', linewidths=2, alpha=0.7)
    ax3.clabel(contour_blue, inline=1, fontsize=9, fmt=f'ε={ε:.2f}')

# Some illustrative information on the phase space
# Plot the numerical trajectory
ax3.plot(theta_num, dot_theta_num, 'b-', linewidth=1, alpha=0.7, label='Trajectory')

# Mark initial condition with red dot
ax3.plot(theta_0, dot_theta_0, 'ro', markersize=3, label='Initial condition')

# Mark equilibrium states
ax3.plot(0, 0, 'mo', markersize=3, alpha=0.5, label='Stable equilibrium')
ax3.plot([-np.pi, np.pi], [0, 0], 'm^', markersize=4, alpha=0.8, label='Unstable equil. (θ=±π)')

ax3.set_xlabel(r'$\theta$ [rad]', fontsize=11)
ax3.set_ylabel(r'$\dot{\theta}$ [rad/s]', fontsize=11)
ax3.set_title('Phase Space', fontsize=12)
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-2*np.pi, 2*np.pi])
ax3.set_ylim([-4*omega0, 4*omega0])
ax3.set_xticks([-2*np.pi, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 2*np.pi])
ax3.set_xticklabels([r'$-2\pi$', r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$', r'$2\pi$' ])

# ======================
# SUBPLOT 4: θ̇(t) vs time
# ======================

ax4 = plt.subplot(2, 3, 4)
ax4.plot(t_eval, dot_theta_num, 'k--', linewidth=2.5, label='Numerical (ODE)')
ax4.plot(t_eval, dot_theta_exact, 'b-', linewidth=2, alpha=0.8, label='Analytical (elliptic)')
ax4.plot(t_eval, dot_theta_small, 'g-', linewidth=1.5, alpha=0.7, label='Small-angle approx.')

ax4.set_xlabel('Time [s]', fontsize=11)
ax4.set_ylabel(r'$\dot{\theta}(t)$ [rad/s]', fontsize=11)
ax4.set_title('Angular Velocity vs Time', fontsize=12)
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3)

# ======================
# SUBPLOT 5: θ̇ relative error
# ======================

ax5 = plt.subplot(2, 3, 5)

# Compute errors for angular velocity
error_dot_num_analytical = np.abs(dot_theta_num - dot_theta_exact)
error_dot_num_small = np.abs(dot_theta_num - dot_theta_small)

# Plot errors
ax5.semilogy(t_eval, error_dot_num_analytical, 'b-', linewidth=1.5, alpha=0.8, label='Absolute Error (Analytic)')
ax5.semilogy(t_eval, error_dot_num_small, 'g-', linewidth=1.5, alpha=0.7, label='Absolute Error (Small)')

ax5.set_xlabel('Time [s]', fontsize=11)
ax5.set_ylabel(r'Absolute error in $\dot{\theta}$ [rad/s]', fontsize=11)
ax5.set_title('Angular Velocity Errors', fontsize=12)
ax5.legend(loc='lower right', fontsize=9)
ax5.grid(True, alpha=0.3, which='both')

# Add error statistics
max_err_dot_analytical = np.max(error_dot_num_analytical)
max_err_dot_small = np.max(error_dot_num_small)

ax5.text(0.02, 0.02, f'Max errors:\nAnalytical: {max_err_dot_analytical:.2e}\nSmall-angle: {max_err_dot_small:.2e}',
         transform=ax5.transAxes, fontsize=8, va='bottom', ha='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# # ======================
# # SUBPLOT 6: Period comparison
# # ======================

ax6 = plt.subplot(2, 3, 6)

# Prepare data for bar chart
period_values = [T_small]
period_names = ['Small-angle']

if T_exact is not None:
    period_values.append(T_exact)
    period_names.append('Analytical')

if T_num is not None:
    period_values.append(T_num)
    period_names.append('Numerical')
    
colors = ['green', 'blue', 'black']

# Create bars
bars = ax6.bar(period_names, period_values, color=colors, alpha=0.7)

# Annotate each bar with its value
for bar, value in zip(bars, period_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
              f'{value:.4f} s', ha='center', va='bottom', fontsize=9)


# Calculate percentage differences
if T_exact is not None and T_num is not None:
    pct_diff_exact_small = 100 * (T_exact - T_small) / T_small
    pct_diff_num_exact = 100 * (T_num - T_exact) / T_exact

    diff_text = f'Differences:\n'
    diff_text += f'Analytical vs Small: {pct_diff_exact_small:+.1f}%\n'
    diff_text += f'Numerical vs Analytical: {pct_diff_num_exact:+.1f}%'

    ax6.text(0.02, 0.02, diff_text, transform=ax6.transAxes, fontsize=8,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=1.0))

ax6.set_ylabel('Period [s]', fontsize=11)
ax6.set_title('Period Comparison', fontsize=12)
ax6.grid(True, alpha=0.3, axis='y')

# Adjust y-lim to make room for annotations
max_period = max(period_values) if T_num is not None else max(T_small, T_exact)
ax6.set_ylim([0, max_period * 1.15])

plt.tight_layout()
plt.show()





print("  All plots generated successfully!")
print()

# ======================
# 5. ADDITIONAL ANALYSIS
# ======================
print("5. Additional analysis...")

# Check if we have rotational motion
if ε >= 2.0:
    print(f"  Rotational motion detected (ε = {ε:.3f} ≥ 2)")
    if T_num is not None:
        avg_omega = 2 * np.pi / T_num
        print(f"  Average angular velocity: {avg_omega:.4f} rad/s")
        print(f"  Revolutions per second: {avg_omega/(2*np.pi):.4f} Hz")
else:
    print(f"  Oscillatory motion (ε = {ε:.3f} < 2)")
    print(f"  Amplitude: θ_max = {theta_max:.4f} rad ({np.degrees(theta_max):.2f}°)")
    
    theta_max_small = theta_0  # For small angles, amplitude doesn't change much
    print(f"  Small-angle prediction: θ_max ≈ {theta_max_small:.4f} rad")
    print(f"  Amplitude ratio: θ_max/θ_0 = {theta_max/theta_0:.4f}")

# Energy check (dimensionless)
epsilon_num = 0.5 * (dot_theta_num**2) / omega0**2 + (1 - np.cos(theta_num))
epsilon_error = np.max(np.abs(epsilon_num - ε))
print(f"\n  Energy conservation check:")
print(f"    Max deviation in ε: {epsilon_error:.2e}")
print(f"    Relative error: {epsilon_error/ε:.2e}")

# RMS errors
rms_theta_error = np.sqrt(np.mean((theta_num - theta_exact)**2))
rms_dot_theta_error = np.sqrt(np.mean((dot_theta_num - dot_theta_exact)**2))
print(f"  RMS errors:")
print(f"    θ: {rms_theta_error:.2e} rad")
print(f"    dot_θ̇: {rms_dot_theta_error:.2e} rad/s")
    








