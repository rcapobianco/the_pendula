#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Double pendulum simulation with numerical integration.

This script solves the nonlinear double pendulum using
solve_ivp and produces time series plots, and energy 
conservation diagnostics.

Part of the project: The Pendula
https://rcapobianco.github.io/notes/the_pendula/

Author: Rogerio Capobianco
Created: 2025-12-17
Repository: the_pendula
"""

## Double Pendulum

# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


# Constants
g = 9.81  # gravity (m/s^2)
l1 = 1.0  # length of pendulum 1 (m)
l2 = 1.0  # length of pendulum 2 (m)
m1 = 1.0  # mass of pendulum 1 (kg)
m2 = 1.0  # mass of pendulum 2 (kg)



# Initial conditions
theta1_0 = np.pi / 2  # initial angle for m1 (radians)
theta2_0 = np.pi / 2  # initial angle for m2 (radians)
omega1_0 = 0.0        # initial angular velocity for m1 (rad/s)
omega2_0 = 0.0        # initial angular velocity for m2 (rad/s)

tmax = 20 ##Integration time

# System of equations for the double pendulum
def double_pendulum(t, Y):
    theta1, theta2, omega1, omega2 = Y

    delta = theta1 - theta2
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    # denominator
    den = m1 + m2 * sin_delta**2

    # velocity
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    domega1_dt = ( - m2 * sin_delta * ( l2 * omega2**2 + l1 * omega1**2 * cos_delta)
        - (m1 + m2) * g * np.sin(theta1) + m2 * g * np.sin(theta2) * cos_delta) / (l1 * den)

    domega2_dt = ( sin_delta*(m2*l2*cos_delta*omega2**2 + (m1 + m2)*l1*omega1**2 )  +
        (m1 + m2) * (g * np.sin(theta1) * cos_delta - g * np.sin(theta2) )) / (l2 * den)

    return [dtheta1_dt, dtheta2_dt, domega1_dt, domega2_dt]





Y0 = [theta1_0, theta2_0, omega1_0, omega2_0] ## Initial conditions

# Time span for the solution
t_span = (0, tmax)  # from t=0 to t=20  (s)
t_eval = np.linspace(0, tmax, 1000)  # integration grid

# Solve the system of ODEs
sol = solve_ivp(double_pendulum, t_span, Y0, t_eval=t_eval, method='DOP853', 
                rtol=1e-12, atol=1e-12
                )

## Methods: RK45 and DOP853

print("Integration reached t_max?", sol.success)
print("Solution shape:", sol.y.shape)



t = sol.t 

theta1, theta2, omega1, omega2 = sol.y 

# %%

## Plots: Velocity (theta1 and theta2)

fig, (ax_theta, ax_omega) = plt.subplots( nrows=2, ncols=1, figsize= (8,6), sharex = True )

## Positions theta_s (in rad)
ax_theta.plot(t, theta1, "r-" ,label=r'$\theta_1$ (pendulum 1)')
ax_theta.plot(t, theta2, "b-" ,label=r'$\theta_2$ (pendulum 2)')

ax_theta.set_ylabel("Angle [rad]")
ax_theta.legend()
ax_theta.grid()
 
## Velocities omega_s (in rad/s)
ax_omega.plot(t, omega1, "r-" , label=r'$\omega_1$')
ax_omega.plot(t, omega2, "b-" , label=r'$\omega_2$')

ax_omega.set_ylabel("Angular velocity [rad/s]")
ax_omega.set_xlabel("time [s]")
ax_omega.legend()
ax_omega.grid()

fig.tight_layout()
plt.show()

## fig.savefig("double_pendulum_timeseries.png", dpi=300)



# %%

## Conservation of energy

delta = theta1 - theta2

# Kinetic energy
T = (0.5 * m1 * (l1 * omega1)**2
    + 0.5 * m2 * (
        (l1 * omega1)**2
        + (l2 * omega2)**2
        + 2 * l1 * l2 * omega1 * omega2 * np.cos(delta)
    )
)

# Potential energy
V = (- (m1 + m2) * g * l1 * np.cos(theta1)
    - m2 * g * l2 * np.cos(theta2)
)

# Total energy
E = T + V

# Relative energy error
E0 = E[0]
rel_err = (E - E0) / abs(E0)

# ---- Plotting ----

fig_E, (ax_E, ax_dE) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(8, 6),
    sharex=True
)

# Total energy
ax_E.plot(t, E, 'k-', lw=1)
ax_E.set_ylabel("Total energy")
ax_E.set_title("Energy conservation: double pendulum")
ax_E.grid(True)

# Relative energy error
ax_dE.plot(t, rel_err, 'r-', lw=1)
ax_dE.set_xlabel("time [s]")
ax_dE.set_ylabel(r"$(E - E_0)/|E_0|$")
ax_dE.grid(True)

fig_E.tight_layout()
plt.show()







































