"""
Motion planning
===============

Simple motion planning for vehicle with trailer
"""

from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat
from plot_trailer import *
from simulator import *
import yaml
import csv

show_figures = False
use_simulator = True
save_for_gif = False


T_end=70.0
ocp = Ocp(T=T_end)
N = 100
M = 1
refine = 2
Ts = T_end/N

L0 = 0.3375
M0 = 0.1
W0 = 0.2
L1 = 0.3
M1 = 0.06
W1 = 0.2



x1_t0 = coord[0]
y1_t0 = coord[1]
theta1_t0 = hoek
theta0_t0 = hoek
x0_t0 = x1_t0 + L1*cos(theta1_t0) + M0*cos(theta0_t0)
y0_t0 = y1_t0 + L1*sin(theta1_t0) + M0*sin(theta0_t0)

x1_tf = 0.
y1_tf = 0.
theta1_tf = 0.
theta0_tf = 0.

# Trailer model
theta1 = ocp.state()
x1     = ocp.state()
y1     = ocp.state()

theta0 = ocp.state()
x0     = x1 + L1*cos(theta1) + M0*cos(theta0)
y0     = y1 + L1*sin(theta1) + M0*sin(theta0)

delta0 = ocp.control(order=1)
v0     = ocp.control(order=1)

X_0 = ocp.parameter(4);

beta01 = theta0 - theta1

dtheta0 = v0/L0*tan(delta0)
dtheta1 = v0/L1*sin(beta01) - M0/L1*cos(beta01)*dtheta0
v1 = v0*cos(beta01) + M0*sin(beta01)*dtheta0

ocp.set_der(theta1, dtheta1)
ocp.set_der(x1,     v1*cos(theta1))
ocp.set_der(y1,     v1*sin(theta1))

ocp.set_der(theta0, dtheta0)

X = vertcat(theta1, x1, y1, theta0)#, x0, y0)

# Initial constraints
ocp.subject_to(ocp.at_t0(X) == X_0)
#ocp.subject_to(ocp.at_tf(X) == vertcat(theta1_tf, x1_tf, y1_tf, theta0_tf))

# Final constraint
ocp.subject_to(ocp.at_tf(x1) == x1_tf)
ocp.subject_to(ocp.at_tf(y1) == y1_tf)
ocp.subject_to(ocp.at_tf(theta1) == theta1_tf)
ocp.subject_to(ocp.at_tf(beta01) == theta0_tf - theta1_tf)

# # Initial guess
# ocp.set_initial(theta0, .1)
# ocp.set_initial(theta1, 0)
# ocp.set_initial(v0,    -.2)
# ocp.set_initial(x1,     np.linspace(x1_t0, x1_tf, N))
# ocp.set_initial(y1,     np.linspace(y1_t0, y1_tf, N))

# Path constraints
ocp.subject_to(-.2 <= (v0 <= .2))
ocp.subject_to(-1 <= (ocp.der(v0) <= 1))

ocp.subject_to(-pi/6 <= (delta0 <= pi/6))
ocp.subject_to(-pi/10 <= (ocp.der(delta0) <= pi/10))

ocp.subject_to(-pi/2 <= (beta01 <= pi/2))

# Minimal time
#ocp.add_objective(ocp.T)
#ocp.add_objective(ocp.integral(beta01**2))
#ocp.add_objective(ocp.integral(1.001**ocp.t))
#ocp.add_objective(ocp.integral((delta0**2+v0**2)))
ocp.add_objective(ocp.integral(5*((x1-x1_tf)**2+(y1-y1_tf)**2)*((ocp.t/T_end)**2)))
ocp.add_objective(ocp.integral(0.3*((theta1-theta1_tf)**2+(theta0-theta0_tf)**2)*((ocp.t/T_end)**2)))

# Pick a solution method
options = { "expand": True,
            "verbose": False,
            "print_time": False,
            "error_on_fail": True,
            "ipopt": {	#"linear_solver": "ma57",
                        "tol": 1e-8,
                        "print_level": 0}}
ocp.solver('ipopt',options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N,M=M,intg='rk'))

ocp.set_value(X_0, vertcat(theta1_t0, x1_t0, y1_t0, theta0_t0))#, L1+M0, 0))

solve_ocp = ocp.to_function('solve_ocp',
                            [X_0],
                            [ocp.sample(delta0,grid='control')[1], ocp.sample(v0,grid='control')[1]])


sim_system_dyn = ocp.discrete_system()
