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

# Parameters
with open('truck_trailer_para.yaml', 'r') as file:
    para = yaml.safe_load(file)

L0 = para['truck']['L']
M0 = para['truck']['M']
W0 = para['truck']['W']
L1 = para['trailer1']['L']
M1 = para['trailer1']['M']
W1 = para['trailer1']['W']

theta_amount=8
pos_amount= 8
radius = 1

run_counter='test_3'
loop_result=[]
Results=[]
thetas = np.linspace(-np.pi, np.pi, theta_amount+1)
thetas = thetas[:-1]

pos_angles = np.linspace(-np.pi+0.001, np.pi+0.001, pos_amount+1)
pos_angles = pos_angles[:-1]
coord_values = [[radius*np.cos(pos_angle),radius*np.sin(pos_angle)] for pos_angle in pos_angles]
print(coord_values)

hoek_array=[0., pi/2, pi, (3*pi)/2]
coord_array=[[0.01,1.],[1.,0.],[0.01,-1.],[-1.,0.]]
for hoek in pos_angles:
    for coord in coord_values:
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

        # Solve
        sol = ocp.solve()

        solve_ocp = ocp.to_function('solve_ocp',
                                    [X_0],
                                    [ocp.sample(delta0,grid='control')[1], ocp.sample(v0,grid='control')[1]])

        # Get solution
        ts_ctrl, theta1_ctrl = sol.sample(theta1, grid='control', refine=refine)
        ts_ctrl, x1_ctrl     = sol.sample(x1,     grid='control', refine=refine)
        ts_ctrl, y1_ctrl     = sol.sample(y1,     grid='control', refine=refine)

        ts_ctrl, theta0_ctrl = sol.sample(theta0, grid='control', refine=refine)
        ts_ctrl, x0_ctrl     = sol.sample(x0,     grid='control', refine=refine)
        ts_ctrl, y0_ctrl     = sol.sample(y0,     grid='control', refine=refine)

        ts_ctrl, delta0_ctrl = sol.sample(delta0, grid='control', refine=refine)
        ts_ctrl, v0_ctrl     = sol.sample(v0,     grid='control', refine=refine)

        delta0_test, v0_test =solve_ocp(vertcat(theta1_t0, x1_t0, y1_t0, theta0_t0))

        delta0_test=system_result=casadi_helpers.DM2numpy(delta0_test, [2,1])
        v0_test=casadi_helpers.DM2numpy(v0_test, [2,1])


        sim_system_dyn = ocp.discrete_system()

        Nsim = N
        if use_simulator:
            # -------------------------------
            # Logging variables
            # -------------------------------
            theta1_sim = np.zeros(Nsim+1)
            x1_sim     = np.zeros(Nsim+1)
            y1_sim     = np.zeros(Nsim+1)

            theta0_sim = np.zeros(Nsim+1)
            x0_sim     = np.zeros(Nsim+1)
            y0_sim     = np.zeros(Nsim+1)
            delta0_sim = np.zeros(Nsim+1)
            v0_sim     = np.zeros(Nsim+1)

            theta1_sim[0] = theta1_t0
            x1_sim[0]     = x1_t0
            y1_sim[0]     = y1_t0
            theta0_sim[0] = theta0_t0
            x0_sim[0]     = x0_t0
            y0_sim[0]     = y0_t0

            x_current = vertcat(theta1_t0, x1_t0, y1_t0, theta0_t0)

            simu = simulator_delta_init()

            for k in range(Nsim):
                print(str(k)+"/"+str(Nsim))

                delta0_loop, v0_loop = solve_ocp(vertcat(theta1_sim[k], x1_sim[k], y1_sim[k], theta0_sim[k]))
                delta0_loop = casadi_helpers.DM2numpy(delta0_loop[0], [2,1])
                v0_loop = casadi_helpers.DM2numpy(v0_loop[0], [2,1])

                delta0_sim[k] = delta0_loop
                v0_sim[k]     = v0_loop

                with open('truck_trailer_run_'+str(run_counter)+'.csv', "a+") as output_file:
                    writer = csv.writer(output_file, lineterminator='\n')
                    writer.writerow([x_current[0], x_current[1], x_current[2], x_current[3], delta0_loop, v0_loop])

                u = vertcat(delta0_loop, v0_loop)
                x_next = simulator(simu, x_current, u, Ts)

                theta1_sim[k+1] = x_next[0]
                x1_sim[k+1]     = x_next[1]
                y1_sim[k+1]     = x_next[2]

                theta0_sim[k+1] = x_next[3]
                x0_sim[k+1]     = x_next[1] + L1*cos(x_next[0]) + M0*cos(x_next[3])
                y0_sim[k+1]     = x_next[2] + L1*sin(x_next[0]) + M0*sin(x_next[3])

                x_current = x_next
            
            k= k+1
            delta0_loop, v0_loop = solve_ocp(vertcat(theta1_sim[k], x1_sim[k], y1_sim[k], theta0_sim[k]))
            delta0_loop = casadi_helpers.DM2numpy(delta0_loop[0], [2,1])
            v0_loop = casadi_helpers.DM2numpy(v0_loop[0], [2,1])

            delta0_sim[k] = delta0_loop
            v0_sim[k]     = v0_loop

            with open('truck_trailer_run_'+str(run_counter)+'.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow([x_current[0], x_current[1], x_current[2], x_current[3], delta0_loop, v0_loop])

            result=np.square(x_next[0])+np.square(x_next[1])+np.square(x_next[2])+np.square(x_next[3])
            loop_result.append(result[0][0])
            print("end")



