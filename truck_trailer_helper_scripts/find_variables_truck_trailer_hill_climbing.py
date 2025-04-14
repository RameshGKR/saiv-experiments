"""
Motion planning
===============

Simple motion planning for vehicle with trailer
"""

from truck_trailer_helper_scripts.run_one_truck_trailer_simulation import run_one_simulation
from numpy import pi
import copy

def run_all_simulations(variables,run_counter):
    loop_weight=0
    loop_results=[]
    hoek_array=[0., pi/2, pi, (3*pi)/2]
    coord_array=[[0.01,1.],[1.,0.],[0.01,-1.],[-1.,0.]]
    for hoek in hoek_array:
        for coord in coord_array:
            try:
                sim_weight=run_one_simulation(coord, hoek, variables, run_counter)
            except:
                sim_weight=1000000000

            loop_weight=loop_weight+sim_weight
            loop_results.append(sim_weight)

    return loop_weight, loop_results

def get_all_combs(variables, stepsize):
    all_combs=[]

    for i in range(len(variables)):
        variables_plus=copy.deepcopy(variables)
        variables_plus[i]=variables_plus[i]+stepsize
        all_combs.append(variables_plus)
        variables_min=copy.deepcopy(variables)
        variables_min[i]=variables_min[i]-stepsize
        all_combs.append(variables_min)

    return all_combs

def do_hill_climbing():
    original_weight = 1000000
    current_variables=[5,1,2,0.3,1,2]
    Results=[]
    for i in range(100):
        changes=0
        all_combs=get_all_combs(current_variables,0.1)

        for new_comb in all_combs:
            loop_weight, loop_results=run_all_simulations(new_comb,i)
            Results.append(loop_results)

            if loop_weight<original_weight:
                changes=changes+1
                original_weight=loop_weight
                current_variables=new_comb

        if changes == 0:
            print(i)
            print(current_variables)
            print(original_weight)
            print(Results)
            break
    
if __name__ == "__main__":
    do_hill_climbing()