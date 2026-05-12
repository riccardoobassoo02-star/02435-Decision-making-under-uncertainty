# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 12:30:18 2025

@author: geots
"""
import pandas as pd
from pyomo.environ import *
import numpy as np
from Data.DataTask7 import fetch_data
from pathlib import Path
import matplotlib.pyplot as plt


#Obtained data from the file Data7 in order to create a dictionary with it.
FILE_DIR = Path(__file__).parent.parent  # directory where this file is located
DATA_DIR = FILE_DIR / 'Data'  # name of the folder containing csv to be imported
occupancy = pd.read_csv(DATA_DIR / "Task7Occupancies.csv")


data7 = fetch_data()

time_slots = data7['num_timeslots']
p_mall = data7['P_mall']
T_ref = data7['Temperature_reference']
T_initial = data7['initial_temperature']
heater_max_power = data7['heating_max_power'] # Maximum heating power (kW)
exchange_coeff = data7['heat_exchange_coeff'] # Heat exchange coefficient between rooms
heater_efficiency = data7['heating_efficiency_coeff'] # Heating efficiency:
thermal_loss_coeff = data7['thermal_loss_coeff'] # Fraction of indoor-outdoor temperature difference lost per hour
heater_vent_coeff= data7['heat_vent_coeff']# Ventilation cooling effect: # Temperature decrease in the room for each hour that ventilation is ON (°C)
heat_occupancy_coeff = data7['heat_occupancy_coeff'] # Occupancy heat gain: # Temperature increase per hour per person in the room (°C)
T_outdoor = data7['outdoor_temperature']# Outdoor temperature (°C)


#----------------------------------------------------------------------------

#Sub section : Each Store ssolves their own sub-optimization problem

#----------------------------------------------------------------------------
def solve_store_subproblem(n, lambda_t_current, data7, occupancy_df):
    model = ConcreteModel()

    model.R = RangeSet(1, 2)
    model.T = RangeSet(0, data7['num_timeslots'] - 1)

    # Variables
    model.p = Var(model.R, model.T, bounds=(0, data7['heating_max_power']))
    model.Temp = Var(model.R, model.T)

    # 1. Objective: Minimize (Discomfort + Power Tax)
    def obj_rule(m):
        w_n = n + 1
        # Comfort Penalty
        comfort = sum(w_n * (m.Temp[r, t] - data7['Temperature_reference']) ** 2
                      for r in m.R for t in m.T)
        # Coordination Tax (Dual variable logic)
        tax = sum(lambda_t_current[t] * sum(m.p[r, t] for r in m.R)
                  for t in m.T)

        # ADD THIS: A tiny penalty to ensure p[r,9] is initialized to 0
        tiny_penalty = sum(m.p[r, t] * 1e-9 for r in m.R for t in m.T)

        return comfort + tax + tiny_penalty

    model.obj = Objective(rule=obj_rule, sense=minimize)

    # 2. Temperature Dynamics
    def temp_dynamics_rule(m, r, t):
        if t == 0:
            return m.Temp[r, t] == data7['initial_temperature']

        r_prime = 2 if r == 1 else 1

        # NOTE: t-1 ensures we use the state/action from the previous hour
        # to define the temperature at the current hour 't'
        return m.Temp[r, t] == m.Temp[r, t - 1] + \
            data7['heat_exchange_coeff'] * (m.Temp[r_prime, t - 1] - m.Temp[r, t - 1]) - \
            data7['thermal_loss_coeff'] * (m.Temp[r, t - 1] - data7['outdoor_temperature'][t - 1]) + \
            data7['heating_efficiency_coeff'] * m.p[r, t - 1] - \
            data7['heat_vent_coeff'] * 1 + \
            data7['heat_occupancy_coeff'] * occupancy_df.iloc[r - 1, t - 1]

    model.temp_cons = Constraint(model.R, model.T, rule=temp_dynamics_rule)

    # 3. Solver and Error Handling
    solver = SolverFactory('gurobi')
    results = solver.solve(model)

    # Check if the solver actually found a solution
    if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
        return [value(sum(model.p[r, t] for r in model.R)) for t in model.T]
    else:
        # If infeasible, this prints the reason so you can debug the physics
        print(f"--- Store {n} Solver Error ---")
        print(f"Status: {results.solver.status}")
        print(f"Condition: {results.solver.termination_condition}")

        # Return zeros so the Master Loop doesn't crash, allowing you to see the error
        return [0.0 for t in model.T]

# -----------------------------------------------------------------------------------

# Coordinator simulation ( k = 100 iterations / alpha = {0.001....., 10}

# -----------------------------------------------------------------------------------

N_stores = 15
alpha_0 = 0.1
lambda_t = np.zeros(data7['num_timeslots'])

# Containers for plotting
power_history = []
violation_history = []

for k in range(5):
    alpha_k = alpha_0 / (1 + k)
    total_mall_power = np.zeros(data7['num_timeslots'])

    for n in range(N_stores):
        p_n_schedule = solve_store_subproblem(n, lambda_t, data7, occupancy)
        total_mall_power += np.array(p_n_schedule)

    # Calculate violation: Total - Limit
    current_violation = total_mall_power - data7['P_mall']

    # SAVE results for plotting
    power_history.append(total_mall_power.copy())
    violation_history.append(current_violation.copy())

    for t in range(data7['num_timeslots']):
        lambda_t[t] = max(0, lambda_t[t] + alpha_k * current_violation[t])

    max_violation = max(current_violation)
    print(f"Iteration {k}: Max Power Over-Limit = {max_violation:.2f} kW")

#----------------------------------------------------------------------------------

#Plotting Module

#----------------------------------------------------------------------------------

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Total Mall Power vs Limit (Final Iteration)
time_axis = range(data7['num_timeslots'])
ax1.step(time_axis, power_history[-1], where='post', label='Aggregated Power (Final Iter)', color='blue', linewidth=2)
ax1.axhline(y=data7['P_mall'], color='red', linestyle='--', label='Mall Power Limit ($P^{mall}$)')
ax1.set_title("Aggregated Mall Power Consumption", fontsize=12)
ax1.set_xlabel("Time Slots (t)", fontsize=10)
ax1.set_ylabel("Power [kW]", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Evolution of Violations across Iterations
for k_idx, viol in enumerate(violation_history):
    ax2.plot(time_axis, viol, label=f'Iteration {k_idx}', marker='o', markersize=4)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1) # The 0-line represents no violation
ax2.set_title("Evolution of Mall Limit Violations", fontsize=12)
ax2.set_xlabel("Time Slots (t)", fontsize=10)
ax2.set_ylabel("Violation ($\sum p_{n,t} - P^{mall}$) [kW]", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize='small', ncol=2)

plt.tight_layout()
plt.show()