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

#------------------------------------------------------------------------------
# 1. Centralized Problem (The Benchmark)
#------------------------------------------------------------------------------
def solve_centralized_problem(data7, occupancy_df):
    model = ConcreteModel()
    model.N = RangeSet(1, 15) # 15 stores
    model.R = RangeSet(1, 2)
    model.T = RangeSet(0, data7['num_timeslots'] - 1)

    model.p = Var(model.N, model.R, model.T, bounds=(0, data7['heating_max_power']))
    model.Temp = Var(model.N, model.R, model.T)

    def obj_rule(m):
        return sum((n + 1) * (m.Temp[n, r, t] - data7['Temperature_reference'])**2
                   for n in m.N for r in m.R for t in m.T)
    model.obj = Objective(rule=obj_rule, sense=minimize)

    def mall_limit_rule(m, t):
        return sum(m.p[n, r, t] for n in m.N for r in m.R) <= data7['P_mall']
    model.limit = Constraint(model.T, rule=mall_limit_rule)

    def dynamics_rule(m, n, r, t):
        if t == 0: return m.Temp[n, r, t] == data7['initial_temperature']
        r_prime = 2 if r == 1 else 1
        return m.Temp[n, r, t] == m.Temp[n, r, t-1] + \
               data7['heat_exchange_coeff']*(m.Temp[n, r_prime, t-1] - m.Temp[n, r, t-1]) - \
               data7['thermal_loss_coeff']*(m.Temp[n, r, t-1] - data7['outdoor_temperature'][t-1]) + \
               data7['heating_efficiency_coeff']*m.p[n, r, t-1] - \
               data7['heat_vent_coeff']*1 + \
               data7['heat_occupancy_coeff']*occupancy_df.iloc[r-1, t-1]
    model.cons = Constraint(model.N, model.R, model.T, rule=dynamics_rule)

    SolverFactory('gurobi').solve(model)
    return value(model.obj)

# Get baseline before starting
print("Solving Centralized Benchmark...")
centralized_optimal_obj = solve_centralized_problem(data7, occupancy)

#------------------------------------------------------------------------------
# 2. Store Sub-Problem (The Distributed Agent)
#------------------------------------------------------------------------------
def solve_store_subproblem(n, lambda_t_current, data7, occupancy_df):
    model = ConcreteModel()
    model.R = RangeSet(1, 2)
    model.T = RangeSet(0, data7['num_timeslots'] - 1)

    # Local Variables
    model.p = Var(model.R, model.T, bounds=(0, data7['heating_max_power']))
    model.Temp = Var(model.R, model.T)

    # Temperature Dynamics (Store-specific)
    def temp_dynamics_rule(m, r, t):
        if t == 0: return m.Temp[r, t] == data7['initial_temperature']
        r_prime = 2 if r == 1 else 1
        return m.Temp[r, t] == m.Temp[r, t - 1] + \
            data7['heat_exchange_coeff'] * (m.Temp[r_prime, t - 1] - m.Temp[r, t - 1]) - \
            data7['thermal_loss_coeff'] * (m.Temp[r, t - 1] - data7['outdoor_temperature'][t - 1]) + \
            data7['heating_efficiency_coeff'] * m.p[r, t - 1] - \
            data7['heat_vent_coeff'] * 1 + \
            data7['heat_occupancy_coeff'] * occupancy_df.iloc[r - 1, t - 1]
    model.temp_cons = Constraint(model.R, model.T, rule=temp_dynamics_rule)

    # Discomfort Calculation (used for System Obj)
    comfort_penalty = sum((n + 1) * (model.Temp[r, t] - data7['Temperature_reference']) ** 2
                          for r in model.R for t in model.T)

    # Local Objective: Discomfort + Multiplier Cost + Tiny Penalty for stability
    model.obj = Objective(expr=comfort_penalty +
                               sum(lambda_t_current[t] * sum(model.p[r, t] for r in model.R) for t in model.T) +
                               sum(model.p[r, t] * 1e-9 for r in model.R for t in model.T),
                          sense=minimize)

    solver = SolverFactory('gurobi')
    results = solver.solve(model)

    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        p_values = [value(sum(model.p[r, t] for r in model.R)) for t in model.T]
        discomfort_value = value(comfort_penalty)
        return p_values, discomfort_value
    else:
        return [0.0] * data7['num_timeslots'], 0.0

#------------------------------------------------------------------------------
# 3. Master Loop: Sensitivity Analysis
#------------------------------------------------------------------------------
alpha_set = [0.001, 0.01, 0.1, 1, 10, 'adaptive']
results = {a: {'obj': [], 'lambda': [], 'viol': []} for a in alpha_set}

for alpha_val in alpha_set:
    print(f"Testing alpha: {alpha_val}")
    lambda_t = np.zeros(data7['num_timeslots'])

    for k in range(100):
        total_p = np.zeros(data7['num_timeslots'])
        current_iter_system_discomfort = 0

        # Step size logic
        step = (5.0 / (1 + k)) if alpha_val == 'adaptive' else alpha_val

        # 1. Solve each store independently
        for n in range(15):
            p_n, discomfort_n = solve_store_subproblem(n, lambda_t, data7, occupancy)
            total_p += np.array(p_n)
            current_iter_system_discomfort += discomfort_n

        # 2. Store Metrics for this iteration
        results[alpha_val]['obj'].append(current_iter_system_discomfort)
        results[alpha_val]['lambda'].append(lambda_t.copy())
        results[alpha_val]['viol'].append(total_p - data7['P_mall'])

        # 3. Update Lagrange Multipliers (Coordinator)
        for t in range(data7['num_timeslots']):
            violation = total_p[t] - data7['P_mall']
            lambda_t[t] = max(0, lambda_t[t] + step * violation)

#------------------------------------------------------------------------------
# 4. Plots
#------------------------------------------------------------------------------

# PLOT 1: System Objective
plt.figure(figsize=(10, 5))
for a in alpha_set:
    plt.plot(results[a]['obj'], label=f'alpha={a}')
plt.axhline(y=centralized_optimal_obj, color='r', linestyle='--', label='Centralized Optimal')
plt.title("System Objective Value (Social Welfare) across Iterations")
plt.xlabel("Iteration (k)"); plt.ylabel("Objective Value"); plt.legend(); plt.show()

# PLOT 2: Multiplier Evolution (using alpha=0.1 as example)
plt.figure(figsize=(10, 5))
lambda_history = np.array(results[0.1]['lambda'])
for t in range(data7['num_timeslots']):
    plt.plot(lambda_history[:, t], label=f'Slot t={t}')
plt.title("Evolution of Lagrange Multipliers (λ_t) for alpha=0.1")
plt.xlabel("Iteration (k)"); plt.ylabel("Price λ_t"); plt.legend(); plt.show()

# PLOT 3: Violations (using alpha=0.1 as example)
plt.figure(figsize=(10, 5))
viol_history = np.array(results[0.1]['viol'])
for t in range(data7['num_timeslots']):
    plt.plot(viol_history[:, t], label=f'Slot t={t}')
plt.axhline(y=0, color='black', linewidth=1)
plt.title("Evolution of Power Violations per Timeslot (alpha=0.1)")
plt.xlabel("Iteration (k)"); plt.ylabel("Violation [kW]"); plt.legend(); plt.show()

# ------------------------------------------------------------------------------
# 5. FINAL PLOT: Power Distribution per Store (Adaptive Alpha, Final Iteration)
# ------------------------------------------------------------------------------

# We will re-run a quick capture for the 'adaptive' case to get individual store data
final_store_power = []  # To store (15 stores x 10 hours)
lambda_final = np.zeros(data7['num_timeslots'])
alpha_0 = 5

# Run the 100 iterations one last time specifically to extract store-level data
for k in range(100):
    total_p_iter = np.zeros(data7['num_timeslots'])
    step = alpha_0 / (1 + k)

    # Temporary list to hold this iteration's store choices
    current_iter_stores = []

    for n in range(15):
        p_n, _ = solve_store_subproblem(n, lambda_final, data7, occupancy)
        total_p_iter += np.array(p_n)
        if k == 99:  # Only save the very last iteration
            current_iter_stores.append(p_n)

    # Update lambda
    for t in range(data7['num_timeslots']):
        lambda_final[t] = max(0, lambda_final[t] + step * (total_p_iter[t] - data7['P_mall']))

    if k == 99:
        final_store_power = np.array(current_iter_stores)

# Plotting
plt.figure(figsize=(12, 6))
time_slots_range = range(data7['num_timeslots'])
bottom_stack = np.zeros(data7['num_timeslots'])

# Use a colormap to distinguish 15 stores
colors = plt.cm.viridis(np.linspace(0, 1, 15))

for n in range(15):
    plt.bar(time_slots_range, final_store_power[n, :], bottom=bottom_stack,
            label=f'Store {n + 1} (w={n + 1})', color=colors[n])
    bottom_stack += final_store_power[n, :]

plt.axhline(y=data7['P_mall'], color='red', linestyle='--', linewidth=2, label='Mall Limit')
plt.title("Final Power Allocation per Store (Adaptive Step Size)", fontsize=14)
plt.xlabel("Time Slot (Hour)", fontsize=12)
plt.ylabel("Power Consumption [kW]", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()