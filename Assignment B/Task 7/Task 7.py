# -*- coding: utf-8 -*-
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

# 1. Centralized Problem (The Benchmark)
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

# 2. Store Sub-Problem (The Distributed Agent)
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

    # Discomfort Calculation (used for System Objective)
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

# 3. Sensitivity Analysis
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

# 4. Plots

# PLOT 1: System Objective
print("\n--- Final Objective Values after 100 iterations ---")
for a in alpha_set:
    print(f"  alpha={a}: {results[a]['obj'][-1]:.4f}")
print(f"  Centralized Optimal: {centralized_optimal_obj:.4f}")
plt.figure(figsize=(11, 5))
for a in alpha_set:
    final_obj_val = results[a]['obj'][-1]
    plt.plot(results[a]['obj'], label=f'alpha={a} (final obj={final_obj_val:.1f})')
plt.axhline(y=centralized_optimal_obj, color='r', linestyle='--', label=f'Centralized Optimal ({centralized_optimal_obj:.1f})')
plt.title("System Objective Value (Social Welfare) across Iterations")
plt.xlabel("Iteration (k)"); plt.ylabel("Objective Value"); plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(FILE_DIR / 'plot1_objective.png', dpi=150)
plt.show()

# PLOT 2: Multiplier Evolution
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for idx, a in enumerate(alpha_set):
    lambda_history = np.array(results[a]['lambda'])
    for t in range(data7['num_timeslots']):
        axes[idx].plot(lambda_history[:, t], label=f't={t}')
    axes[idx].set_title(f"Multipliers λ_t  (alpha={a})")
    axes[idx].set_xlabel("Iteration (k)")
    axes[idx].set_ylabel("Price λ_t")
    axes[idx].legend(fontsize=6, ncol=2)
plt.suptitle("Evolution of Lagrange Multipliers (λ_t) — All Step Sizes", fontsize=13)
plt.tight_layout()
plt.savefig(FILE_DIR / 'plot2_lambdas.png', dpi=150)
plt.show()

# PLOT 3: Constraint Violations
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for idx, a in enumerate(alpha_set):
    viol_history = np.array(results[a]['viol'])
    for t in range(data7['num_timeslots']):
        axes[idx].plot(viol_history[:, t], label=f't={t}')
    axes[idx].axhline(y=0, color='black', linewidth=1)
    axes[idx].set_title(f"Power Violations  (alpha={a})")
    axes[idx].set_xlabel("Iteration (k)")
    axes[idx].set_ylabel("Violation [kW]")
    axes[idx].legend(fontsize=6, ncol=2)
plt.suptitle("Evolution of Power Violations per Timeslot — All Step Sizes", fontsize=13)
plt.tight_layout()
plt.savefig(FILE_DIR / 'plot3_violations.png', dpi=150)
plt.show()

# 5. FINAL PLOT: Energy per Store — Centralized vs Distributed (Adaptive Alpha)

final_store_power = []
lambda_final = np.zeros(data7['num_timeslots'])
alpha_0 = 5

for k in range(100):
    total_p_iter = np.zeros(data7['num_timeslots'])
    step = alpha_0 / (1 + k)
    current_iter_stores = []
    for n in range(15):
        p_n, _ = solve_store_subproblem(n, lambda_final, data7, occupancy)
        total_p_iter += np.array(p_n)
        if k == 99:
            current_iter_stores.append(p_n)
    for t in range(data7['num_timeslots']):
        lambda_final[t] = max(0, lambda_final[t] + step * (total_p_iter[t] - data7['P_mall']))
    if k == 99:
        final_store_power = np.array(current_iter_stores)

energy_dist = np.sum(final_store_power, axis=1)

def solve_centralized_with_power(data7, occupancy_df):
    model = ConcreteModel()
    model.N = RangeSet(1, 15)
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
    energy_cent = np.array([
        sum(value(model.p[n, r, t]) for r in [1, 2] for t in range(data7['num_timeslots']))
        for n in range(1, 16)
    ])
    return energy_cent

print("Re-solving centralized to extract per-store energy...")
energy_cent = solve_centralized_with_power(data7, occupancy)

weights      = np.array([n + 1 for n in range(1, 16)])
store_labels = [f'S{n}' for n in range(1, 16)]
x_pos        = np.arange(15)

plt.figure(figsize=(14, 5))
plt.bar(x_pos - 0.2, energy_cent, 0.4, label='Centralized', color='orange', alpha=0.85)
plt.bar(x_pos + 0.2, energy_dist, 0.4, label='Adaptive α₀=5 (distributed)', color='steelblue', alpha=0.85)
plt.xticks(x_pos, store_labels, fontsize=9)
plt.ylabel('Total Energy Consumed (kWh)', fontsize=12)
plt.title('Energy per Store: Centralized vs Distributed (Adaptive α₀=5)', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FILE_DIR / 'plot4_energy_per_store.png', dpi=150)
plt.show()