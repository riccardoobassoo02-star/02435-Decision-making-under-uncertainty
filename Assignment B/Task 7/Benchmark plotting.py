import pandas as pd
from pyomo.environ import *
import numpy as np
from Data.DataTask7 import fetch_data
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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
    return model

# Get baseline before starting
print("Solving Centralized Benchmark...")
centralized_optimal_obj = solve_centralized_problem(data7, occupancy)

def plot_model_results(model, data7):
    # 1. Extract Data into a structured format
    temp_results = []
    power_results = []

    for n in model.N:
        for r in model.R:
            for t in model.T:
                temp_results.append({
                    'Store': f"Store {n}",
                    'Room': f"Room {r}",
                    'Time': t,
                    'Temperature': value(model.Temp[n, r, t]),
                    'Power': value(model.p[n, r, t])
                })

    df = pd.DataFrame(temp_results)

    # 2. Setup Figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Plot 1: Temperature Trajectories ---
    sns.lineplot(data=df, x='Time', y='Temperature', hue='Store', style='Room', ax=axes[0])
    axes[0].axhline(y=data7['Temperature_reference'], color='r', linestyle='--', label='Reference Temp')
    axes[0].set_title("Temperature Profiles per Store/Room")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # --- Plot 2: Power Consumption vs Mall Limit ---
    # Aggregate power across all stores/rooms per time step
    total_power = df.groupby('Time')['Power'].sum()

    axes[1].fill_between(total_power.index, total_power.values, color='skyblue', alpha=0.4, label='Total Power Usage')
    axes[1].step(total_power.index, total_power.values, where='post', color='blue', lw=1.5)
    axes[1].axhline(y=data7['P_mall'], color='red', linestyle='-', label='Mall Power Limit ($P_{mall}$)')

    axes[1].set_title("Aggregate Power Consumption")
    axes[1].set_ylabel("Power (kW)")
    axes[1].set_xlabel("Timeslot")
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def save_model_to_csv(model, filename="centralized_model_results.csv"):
    """
    Extracts power and temperature results from the Pyomo model
    and saves them to a structured CSV file.
    """
    print(f"Exporting results to {filename}...")

    results_data = []

    # Loop through the sets defined in your model
    for n in model.N:
        for r in model.R:
            for t in model.T:
                results_data.append({
                    'Store_ID': n,
                    'Room_ID': r,
                    'Timeslot_Hour': t,
                    'Power_kW': value(model.p[n, r, t]),
                    'Temperature_C': value(model.Temp[n, r, t])
                })

    # Convert to DataFrame
    df_results = pd.DataFrame(results_data)

    # Save to CSV
    df_results.to_csv(filename, index=False)

    print("Export complete.")
    return df_results

model = solve_centralized_problem(data7, occupancy)
plot_model_results(model, data7)
df = save_model_to_csv(model, "benchmark_results.csv")