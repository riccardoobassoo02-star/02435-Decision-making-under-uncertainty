from pathlib import Path
from pyomo.environ import *
import numpy as np
from Utils.v2_SystemCharacteristics import get_fixed_data
import pandas as pd

FILE_DIR = Path(__file__).parent  # directory where this file is located
DATA_DIR = FILE_DIR / 'Data'  # name of the folder containing csv to be imported


# Import data
data = get_fixed_data()
occupancy1_matrix = np.genfromtxt("Data/OccupancyRoom1.csv", delimiter=",", skip_header=1)
occupancy2_matrix = np.genfromtxt("Data/OccupancyRoom2.csv", delimiter=",", skip_header=1)
price_data        = np.genfromtxt("Data/v2_PriceData.csv",   delimiter=",", skip_header=1)

# Join data to index by room
occupancy = [occupancy1_matrix, occupancy2_matrix]
T0        = [data['T1'], data['T2']]

# Rest of data
L        = data["num_timeslots"] # 10 hours
P_max       = data["heating_max_power"] # 3 kW
U_vent      = data["vent_min_up_time"] # 3 hours
P_vent      = data['ventilation_power'] # 2 kW
H0          = data['H'] # initial humidity
zeta_exch   = data['heat_exchange_coeff'] # heat exchange coefficient between rooms
zeta_conv   = data['heating_efficiency_coeff'] # heating efficiency: increase in room temperature per kW of heating power
zeta_loss   = data['thermal_loss_coeff'] # thermal loss coefficient: fraction of indoor-outdoor temperature difference lost per hour    
zeta_cool   = data['heat_vent_coeff'] # ventilation cooling effect: temperature decrease in the room for each hour that ventilation is ON (°C)
zeta_occ    = data['heat_occupancy_coeff'] # occupancy heat gain: temperature increase per hour per person in the room (°C)
T_outside   = data["outdoor_temperature"] # list of external temperatures (°C)
eta_occ     = data['humidity_occupancy_coeff'] # humidity increase per hour per person in the room (%)
eta_vent    = data['humidity_vent_coeff'] # humidity decrease per hour when ventilation is ON (%)
T_low       = data['temp_min_comfort_threshold'] # minimum comfortable temperature threshold (°C)
T_ok        = data['temp_OK_threshold'] # comfortable temperature threshold (°C)
T_high      = data['temp_max_comfort_threshold'] # maximum comfortable temperature threshold (°C)
H_high      = data['humidity_threshold'] # maximum comfortable humidity threshold (%)
M = 1000
M_temp = 50  # big-M constant for temperature
M_hum = 100   # big-M constant for humidity

def solve_MILP(day):
    # Create model
    model = ConcreteModel()

    # Sets
    model.T = RangeSet(0, L - 1)
    model.R = RangeSet(0, 1)

    # Decision variables
    model.p = Var(model.R, model.T, domain=NonNegativeReals, bounds=(0, P_max)) # heating power
    model.v = Var(model.T, domain=Binary) # ventilation ON/OFF (1 if ON, 0 if OFF)

    # Temperature and humidity
    model.temp = Var(model.R, model.T, domain=Reals) # indoor temperature in room r at time time t (°C)
    model.hum  = Var(model.T, domain=NonNegativeReals) # indoor humidity at time t (%)

    # Overrule controller
    model.temp_high = Var(model.R, model.T, domain=Binary) # binary variable indicating if temperature in room r at time t is above the maximum comfort threshold (1 if temp > T_high, 0 otherwise)
    model.temp_low  = Var(model.R, model.T, domain=Binary) # binary variable indicating if temperature in room r at time t is below the minimum comfort threshold (1 if temp < T_low, 0 otherwise)
    model.temp_ok   = Var(model.R, model.T, domain=Binary) # binary variable indicating if temperature in room r at time t is above the "OK" threshold (1 if temp > T_ok, 0 otherwise)
    model.overrule = Var(model.R, model.T, domain=Binary)  # binary variable indicating if the heater's overrule controller is active in room r at time t (1 if active, 0 otherwise)

    # Ventilation startup at time t
    model.s = Var(model.T, domain=Binary) # binary variable indicating ventilation startup at time t (1 if ventilation starts at time t, 0 otherwise)


    # Objective function
    model.obj = Objective(expr=sum(price_data[day, t] * (P_vent * model.v[t] + model.p[0, t] + model.p[1, t]) for t in model.T), sense=minimize)


    # Constraints
    # 1-2. Temperature dynamics - OK
    model.c1 = Constraint(model.R, rule=lambda model, r: model.temp[r, 0] == T0[r]) # initial temperature constraint at time 0
    model.c2 = Constraint(model.R, model.T, rule=lambda model, r, t: (model.temp[r, t] == model.temp[r, t-1] + 
                                                                        zeta_exch * (model.temp[1-r, t-1] - model.temp[r, t-1]) -
                                                                        zeta_loss * (model.temp[r, t-1] - T_outside[t-1]) +
                                                                        zeta_conv * model.p[r, t-1] - 
                                                                        zeta_cool * model.v[t-1] +
                                                                        zeta_occ * occupancy[r][day, t-1]) 
                                                                        if t > 0 else Constraint.Skip)

    # 3-4. Humidity dynamics - OK
    model.c3 = Constraint(expr=model.hum[0] == H0) # initial humidity constraint at time 0
    model.c4 = Constraint(model.T, rule=lambda model, t: 
                                        (model.hum[t] == model.hum[t-1] + 
                                        eta_occ * sum(occupancy[r][day, t-1] for r in model.R) - 
                                        eta_vent * model.v[t-1]) if t > 0 else Constraint.Skip)

    # 5-6. Temperature cutoff and heater deactivation: - OK
    model.c5 = Constraint(model.R, model.T, rule=lambda model, r, t: model.temp[r, t] >= T_high - M_temp *(1-model.temp_high[r, t]))
    model.c6 = Constraint(model.R, model.T, rule=lambda model, r, t: model.temp[r, t] <= T_high + M_temp * model.temp_high[r, t])

    # 7. Overrule controller forcing heater to zero: - OK
    model.c7 = Constraint(model.R, model.T, rule=lambda model, r, t: model.p[r, t] <= P_max * (1 - model.temp_high[r, t]))

    # 8-9. Detecting when Temperature is below threshold - OK
    model.c8 = Constraint(model.R, model.T, rule=lambda model, r, t: model.temp[r, t] <= T_low + M_temp * (1 - model.temp_low[r, t]))
    model.c9 = Constraint(model.R, model.T, rule=lambda model, r, t: model.temp[r, t] >= T_low - M_temp * model.temp_low[r, t])

    # 10-11. Detecting when Temperature is above the “OK” threshold - OK
    model.c10 = Constraint(model.R, model.T, rule=lambda model, r, t: model.temp[r, t] >= T_ok - M_temp * (1 - model.temp_ok[r, t]))
    model.c11 = Constraint(model.R, model.T, rule=lambda model, r, t: model.temp[r, t] <= T_ok + M_temp * model.temp_ok[r, t])

    # 12-13. Triggering Overrule (only) if Temperature is low - OK
    model.c12 = Constraint(model.R, model.T, rule=lambda model, r, t: model.overrule[r, t] >= model.temp_low[r, t])
    model.c13 = Constraint(model.R, model.T, rule=lambda model, r, t: model.overrule[r, t] <= model.overrule[r, t-1] + model.temp_low[r, t] if t > 0 else Constraint.Skip)

    # 14. Overrule controller forcing heater to maximum - OK
    model.c14 = Constraint(model.R, model.T, rule=lambda model, r, t: model.p[r, t] >= P_max * model.overrule[r, t])

    # 15-16. De-activating Overrule (only) if Temperature is above the “OK” level
    model.c15 = Constraint(model.R, model.T, rule=lambda model, r, t: model.overrule[r, t] >= model.overrule[r, t-1] - model.temp_ok[r, t] if t > 0 else Constraint.Skip)
    model.c16 = Constraint(model.R, model.T, rule=lambda model, r, t: model.overrule[r, t] <= 1 - model.temp_ok[r, t] if t > 0 else Constraint.Skip)

    # 17-20. Ventilation Startup and Minimum Up-Time
    valid_tau = {}
    for t in model.T:
        tau_max = min(t + U_vent - 1, L - 1)
        valid_tau[t] = list(range(t, tau_max + 1))

    model.c17 = Constraint(model.T, rule=lambda model, t: 
    (model.s[t] >= model.v[t]) if t == 0 else (model.s[t] >= model.v[t] - model.v[t-1]))
    model.c18 = Constraint(model.T, rule=lambda model, t: model.s[t] <= model.v[t])
    model.c19 = Constraint(model.T, rule=lambda model, t: model.s[t] <= 1 - model.v[t-1] if t > 0 else Constraint.Skip)
    model.c20 = Constraint(model.T, rule=lambda model, t: sum(model.v[tau] for tau in valid_tau[t]) >= min(U_vent, L - t) * model.s[t])

    # 21. Humidity-Triggered Ventilation - OK
    model.c21 = Constraint(model.T, rule=lambda model, t: model.hum[t] <= H_high + M_hum * model.v[t])


    # Solve model
    solver = SolverFactory('gurobi')
    result = solver.solve(model)

    # Extract results
    p_opt  = {(r,t): value(model.p[r,t])    for r in model.R for t in model.T}
    v_opt  = {t: value(model.v[t])           for t in model.T}
    temp_opt = {(r,t): value(model.temp[r,t]) for r in model.R for t in model.T}
    hum_opt  = {t: value(model.hum[t])        for t in model.T}

    total_cost = value(model.obj)

    return p_opt, v_opt, temp_opt, hum_opt, total_cost 

# initialize empty list to store daily costs and results
daily_costs = []
all_rows = [] # create empty list to store results for all days

# solve the optimization problem for each day and store results
for day in range(100):
    p_opt, v_opt, temp_opt, hum_opt, cost = solve_MILP(day)
    daily_costs.append(cost)
    print(f"Day {day+1}: {cost:.2f}")

    for t in range(L):
        row = {
                'Day': day + 1,
                'Hour': t,
                'Price': price_data[day,t],
                'Occupancy_R1': occupancy[0][day,t],
                'Occupancy_R2': occupancy[1][day,t],
                'Temp_Room1': temp_opt[0, t],
                'Temp_Room2': temp_opt[1, t],
                'Power_Heater1': p_opt[0, t],
                'Power_Heater2': p_opt[1, t],
                'Ventilation_On': v_opt[t],
                'Humidity': hum_opt[t],
        }
        all_rows.append(row)

results_df = pd.DataFrame(all_rows)
OUTPUT_DIR = FILE_DIR/ "results" 
OUTPUT_DIR.mkdir(exist_ok=True)
results_df.to_csv(OUTPUT_DIR / 'Results_OIH.csv', index=False)
print("Results saved to Results_OIH.csv")

# out of the for loop, calculates and prints the average daily cost over the 100 days
average_cost = np.mean(daily_costs)
print(f"Average daily electricity cost: {average_cost:.2f}") 
total_cost = np.sum(daily_costs)
print(f"Total cost over the 100 days: {total_cost:.2f}")

