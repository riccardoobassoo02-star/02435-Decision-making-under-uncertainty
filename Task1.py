# imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from pyomo.environ import * 
from pathlib import Path
from SystemCharacteristics import get_fixed_data 

# load data 
FILE_DIR = Path(__file__).parent  # directory where this file is located
DATA_DIR = FILE_DIR / 'data'  # name of the folder containing csv to be imported
price_data = pd.read_csv(DATA_DIR / 'PriceData.csv') 
occupancy_r1 = pd.read_csv(DATA_DIR / "OccupancyRoom1.csv")
occupancy_r2 = pd.read_csv(DATA_DIR / "OccupancyRoom2.csv")

# parameters extraction from system characteristics
data = get_fixed_data()
T           = data['num_timeslots'] # number of hours (10)
T0_prev = data['previous_initial_temperature'] # every day's initial temperature
T0          = data['initial_temperature'] # initial indoor temperature (°C)
H0          = data['initial_humidity'] # initial indoor humidity (%)
P_max       = data['heating_max_power'] # maximum heating power (kW)
zeta_exch   = data['heat_exchange_coeff'] # heat exchange coefficient between rooms
zeta_conv   = data['heating_efficiency_coeff'] # heating efficiency: increase in room temperature per kW of heating power
zeta_loss   = data['thermal_loss_coeff'] # thermal loss coefficient: fraction of indoor-outdoor temperature difference lost per hour    
zeta_cool   = data['heat_vent_coeff'] # ventilation cooling effect: temperature decrease in the room for each hour that ventilation is ON (°C)
zeta_occ    = data['heat_occupancy_coeff'] # occupancy heat gain: temperature increase per hour per person in the room (°C)
T_low       = data['temp_min_comfort_threshold'] # minimum comfortable temperature threshold (°C)
T_ok        = data['temp_OK_threshold'] # comfortable temperature threshold (°C)
T_high      = data['temp_max_comfort_threshold'] # maximum comfortable temperature threshold (°C)
T_out       = data['outdoor_temperature'] # outdoor temperature (°C)
P_vent      = data['ventilation_power'] # power consumption of ventilation system when ON (kW)
H_high      = data['humidity_threshold'] # maximum comfortable humidity threshold (%)
eta_occ     = data['humidity_occupancy_coeff'] # humidity increase per hour per person in the room (%)
eta_vent    = data['humidity_vent_coeff'] # humidity decrease per hour when ventilation is ON (%)
min_up_time = data['vent_min_up_time'] # minimum number of consecutive hours that ventilation must be ON once turned ON (hours)
M = 1000 # big M constant for linearization of logical conditions in the overrule controllers
epsilon = 0.01 # small value to express the strict inequality for the controllers rules

# definition of the optimization model
def solve_milp(price,occ_r1,occ_r2):  
    model = ConcreteModel() # creating an instance of the concrete model class 

    # sets
    model.T = RangeSet(0,T-1) # set of time periods (0 to T-1) 
    model.R = RangeSet(1,2) # set of rooms (1 and 2) 

    # decision variables 
    model.p = Var(model.R,model.T, within=NonNegativeReals, bounds=(0,P_max)) # heating power at each time period (kW) 
    model.v = Var(model.T, within=Binary) # ventilation ON/OFF at each time period (1 if ON, 0 if OFF) 

    # state variables 
    model.temp = Var(model.R, model.T, within=Reals) # temperature in each room at each time period (°C)
    model.hum  = Var(model.T, within=NonNegativeReals) # humidity at each time period (%)
     
    # auxiliary variables for controllers (state variables)
    model.delta_low  = Var(model.R, model.T, within=Binary) # 1 when the low temperature overrule controller is activated, 0 otherwise
    model.delta_high = Var(model.R, model.T, within=Binary) # 1 when the high temperature overrule controller is activated, 0 otherwise
    model.delta_hum  = Var(model.T, within=Binary) # 1 when the humidity overrule controller is activated, 0 otherwise

    # objective function - minimize total cost of heating and ventilation
    model.obj = Objective(expr = sum (price[t] * (model.p[1,t] + model.p[2,t] + P_vent * model.v[t]) for t in model.T), sense=minimize)

    # constraints 
    # 1. initial conditions for temperature
    model.temp_init = ConstraintList() 
    for r in model.R: 
        model.temp_init.add(model.temp[r,0] == T0) # initial temperature at time 0 is T0, for every room and for every day 

    # 2. initial conditions for humidity
    model.hum_init = ConstraintList() 
    model.hum_init.add(model.hum[0] == H0) # initial humidity at time 0 is H0, for every day  

    # 3. temperature dynamics for each room and time period
    model.temp_dyn = ConstraintList() 
    for t in model.T: 
        for r in model.R:
            if t > 0:
                occ = occ_r1[t-1] if r == 1 else occ_r2[t-1]
                model.temp_dyn.add(
                    model.temp[r,t] == model.temp[r,t-1] 
                                    + zeta_conv * model.p[r,t-1] 
                                    - zeta_loss * (model.temp[r,t-1] - T_out[t-1]) 
                                    + zeta_exch * (model.temp[3-r,t-1] - model.temp[r,t-1]) 
                                    - zeta_cool * model.v[t-1] 
                                    + zeta_occ * occ
            )
                
    # 4. humidity dynamics for each time period
    model.hum_dyn = ConstraintList() 
    for t in model.T: 
        if t > 0: 
            model.hum_dyn.add(model.hum[t] == model.hum[t-1] 
                                           + eta_occ * (occ_r1[t-1] 
                                           + occ_r2[t-1]) 
                                           - eta_vent * model.v[t-1]
            ) 
    
    # 5. TEMPERATURE OVERRULE CONTROLLER   
    # 5.1 low temperature overrule controller: activation (if temp < T_low, controller must be ON)
    model.low_act = ConstraintList()
    for r in model.R:
        for t in model.T:
            model.low_act.add(M * model.delta_low[r,t] >= T_low - model.temp[r,t]) 

    # 5.2 low temperature overrule controller: memory (if temp <= T_ok, controller must remain ON if it was previously activated)
    model.low_mem = ConstraintList()
    for r in model.R:
        for t in model.T:
            if t > 0:
                model.low_mem.add(M * model.delta_low[r,t] >= (T_ok + epsilon - model.temp[r,t]) - M * (1 - model.delta_low[r,t-1]))
                # epsilon is forcing the controller to remain active even when T = T_ok
    # 5.3 low temperature overrule controller: force power to max when activated
    model.power_max = ConstraintList()
    for r in model.R:
        for t in model.T:
            model.power_max.add(model.p[r,t] >= P_max * model.delta_low[r,t]) 

    # 5.4 low temperature overrule controller: if temp > T_ok, deactivate
    model.low_deact = ConstraintList()
    for r in model.R:
        for t in model.T:
            model.low_deact.add(M * (1 - model.delta_low[r,t]) >= model.temp[r,t] - T_ok) 

    # HIGH TEMPERATURE OVERRULE CONTROLLER
    # 5.5 high temperature overrule controller: activation 
    model.high_act = ConstraintList()
    for r in model.R:
        for t in model.T:
            model.high_act.add(M * model.delta_high[r,t] >= model.temp[r,t] - T_high) 

    # 5.6 high temperature overrule controller: force power to 0 when activated 
    model.power_off = ConstraintList() 
    for r in model.R:
        for t in model.T:
            model.power_off.add(model.p[r,t] <= P_max * (1 - model.delta_high[r,t])) 

    # 5.7 high temperature overrule controller: if temp <= T_high, deactivate
    model.high_deact = ConstraintList()
    for r in model.R:
        for t in model.T:
            model.high_deact.add(M * (1 - model.delta_high[r,t]) >= T_high - model.temp[r,t]) 

    # 6. HUMIDITY OVERRULE CONTROLLER
    # 6.1 activation
    model.hum_act = ConstraintList()
    for t in model.T: 
        model.hum_act.add(M * model.delta_hum[t] >= model.hum[t] - H_high)  

    # 6.2 ventilation on when humidity overrule controller is activated
    model.hum_force = ConstraintList()
    for t in model.T:
        model.hum_force.add(model.v[t] >= model.delta_hum[t]) 

    # 6.3 deactivation: if humidity <= H_high, deactivate
    model.hum_deact = ConstraintList()
    for t in model.T:
        model.hum_deact.add(M * (1 - model.delta_hum[t]) >= H_high - model.hum[t]) 

    # 7. VENTILATION SYSTEM INERTIA 
    model.vent_inertia = ConstraintList()
    for t in model.T: 
        # we need to manage the first two hours of the day separately, since the ventilation inertia constraint looks to the past
        if t == 1:
            model.vent_inertia.add(model.v[1] >= model.v[0])
        if t == 2:
            model.vent_inertia.add(model.v[2] >= model.v[1])
        # from the fourth hour of the day, we can start to consider the backwards ventilation inertia 
        if t >= 3:
            model.vent_inertia.add(
                model.v[t-1] + model.v[t-2] + model.v[t-3] >= 3 * (model.v[t-1] - model.v[t])) 

    # solver call
    solver = SolverFactory('gurobi')
    result = solver.solve(model)

    # extract results
    p_opt  = {(r,t): value(model.p[r,t])    for r in model.R for t in model.T}   # optimal heating power
    v_opt  = {t: value(model.v[t])           for t in model.T}                   # optimal ventilation status
    temp_opt = {(r,t): value(model.temp[r,t]) for r in model.R for t in model.T} # temperature value
    hum_opt  = {t: value(model.hum[t])        for t in model.T}                  # humidity value

    total_cost = value(model.obj)

    return p_opt, v_opt, temp_opt, hum_opt, total_cost 

# initialize empty list to store daily costs and results
daily_costs = []
all_rows = [] # create empty list to store results for all days

# solve the optimization problem for each day and store results
for day in range(100):
    price  = price_data.iloc[day].values
    occ_r1 = occupancy_r1.iloc[day].values
    occ_r2 = occupancy_r2.iloc[day].values

    p_opt, v_opt, temp_opt, hum_opt, cost = solve_milp(price, occ_r1, occ_r2)
    daily_costs.append(cost)
    print(f"Day {day+1}: {cost:.2f}")

    for t in range(T):
        row = {
            'Day': day + 1,
            'Hour': t,
            'Price': price[t],
            'Occupancy_R1': occ_r1[t],
            'Occupancy_R2': occ_r2[t],
            'Temp_Room1': temp_opt[1, t],
            'Temp_Room2': temp_opt[2, t],
            'Power_Heater1': p_opt[1, t],
            'Power_Heater2': p_opt[2, t],
            'Ventilation_On': v_opt[t],
            'Humidity': hum_opt[t],
        }
        all_rows.append(row)

results_df = pd.DataFrame(all_rows)
OUTPUT_DIR = FILE_DIR/ "results" 
OUTPUT_DIR.mkdir(exist_ok = True)
results_df.to_csv(OUTPUT_DIR / 'HVAC_Optimization_Results.csv', index = False) 
print("Results saved to HVAC_Optimization_Results.csv (results folder)")

# out of the for loop, calculates and prints the average daily cost over the 100 days
average_cost = np.mean(daily_costs)
print(f"Average daily electricity cost: {average_cost:.2f}") 
total_cost = np.sum(daily_costs)
print(f"Total cost over the 100 days: {total_cost:.2f}")

