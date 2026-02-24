# imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from pyomo.environ import * 
from SystemCharacteristics import get_fixed_data 
from PlotsRestaurant import plot_HVAC_results 
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load data 
price_data = pd.read_csv('PriceData.csv') 
occupancy_r1 = pd.read_csv("OccupancyRoom1.csv")
occupancy_r2 = pd.read_csv("OccupancyRoom2.csv")

# parameters extraction 
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
    model.temp = Var(model.R, model.T, within=Reals)
    model.hum  = Var(model.T, within=NonNegativeReals)
     
    # auxiliary variables 
    model.delta_low  = Var(model.R, model.T, within=Binary)
    model.delta_high = Var(model.R, model.T, within=Binary)
    model.delta_hum  = Var(model.T, within=Binary)
    # objective function 
    model.obj = Objective(expr = sum (price[t] * (model.p[1,t] + model.p[2,t] + P_vent * model.v[t]) for t in model.T), sense=minimize) # minimize total cost of heating and ventilation

    # constraints 
    # 1. Initial conditions for temperature
    model.temp_init = ConstraintList() 
    for r in model.R: 
        model.temp_init.add(model.temp[r,0] == T0_prev) # initial temperature at time 0 is T0_prev, for every room 
    # 2. Initial conditions for humidity
    model.hum_init = ConstraintList() # initial humidity at time 0 is H0  
    model.hum_init.add(model.hum[0] == H0)
    # 3. Temperature dynamics for each room and time period
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
    # 4. Humidity dynamics for each time period
    model.hum_dyn = ConstraintList() 
    for t in model.T: 
        if t > 0: 
            model.hum_dyn.add(model.hum[t] == model.hum[t-1] + eta_occ * (occ_r1[t-1] + occ_r2[t-1]) - eta_vent * model.v[t-1]) 
    # 5. Low Temperature Overrule Controller  
    # 5.1 activation of the overrule controller: if temp < T_low, heater must be ON
    model.low_act = ConstraintList()
    M = 100
    for r in model.R:
        for t in model.T:
            model.low_act.add(M * model.delta_low[r,t] >= T_low - model.temp[r,t]) 
    # 5.2 Low Temperature Overrule Controller: memory
    model.low_mem = ConstraintList()
    for r in model.R:
        for t in model.T:
            if t > 0:
                model.low_mem.add(M * model.delta_low[r,t] >= T_ok * model.delta_low[r,t-1] - model.temp[r,t]) 
    # 5.3 Low Temperature Overrule Controller: force power to max when activated
    model.power_max = ConstraintList()
    for r in model.R:
        for t in model.T:
            model.power_max.add(model.p[r,t] >= P_max * model.delta_low[r,t]) 
    # 5.4 High Temperature Overrule Controller: activation 
    model.high_act = ConstraintList()
    M = 100
    for r in model.R:
        for t in model.T:
            model.high_act.add(M * model.delta_high[r,t] >= model.temp[r,t] - T_high) 
    # 5.5 High Temperature Overrule Controller: memory 
    model.high_mem = ConstraintList()
    for r in model.R:
        for t in model.T:
            if t > 0:
                model.high_mem.add(M * model.delta_high[r,t] >= (model.temp[r,t] - T_ok) - M * (1 - model.delta_high[r,t-1])) 
    # 5.6 High Temperature Overrule Controller: force power to 0 when activated 
    model.power_off = ConstraintList() 
    for r in model.R:
        for t in model.T:
            model.power_off.add(model.p[r,t] <= P_max * (1 - model.delta_high[r,t])) 
    # 5.7 Conflict resolution between Low and High Temperature Overrule Controllers: both cannot be activated at the same time 
    model.conflict_res = ConstraintList()
    for r in model.R:
        for t in model.T:
            model.conflict_res.add(model.delta_low[r,t] + model.delta_high[r,t] <= 1)          
    # 6. Humidity Overrulle Controller 
    # 6.1 Activation
    model.hum_act = ConstraintList()
    M = 100
    for t in model.T: 
        model.hum_act.add(M * model.delta_hum[t] >= model.hum[t] - H_high) 
    # 6.2 Ventilation On when Humidity Overrule Controller is activated
    model.hum_force = ConstraintList()
    for t in model.T:
        model.hum_force.add(model.v[t] >= model.delta_hum[t])
    # 7.Ventilation System Inertia 
    model.vent_inertia = ConstraintList()
    for t in model.T: 
        if t >= 3:    
            model.vent_inertia.add(model.v[t-3] + model.v[t-1] + model.v[t-2] >= 3 * (model.v[t-1] - model.v[t]))
    # 8. Initial Condition — Low Temperature Controller
    model.low_init = ConstraintList()
    for r in model.R:
        model.low_init.add(model.delta_low[r, 0] == 0)
    # 9. Initial Condition — High Temperature Controller
    model.high_init = ConstraintList()
    for r in model.R:
        model.high_init.add(model.delta_high[r, 0] == 0) 
    # 10. Initial Condition — Humidity Controller
    model.hum_ctrl_init = ConstraintList() 
    model.hum_ctrl_init.add(model.delta_hum[0] == 0) 
    # 11. Controller Deactivation Constraints 
    # 11.1 Low Temperature Overrule Controller: if temp >= T_ok,deactivate
    model.low_deact = ConstraintList()
    for r in model.R:
        for t in model.T:
            model.low_deact.add(M * (1 - model.delta_low[r,t]) >= model.temp[r,t] - T_ok) 
    # 11.2 High Temperature Overrule Controller: if temp <= T_ok, deactivate 
    model.high_deact = ConstraintList()
    for r in model.R:
        for t in model.T:
            model.high_deact.add(M * (1 - model.delta_high[r,t]) >= T_ok - model.temp[r,t]) 
    # 11.3  High Temperature Overrule Controller: if T ok <= temp <= T_high, and the controller wasn't previously activated, deactivate it 
    for r in model.R:
        for t in model.T: 
           if t > 0:
                model.high_deact.add(M * (1 - model.delta_high[r,t]) >= T_high - model.temp[r,t] - M * model.delta_high[r,t-1])
    
# solver call
    solver = SolverFactory('gurobi')
    result = solver.solve(model)

    # extract results
    p_opt  = {(r,t): value(model.p[r,t])    for r in model.R for t in model.T}
    v_opt  = {t: value(model.v[t])           for t in model.T}
    temp_opt = {(r,t): value(model.temp[r,t]) for r in model.R for t in model.T}
    hum_opt  = {t: value(model.hum[t])        for t in model.T}

    total_cost = value(model.obj)

    return p_opt, v_opt, temp_opt, hum_opt, total_cost 

# solve for each day
daily_costs = []
for day in range(100):
    price   = price_data.iloc[day].values
    occ_r1  = occupancy_r1.iloc[day].values
    occ_r2  = occupancy_r2.iloc[day].values

    p_opt, v_opt, temp_opt, hum_opt, cost = solve_milp(price, occ_r1, occ_r2)
    daily_costs.append(cost)

all_rows = []
    # Collect hourly data
    for t in range(T):
        row = {
            'Day': day + 1,
            'Hour': t,
            'Price': price_val[t],
            'Occupancy_R1': occ_r1[t],
            'Occupancy_R2': occ_r2[t],
            'Temp_Room1': temp_opt[1, t],
            'Temp_Room2': temp_opt[2, t],
            'Power_Heater1': p_opt[1, t],
            'Power_Heater2': p_opt[2, t],
            'Ventilation_On': v_opt[t],
            'Humidity': hum_opt[t],
            'Daily_Total_Cost': cost if t == 0 else None  # Optional: only log cost once per day
        }
        all_rows.append(row)

# 2. Convert to DataFrame
results_df = pd.DataFrame(all_rows)

# 3. Export to CSV
results_df.to_csv('HVAC_Optimization_Results.csv', index=False)
print("Results successfully saved to HVAC_Optimization_Results.csv")


for day in range(100):
    print(f"Day {day+1}: {daily_costs[day]:.2f}")
average_cost = np.mean(daily_costs)
print(f"Average daily electricity cost: {average_cost:.2f}")

