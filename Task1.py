# imports 
from pyexpat import model
from xml.parsers.expat import model
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from pyomo.environ import * 
from SystemCharacteristics import get_fixed_data 
from PlotsRestaurant import plot_HVAC_results 

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


    # objective function 
    model.obj = Objective(expr = sum (price[t] * (model.p[1,t] + model.p[2,t] + P_vent * model.v[t]) for t in model.T), sense=minimize) # minimize total cost of heating and ventilation

    # constraints 
    # 1. Initial conditions for temperature
    model.temp_init = ConstraintList() 
    for r in model.R: 
        model.temp_init.add(model.temp[r,0] == T0_prev) # initial temperature at time 0 is T0_prev, for every room 
    # 2. Initial conditions for humidity
    model.hum_init = ConstraintList # initial humidity at time 0 is H0 
    for t in model.T: 
        model.hum_init.add(model.hum[0] == H0)
    # 3. Temperature dynamics for each room and time period
    model.temp_dyn = ConstraintList() 
    for t in model.T: 
        for r in model.R:
            if t > 0: 
                model.temp_dyn.add(model.temp[r,t] == model.temp[r,t-1] + zeta_conv * model.p[r,t-1] - zeta_loss * (model.temp[r,t-1] - T_out) + zeta_exch * (model.temp[3-r,t-1] - model.temp[r,t-1]) - zeta_cool * model.v[t-1] + zeta_occ * occ_r1[t-1]) 
    # 4. Humidity dynamics for each time period
    model.hum_dyn = ConstraintList() 
    for t in model.T: 
        if t > 0: 
            model.hum_dyn.add(model.hum[t] == model.hum[t-1] + eta_occ * (occ_r1[t-1] + occ_r2[t-1]) - eta_vent * model.v[t-1]) 
    # 5. Temperature Overrule Controller  
    
    # 6. Humidity Overrulle Controller 

    # 7.Ventilation System Inertia 

    # 8. Initial COnditions For Controllers