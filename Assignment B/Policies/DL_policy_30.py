import time
from pyomo.environ import *
import numpy as np
from Policies.ADP_policy_30 import generate_samples
from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data

# parameters extraction from system characteristics
data        = get_fixed_data()
T           = data['num_timeslots']
P_max       = data['heating_max_power']
zeta_exch   = data['heat_exchange_coeff']
zeta_conv   = data['heating_efficiency_coeff']
zeta_loss   = data['thermal_loss_coeff']
zeta_cool   = data['heat_vent_coeff']
zeta_occ    = data['heat_occupancy_coeff']
T_low       = data['temp_min_comfort_threshold']
T_ok        = data['temp_OK_threshold']
T_high      = data['temp_max_comfort_threshold']
T_out       = data['outdoor_temperature']
P_vent      = data['ventilation_power']
H_high      = data['humidity_threshold']
eta_occ     = data['humidity_occupancy_coeff']
eta_vent    = data['humidity_vent_coeff']
U_vent      = data["vent_min_up_time"] # 3 hours

M_temp = 50  # big-M constant for temperature ()
M_hum = 100   # big-M constant for humidity

epsilon = 1e-9 # small constant to prevent decimal precision issues in constraints


def forecast_uncertainties(state: dict, L: int, n_samples: int):    
    # Obtain current state variables
    current_price = state["price_t"]
    previous_price = state["price_previous"]
    occ1 = state["Occ1"]
    occ2 = state["Occ2"]

    forecast = {
            "price": [],
            "occ1": [],
            "occ2": []
    }

    for _ in range(L):
        list_of_prices = []
        list_of_occ1   = []
        list_of_occ2   = []

        # Generate n_samples for the next time step
        for _ in range(n_samples):
            list_of_prices.append(price_model(current_price, previous_price))
            list_of_occ1.append(occ1)
            list_of_occ2.append(occ2)

        # Calculate the expected next values
        next_price = np.mean(list_of_prices)
        occ1 = np.mean(list_of_occ1)
        occ2 = np.mean(list_of_occ2)

        # Append forecasts for the current time step
        forecast["price"].append(float(next_price))
        forecast["occ1"].append(float(occ1))
        forecast["occ2"].append(float(occ2))

        # Update current state for the next iteration
        previous_price = current_price
        current_price = next_price

    return forecast


def provide_real_future(day: int, t: int, L: int):    
    """Use this one to simulate the OIH"""
    occupancy1_matrix = np.genfromtxt("Data/OccupancyRoom1.csv", delimiter=",", skip_header=1)
    occupancy2_matrix = np.genfromtxt("Data/OccupancyRoom2.csv", delimiter=",", skip_header=1)
    raw_price_data = np.genfromtxt("Data/v2_PriceData.csv", delimiter=",", skip_header=1)
    price_data     = raw_price_data[:, 1:]  


    forecast = {
            "price": price_data[day, t+1:t+L].tolist(),
            "occ1": occupancy1_matrix[day, t+1:t+L].tolist(),
            "occ2": occupancy2_matrix[day, t+1:t+L].tolist()
    }
    return forecast



def solve_MILP(state: dict, forecast: dict, L: int) -> tuple:
    # Extract current state variables
    temperature = [state["T1"], state["T2"]]
    humidity = state["H"]
    current_time = state["current_time"]
    current_price = state["price_t"]
    overrulers = [int(state["low_override_r1"]), int(state["low_override_r2"])]
    current_vent_counter = state["vent_counter"]
    current_occ1 = float(state["Occ1"])
    current_occ2 = float(state["Occ2"])

    # Extract forecasted uncertainties
    price_forecast      = forecast["price"]
    occupancy_forecast  = [[current_occ1] + forecast["occ1"],  # We include the current occupancy as the first element of the forecasted occupancy list for each room
                           [current_occ2] + forecast["occ2"]]

    # Ventilation must remain on if the minimum up-time constraint is active
    v_inertia = 1 if 0 < current_vent_counter < U_vent else 0

    # Ventilation status in the previous time step
    v_prev = 1 if current_vent_counter > 0 else 0


    # Create model
    model = ConcreteModel()

    # Sets
    model.T = RangeSet(current_time, current_time + L-1)
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
    model.overrule  = Var(model.R, model.T, domain=Binary) # binary variable indicating if the heater's overrule controller is active in room r at time t (1 if active, 0 otherwise)

    # Ventilation startup at time t
    model.s = Var(model.T, domain=Binary) # binary variable indicating ventilation startup at time t (1 if ventilation starts at time t, 0 otherwise)


    # Objective function
    inmediate_cost  = current_price * (model.p[0, current_time] + model.p[1, current_time]) + P_vent * model.v[current_time]
    forecasted_cost = sum(price_forecast[t-current_time-1] * (P_vent * model.v[t] + model.p[0, t] + model.p[1, t]) for t in model.T if t > current_time)

    model.obj = Objective(expr=inmediate_cost + forecasted_cost, sense=minimize)

    # Constraints
    # 1-2. Temperature dynamics - OK
    model.c1 = Constraint(model.R, rule=lambda model, r: model.temp[r, current_time] == temperature[r]) # initial temperature constraint at time 0
    model.c2 = Constraint(model.R, model.T, rule=lambda model, r, t: (model.temp[r, t] == model.temp[r, t-1] + 
                                                                        zeta_exch * (model.temp[1-r, t-1] - model.temp[r, t-1]) -
                                                                        zeta_loss * (model.temp[r, t-1] - T_out[t-1]) +
                                                                        zeta_conv * model.p[r, t-1] - 
                                                                        zeta_cool * model.v[t-1] +
                                                                        zeta_occ * occupancy_forecast[r][t-1-current_time] - epsilon) 
                                                                        if t != model.T.first() else Constraint.Skip)

    # 3-4. Humidity dynamics - OK
    model.c3 = Constraint(expr=model.hum[current_time] == humidity) # initial humidity constraint at time 0
    model.c4 = Constraint(model.T, rule=lambda model, t: 
                                        (model.hum[t] == model.hum[t-1] + 
                                        eta_occ * sum(occupancy_forecast[r][t-1-current_time] for r in model.R) - 
                                        eta_vent * model.v[t-1]) if t != model.T.first() else Constraint.Skip)

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
    model.c13 = Constraint(model.R, model.T,rule=lambda model, r, t: 
                           model.overrule[r, t] <= overrulers[r] + model.temp_low[r, t] if t == model.T.first()
                           else model.overrule[r, t] <= model.overrule[r, t-1] + model.temp_low[r, t])
                           

    # 14. Overrule controller forcing heater to maximum - OK
    model.c14 = Constraint(model.R, model.T, rule=lambda model, r, t: model.p[r, t] >= P_max * model.overrule[r, t])

    # 15-16. De-activating Overrule (only) if Temperature is above the “OK” level
    model.c15 = Constraint(model.R, model.T, rule=lambda model, r, t: 
                           model.overrule[r, t] >= model.overrule[r, t-1] - model.temp_ok[r, t] if t != model.T.first() 
                            else model.overrule[r, t] >= overrulers[r] - model.temp_ok[r, t])
    
    model.c16 = Constraint(model.R, model.T, rule=lambda model, r, t: model.overrule[r, t] <= 1 - model.temp_ok[r, t] 
                                if t > 0 else Constraint.Skip)

    # 17-20. Ventilation Startup and Minimum Up-Time
    valid_tau = {}
    for t in model.T:
        tau_max = min(t + U_vent - 1, current_time + L - 1)
        valid_tau[t] = list(range(t, tau_max + 1))

    model.c17 = Constraint(model.T, rule=lambda model, t: 
             (model.s[t] >= model.v[t] - model.v[t-1]) if t != model.T.first() 
             else (model.s[t] >= model.v[t] - v_prev))
    
    model.c18 = Constraint(model.T, rule=lambda model, t: model.s[t] <= model.v[t])
    model.c19 = Constraint(model.T, rule=lambda model, t: 
                            model.s[t] <= 1 - model.v[t-1] if t != model.T.first() 
                            else model.s[t] <= 1 - v_prev
    )
    
    model.c20 = Constraint(model.T, rule=lambda model, t: sum(model.v[tau] for tau in valid_tau[t]) >= min(U_vent, current_time + L - t) * model.s[t])
    model.c21 = Constraint(rule=model.v[current_time] >= v_inertia)


    # 22. Humidity-Triggered Ventilation - OK
    model.c22 = Constraint(model.T, rule=lambda model, t: model.hum[t] <= H_high + M_hum * model.v[t])


    # Solve model
    solver = SolverFactory("gurobi")
    result = solver.solve(model, options={"OutputFlag": 0})

    if result.solver.termination_condition != TerminationCondition.optimal:
        return 0.0, 0.0, 0
    

    # Print results in table format
    # print(f"{'Hour':<6} {'p1':<8} {'p2':<8} {'temp_r1':<10} {'temp_r2':<10} {'overr_r1':<10} {'overr_r2':<10} {'humidity':<10} {'v':<4} {'price_f':<10} {'occ1_f':<8} {'occ2_f':<8}")
    # print("-" * 110)
    # for t in model.T:
    #     p1_val = value(model.p[0, t])
    #     p2_val = value(model.p[1, t])
    #     v_val = int(value(model.v[t]) > 0.5)
    #     temp_r1_val = value(model.temp[0, t])
    #     temp_r2_val = value(model.temp[1, t])
    #     hum_val = value(model.hum[t])
    #     overr_r1_val = int(value(model.overrule[0, t]) > 0.5)
    #     overr_r2_val = int(value(model.overrule[1, t]) > 0.5)
    #     idx = t - current_time
    #     # print(t, current_time, idx)
    #     occ1_pred = occupancy_forecast[0][idx]
    #     occ2_pred = occupancy_forecast[1][idx]

    #     if t == current_time:
    #         price_pred = current_price
    #         # occ1_pred = current_occ1
    #         # occ2_pred = current_occ2
    #     else:
    #         idx = t - current_time - 1
    #         price_pred = price_forecast[idx]


    #     print(f"{t:<6} {p1_val:<8.3f} {p2_val:<8.3f} {temp_r1_val:<10f} {temp_r2_val:<10f} {overr_r1_val:<10} {overr_r2_val:<10} {hum_val:<10.3f} {v_val:<4} {price_pred:<10.3f} {occ1_pred:<8.3f} {occ2_pred:<8.3f}")
    # print("\n")


    # We only return the here-and-now decisions, the rest is discarded  
    p1 = value(model.p[0, current_time])
    p2 = value(model.p[1, current_time])
    v = int(value(model.v[current_time]) > 0.5)

    return p1, p2, v




def select_action(state):
    start = time.time()

    day = state[1] # extract day from the state tuple
    state = state[0] # extract state from the list (since we are only passing one state, we can just take the first element)

    # Length of lookahead horizon (can't be longer than remaining time steps in the day)
    L = min(10, T - state["current_time"])

    # Forecast uncertainties across the lookahead horizon
    forecast = forecast_uncertainties(state, L, n_samples=10_000)
    # forecast = provide_real_future(day, state["current_time"], L)

    # Solve MILP to get optimal actions
    p1, p2, v = solve_MILP(state, forecast, L)

    # print(state["current_time"], f": p1={p1:.2f}, p2={p2:.2f}, v={v}")

    return {
        "HeatPowerRoom1": p1,
        "HeatPowerRoom2": p2,
        "VentilationON": v
    }


