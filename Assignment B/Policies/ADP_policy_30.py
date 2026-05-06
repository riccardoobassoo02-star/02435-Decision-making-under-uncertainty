from pandas.core.common import temp_setattr
from pyomo.environ import *
from sklearn.cluster import KMeans
import numpy as np
import time


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
min_up_time = data['vent_min_up_time']
U_vent      = data["vent_min_up_time"] # 3 hours
L           = data["num_timeslots"] # 10 hours



M_temp = 50  # big-M constant for temperature ()
M_hum = 100   # big-M constant for humidity


eta_weights = np.load("eta_weights.npy")


def generate_samples(state, B, N_samples):
    sample_prices = []
    sample_occ1s  = []
    sample_occ2s  = []

    for _ in range(N_samples):
        p = price_model(state["price_t"], state["price_previous"]) 
        o1, o2 = next_occupancy_levels(state["Occ1"], state["Occ2"])
        sample_prices.append(p)
        sample_occ1s.append(o1)
        sample_occ2s.append(o2)


    # CLUSTERING: reduce N_samples to B representative centroids
    X         = np.column_stack([sample_prices, sample_occ1s, sample_occ2s]) # feature matrix with all the N samples rows
    km        = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
    labels    = km.labels_
    centroids = km.cluster_centers_   # reduced matrix of shape (B, 3)

    clusters = []

    for b in range(B):
        cluster_prob = np.sum(labels == b) / N_samples

        data = {
            "price": centroids[b, 0],                 
            "occ_room_0":  centroids[b, 1],                 
            "occ_room_1":  centroids[b, 2],
            "prob":  cluster_prob
        }
        clusters.append(data)

    return clusters


def value_function(model, scenario_idx, scenarios, current_state):
    # Current time (t)
    hour = current_state["current_time"]
     
    # State variables for scenario s in time t+1
    temp_r1_next      = model.temp_next[0, scenario_idx]
    temp_r2_next      = model.temp_next[1, scenario_idx]
    humidity_next     = model.humidity_next[scenario_idx]
    occ_room_0_next   = scenarios[scenario_idx]["occ_room_0"]
    occ_room_1_next   = scenarios[scenario_idx]["occ_room_1"]
    price_next        = scenarios[scenario_idx]["price"]
    prev_price_next   = current_state["price_t"]     # Price at time t (previous price in t+1)
    vent_counter_next = model.vent_counter_next
    overrule_r1_next  = model.overrule_next[0, scenario_idx]
    overrule_r2_next  = model.overrule_next[1, scenario_idx]
    
    # Create next state vector
    next_state = np.array([temp_r1_next, 
                          temp_r2_next, 
                          humidity_next, 
                          occ_room_0_next, 
                          occ_room_1_next, 
                          price_next, 
                          prev_price_next, 
                          vent_counter_next,  
                          overrule_r1_next, 
                          overrule_r2_next])

    # Multiply future state with weights
    value = sum(eta_weights[hour][i] * next_state[i] for i in range(len(next_state)))
    
    return value


    # "T1": temperature_room1,
    # "T2": temperature_room2,
    # "H": humidity,
    # "Occ1": occupancy1_matrix[day][hour],
    # "Occ2": occupancy2_matrix[day][hour],
    # "price_t": price_matrix[day][hour],
    # "price_previous": previous_price,
    # "vent_counter": vent_counter,
    # "low_override_r1": is_override_room1,
    # "low_override_r2": is_override_room2,
    # "current_time": hour

def solve_MILP(state, scenarios):
    # State variables at time t
    current_temp = [state["T1"], state["T2"]]
    current_humidity = state["H"]
    current_price = state["price_t"]
    current_vent_counter = state["vent_counter"]
    t = state["current_time"]

    # Binary variable indicating if ventilation has to be kept on because of inertia
    v_inertia = 1 if current_vent_counter in [1, 2] else 0

    # variable indicating if ventilation was on in the previous time step
    v_prev = current_vent_counter > 0

    # Variable indicating if overrulers were active in the previous time step
    overrulers_prev = [state["low_override_r1"], state["low_override_r2"]]


    # Create model
    model = ConcreteModel()

    # Sets
    model.R = RangeSet(0, 1)
    model.Scenarios = RangeSet(0, len(scenarios) - 1)

    # Decision variables
    model.p = Var(model.R, domain=NonNegativeReals, bounds=(0, P_max)) # heating power
    model.v = Var(domain=Binary) # ventilation ON/OFF (1 if ON, 0 if OFF)

    # Overrule controller
    model.y_high = Var(model.R, domain=Binary) # binary variable indicating if temperature in room r at time t is above the maximum comfort threshold (1 if temp > T_high, 0 otherwise)
    model.y_low  = Var(model.R, domain=Binary) # binary variable indicating if temperature in room r at time t is below the minimum comfort threshold (1 if temp < T_low, 0 otherwise)
    model.y_ok   = Var(model.R, domain=Binary) # binary variable indicating if temperature in room r at time t is above the "OK" threshold (1 if temp > T_ok, 0 otherwise)
    model.overrule  = Var(model.R, domain=Binary) # binary variable indicating if the heater's overrule controller is active in room r at time t (1 if active, 0 otherwise)

    # Ventilation startup at time t
    model.s = Var(domain=Binary) # binary variable indicating ventilation startup at time t (1 if ventilation starts at time t, 0 otherwise)

    # State Variables for each scenario at t+1
    model.temp_next         = Var(model.R, model.Scenarios, domain=NonNegativeReals)
    model.humidity_next     = Var(model.Scenarios, domain=NonNegativeReals)
    model.vent_counter_next = Var(domain=NonNegativeIntegers)
    model.overrule_next     = Var(model.R, model.Scenarios, domain=Binary)
    model.y_low_next        = Var(model.R, model.Scenarios, domain=Binary)
    model.y_ok_next         = Var(model.R, model.Scenarios, domain=Binary)


    # Objective function
    if True:#t == 9:  # Only inmediate reward (there's no future)
        model.obj = Objective(rule = current_price * (
            P_vent * model.v
            + model.p[0]
            + model.p[1]
        ), sense=minimize)

    else:   # Inmediate reward + future reward
        model.obj = Objective(rule = current_price * (
            P_vent * model.v
            + model.p[0]
            + model.p[1]
        ) + sum(value_function(model, s, scenarios, state) for s in model.Scenarios)
        , sense=minimize)
    

    # Constraints
    # 1-2. Temperature cutoff and heater deactivation:
    model.c1 = Constraint(model.R, rule=lambda model, r: current_temp[r] >= T_high - M_temp * (1 - model.y_high[r]))
    model.c2 = Constraint(model.R, rule=lambda model, r: current_temp[r] <= T_high + M_temp * model.y_high[r])

    # 3. Overrule controller forcing heater to zero:
    model.c3 = Constraint(model.R, rule=lambda model, r: model.p[r] <= P_max * (1 - model.y_high[r]))

    # 4-5. Detecting when Temperature is below threshold Tlow
    model.c4 = Constraint(model.R, rule=lambda model, r: current_temp[r] <= T_low + M_temp * (1 - model.y_low[r]))
    model.c5 = Constraint(model.R, rule=lambda model, r: current_temp[r] >= T_low - M_temp * model.y_low[r])

    # 6-7. Detecting when Temperature is above the “OK” threshold Tok
    model.c6 = Constraint(model.R, rule=lambda model, r: current_temp[r] >= T_ok - M_temp * (1 - model.y_ok[r]))
    model.c7 = Constraint(model.R, rule=lambda model, r: current_temp[r] <= T_ok + M_temp * model.y_ok[r])

    # 8-9. Triggering Overrule (only) if Temperature is low
    model.c8 = Constraint(model.R, rule=lambda model, r: model.overrule[r] >= model.y_low[r])
    model.c9 = Constraint(model.R, rule=lambda model, r: model.overrule[r] <= int(overrulers_prev[r]) + model.y_low[r])

    # 10. Overrule controller forcing heater to maximum
    model.c10 = Constraint(model.R, rule=lambda model, r: model.p[r] >= P_max * model.overrule[r])

    # 11-12. De-activating Overrule (only) if Temperature is above the “OK” level
    model.c11 = Constraint(model.R, rule=lambda model, r: model.overrule[r] >= int(overrulers_prev[r]) - model.y_ok[r])
    model.c12 = Constraint(model.R, rule=lambda model, r: model.overrule[r] <= 1 - model.y_ok[r])

    # 13-16. Ventilation Startup and Minimum Up-Time
    model.c13 = Constraint(rule = (model.s >= model.v) if t == 0 else (model.s >= model.v - v_prev))
    model.c14 = Constraint(rule = model.s <= model.v)
    model.c15 = Constraint(rule = model.s <= 1 - v_prev if t > 0 else Constraint.Skip)
    model.c16 = Constraint(rule = model.v >= v_inertia)

    # 17. Humidity-Triggered Ventilation
    model.c17 = Constraint(rule = current_humidity <= H_high + M_hum * model.v)


    #### CONSTRAINTS FOR NEXT STATE (montecarlo scenarios) ####
    # 18. Temperature dynamics at t+1 for each scenario
    model.c18 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: (model.temp_next[r, s] == current_temp[r] + 
                                                                        zeta_exch * (current_temp[1-r] - current_temp[r]) -
                                                                        zeta_loss * (current_temp[r] - T_out[t]) +
                                                                        zeta_conv * model.p[r] - 
                                                                        zeta_cool * model.v +
                                                                        zeta_occ * scenarios[s]["occ_room_" + str(r)]) 
                                                                        if t < 9 else Constraint.Skip)


    # 19. Humidity dynamics at t+1 for each scenario
    model.c19 = Constraint(model.Scenarios, rule=lambda model, s: 
                                        (model.humidity_next[s] == current_humidity + 
                                        eta_occ * sum(scenarios[s]["occ_room_" + str(r)] for r in model.R) - 
                                        eta_vent * model.v) if t < 9 else Constraint.Skip)

    # 20. Vent counter at t+1 for ALL scenarios
    model.c20 = Constraint(rule = model.vent_counter_next == model.v * (current_vent_counter + model.v))

    # 21-22. Detecting when Temperature is below threshold Tlow:
    model.c21 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.temp_next[r, s] <= T_low + M_temp * (1 - model.y_low_next[r, s]))
    model.c22 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.temp_next[r, s] >= T_low - M_temp * model.y_low_next[r, s])

    # 23-24. Triggering Overrule (only) if Temperature is low
    model.c23 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.overrule_next[r, s] >= model.y_low_next[r, s])
    model.c24 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.overrule_next[r, s] <= model.overrule[r] + model.y_low_next[r, s])

    # 25-26. De-activating Overrule (only) if Temperature is above the “OK” level
    model.c25 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.overrule_next[r, s] >= model.overrule[r] - model.y_ok_next[r, s])
    model.c26 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.overrule_next[r, s] <= 1 - model.y_ok_next[r, s])

    # 27-28. Detecting when Temperature is above the “OK” threshold
    model.c27 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.temp_next[r, s] >= T_ok - M_temp * (1 - model.y_ok_next[r, s]))
    model.c28 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.temp_next[r, s] <= T_ok + M_temp * model.y_ok_next[r, s])


    # SOLVE
    solver = SolverFactory('gurobi')
    result = solver.solve(model, options={"OutputFlag": 0}) # suppress solver output for cleaner logs

    if result.solver.termination_condition != TerminationCondition.optimal: 
        print("[WARNING] SP did not solve to optimality — returning zeros")
        return 0.0, 0.0, 0

    p1 = value(model.p[0])           # heating power of room 1 at tau=0
    p2 = value(model.p[1])           # heating power of room 2 at tau=0
    v  = int(value(model.v) > 0.5)   # ventilation ON/OFF at tau=0 (binary variable, thresholded at 0.5 for the solver tollerance)

    return p1, p2, v



# ENTRY POINT (called by the environment)
def select_action(state):
    start_time = time.time()

    scenarios = generate_samples(state, B=5, N_samples=1_000)
    p1, p2, v = solve_MILP(state, scenarios)

    # print(f"Total policy time: {time.time() - start_time:.2f} s")
    
    HereAndNowActions = {
        "HeatPowerRoom1": p1,
        "HeatPowerRoom2": p2,
        "VentilationON":  v
    } 

    return HereAndNowActions




# RICKY's code
        # if 0 < vent_counter < min_up_time:
    #     model.v.fix(1)

    # if state["low_override_r1"] and state["T1"] < T_ok:
    #     model.p[0].fix(P_max)
    # if state["T1"] >= T_high:
    #     model.p[0].fix(0)

    # if state["low_override_r2"] and state["T2"] < T_ok:
    #     model.p[1].fix(P_max)

    # if state["T2"] >= T_high:
    #     model.p[1].fix(0)

    # if state["H"] > H_high:
    #     model.v.fix(1)



# FIRST TRY - NOT WORKING
# def calculate_room_temperature(P, occupancy, prev_temperature, other_room_prev_temp, V, outside_temperature):
#     """Calculate the new temperature of a room based on the previous temperature, the heating power, occupancy, 
#     ventilation, and outdoor temperature.
#     Inputs:
#     - P: heating power applied to the room (here-and-now decision p1 or p2)
#     - occupancy: number of people in the room
#     - prev_temperature: previous temperature of the room
#     - other_room_prev_temp: previous temperature of the other room
#     - data: system characteristics dictionary
#     - V: ventilation system status (here-and-now decision v)
#     - outside_temperature: current outdoor temperature
#     """
#     return (
#         prev_temperature + 
#         zeta_exch * (other_room_prev_temp - prev_temperature) -
#         zeta_loss * (prev_temperature - outside_temperature) +
#         zeta_conv * P - 
#         zeta_cool * V + 
#         zeta_occ * occupancy
#     )

# def calculate_humidity(prev_humidity, occupancy1, occupancy2, V):
#     """Calculate the new humidity of a room based on the previous humidity, occupancy, and ventilation.
#     Inputs:
#     - prev_humidity: previous humidity of the room
#     - occupancy1: number of people in room 1
#     - occupancy2: number of people in room 2
#     - V: ventilation system status (here-and-now decision v)
#     """
#     return (prev_humidity +
#             eta_occ * (occupancy1 + occupancy2) -
#             eta_vent * V
#     )

# def update_overrule_controler_state(overrule_state, temperature):
#     return (
#         (overrule_state or temperature < T_low)
#         and
#         not (overrule_state and temperature > T_ok)
#     )


# def value_function(model, scenario, state):
#     # Variables from time t
#     hour = state["current_time"]
#     t1 = state["T1"]
#     t2 = state["T2"]
#     H  = state["H"]
#     new_price_previous = state["price_t"] # price in t = previous price in t+1
#     outside_temperature = T_out[hour]
#     vent_counter = state["vent_counter"]
#     low_override_r1 = state["low_override_r1"]
#     low_override_r2 = state["low_override_r2"]

#     # Simulated variables for time t+1
#     new_occupancy1 = scenario["occ1"]
#     new_occupancy2 = scenario["occ2"]
#     new_price = scenario["price"]

#     # Tempreature dynamics for t+1 depending on decisions for t: v and p
#     new_t1 = calculate_room_temperature(model.p[0], new_occupancy1, t1, t2, model.v, outside_temperature)
#     new_t2 = calculate_room_temperature(model.p[1], new_occupancy2, t2, t1, model.v, outside_temperature)

#     # Humidity dynamics for t+1 depending on decisions for t: v
#     new_H = calculate_humidity(H, new_occupancy1, new_occupancy2, model.v)

#     # Update vent_counter
#     new_vent_counter = vent_counter + model.v
   
#     # Update overrule controlers
#     new_low_override_r1 = update_overrule_controler_state(low_override_r1, new_t1)
#     new_low_override_r2 = update_overrule_controler_state(low_override_r2, new_t2)


#     new_state = np.array([new_t1, 
#                           new_t2, 
#                           new_H, 
#                           new_occupancy1, 
#                           new_occupancy2, 
#                           new_price, 
#                           new_price_previous, 
#                           new_vent_counter,  
#                           new_low_override_r1, 
#                           new_low_override_r2])

    
#     value = sum(eta_weights[hour][i] * new_state[i] for i in range(len(new_state)))
    
#     return value
