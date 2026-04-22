from pathlib import Path
from pyomo.environ import *
import numpy as np
import pandas as pd
from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data

data = get_fixed_data()
N               = 100
L               = data['num_timeslots']
P_max           = data['heating_max_power']
zeta_exch       = data['heat_exchange_coeff']
zeta_conv       = data['heating_efficiency_coeff']
zeta_loss       = data['thermal_loss_coeff']
zeta_cool       = data['heat_vent_coeff']
zeta_occ        = data['heat_occupancy_coeff']
T_low           = data['temp_min_comfort_threshold']
T_ok            = data['temp_OK_threshold']
T_high          = data['temp_max_comfort_threshold']
T_out           = data['outdoor_temperature']
P_vent          = data['ventilation_power']
H_high          = data['humidity_threshold']
eta_occ         = data['humidity_occupancy_coeff']
eta_vent        = data['humidity_vent_coeff']
min_up_time     = data['vent_min_up_time']

np.random.seed(42)

initial_state = {
    'T1'             : data['T1'],
    'T2'             : data['T2'],
    'H'              : data['H'],
    'vent_counter'   : data['vent_counter'],
    'low_override_r1': data['low_override_r1'],
    'low_override_r2': data['low_override_r2'],
    'current_time'   : 0,
    'Occ1'           : data['Occ1'],
    'Occ2'           : data['Occ2'],
    'price_t'        : data['price_t'],
    'price_previous' : data['price_previous']
}

def phi(state):
    return np.array([
        (state["T1"] - 22) / 8,
        (state["T2"] - 22) / 8,
        (state["H"] - 30) / 70,
        (state["Occ1"] - 20) / 30,    
        (state["Occ2"] - 10) / 20,    
        state["price_t"] / 12,
        state["price_previous"] / 12,
        state["vent_counter"] / 3,
        state["low_override_r1"],
        state["low_override_r2"]
    ])

def generate_exogenous(state):
    price_new          = price_model(state["price_t"], state["price_previous"])
    occ1_new, occ2_new = next_occupancy_levels(state["Occ1"], state["Occ2"])
    return {
        "price_t": price_new,
        "Occ1":    occ1_new,
        "Occ2":    occ2_new
    }

def apply_overrule(state, action):
    p1 = action["p1"]
    p2 = action["p2"]
    v  = action["v"]

    if state["low_override_r1"] and state["T1"] < T_ok:
        p1 = P_max
    if state["low_override_r2"] and state["T2"] < T_ok:
        p2 = P_max

    if state["T1"] >= T_high:
        p1 = 0
    if state["T2"] >= T_high:
        p2 = 0

    if state["H"] > H_high:
        v = 1

    return {"p1": p1, "p2": p2, "v": v}

def update_override(T_new, lr_old):
    if lr_old == 0:
        return 1 if T_new < T_low else 0
    else:
        return 0 if T_new >= T_ok else 1

def simulate_transition(state, action, exogenous):
    T1 = state["T1"]
    T2 = state["T2"]
    H  = state["H"]
    vent_counter = state["vent_counter"]
    t  = state["current_time"]

    action = apply_overrule(state, action)
    p1 = action["p1"]
    p2 = action["p2"]
    v  = action["v"]

    T1_new = (T1
              + zeta_conv * p1
              + zeta_exch * (T2 - T1)
              + zeta_loss * (T_out[t] - T1)
              + zeta_occ * exogenous["Occ1"]
              - zeta_cool * v)

    T2_new = (T2
              + zeta_conv * p2
              + zeta_exch * (T1 - T2)
              + zeta_loss * (T_out[t] - T2)
              + zeta_occ * exogenous["Occ2"]
              - zeta_cool * v)

    H_new = H + eta_occ * (exogenous["Occ1"] + exogenous["Occ2"]) - eta_vent * v

    low_override_r1_new = update_override(T1_new, state["low_override_r1"])
    low_override_r2_new = update_override(T2_new, state["low_override_r2"])

    vent_counter_new = vent_counter + 1 if v == 1 else 0

    return {
        "T1":              T1_new,
        "T2":              T2_new,
        "H":               H_new,
        "vent_counter":    vent_counter_new,
        "low_override_r1": low_override_r1_new,
        "low_override_r2": low_override_r2_new,
        "current_time":    t + 1,
        "Occ1":            exogenous["Occ1"],
        "Occ2":            exogenous["Occ2"],
        "price_t":         exogenous["price_t"],
        "price_previous":  state["price_t"]
    }

def compute_cost(state, action):
    price = state["price_t"]
    p1 = action["p1"]
    p2 = action["p2"]
    v  = action["v"]
    return price * (p1 + p2 + P_vent * v)

def solve_forward_pass_fast(state, eta):
    """solution with algebra"""
    t = state["current_time"]
    eta_next = eta[t + 1] if t < (L - 1) else None
    price = state["price_t"]

    if eta_next is None:
        p1, p2, v = 0, 0, 0
    else:
        coeff_p1 = price + eta_next[0] * zeta_conv / 8
        coeff_p2 = price + eta_next[1] * zeta_conv / 8
        coeff_v  = (price * P_vent
                    - zeta_cool * (eta_next[0] / 8 + eta_next[1] / 8)
                    - eta_vent * eta_next[2] / 70
                    + eta_next[7] / 3)

        p1 = 0 if coeff_p1 > 0 else P_max
        p2 = 0 if coeff_p2 > 0 else P_max
        v  = 0 if coeff_v  > 0 else 1

    vc = state["vent_counter"]
    if 0 < vc < min_up_time:
        v = 1

    if state["low_override_r1"] and state["T1"] < T_ok:
        p1 = P_max
    if state["T1"] >= T_high:
        p1 = 0

    if state["low_override_r2"] and state["T2"] < T_ok:
        p2 = P_max
    if state["T2"] >= T_high:
        p2 = 0

    if state["H"] > H_high:
        v = 1

    return {
        "p1": p1,
        "p2": p2,
        "v":  v
    }

def solve_forward_pass_milp(state, eta):
    """solution with pyomo,too slow"""
    model = ConcreteModel()
    t = state["current_time"]
    eta_next = eta[t + 1] if t < (L - 1) else None

    K = 10
    samples = [generate_exogenous(state) for _ in range(K)]
    p_k = 1.0 / K

    model.p1 = Var(within=NonNegativeReals, bounds=(0, P_max))
    model.p2 = Var(within=NonNegativeReals, bounds=(0, P_max))
    model.v  = Var(within=Binary)

    model.K      = RangeSet(0, K - 1)
    model.T1_new = Var(model.K)
    model.T2_new = Var(model.K)
    model.H_new  = Var(model.K)
    model.vc_new = Var(within=NonNegativeReals)

    vent_counter = state["vent_counter"]
    if 0 < vent_counter < min_up_time:
        model.v.fix(1)

    if state["low_override_r1"] and state["T1"] < T_ok:
        model.p1.fix(P_max)
    if state["T1"] >= T_high:
        model.p1.fix(0)

    if state["low_override_r2"] and state["T2"] < T_ok:
        model.p2.fix(P_max)
    if state["T2"] >= T_high:
        model.p2.fix(0)

    if state["H"] > H_high:
        model.v.fix(1)

    model.dyn_T1 = Constraint(model.K, rule=lambda model, k:
        model.T1_new[k] == state["T1"]
        + zeta_conv * model.p1
        + zeta_exch * (state["T2"] - state["T1"])
        + zeta_loss * (T_out[t] - state["T1"])
        + zeta_occ * samples[k]["Occ1"]
        - zeta_cool * model.v)

    model.dyn_T2 = Constraint(model.K, rule=lambda model, k:
        model.T2_new[k] == state["T2"]
        + zeta_conv * model.p2
        + zeta_exch * (state["T1"] - state["T2"])
        + zeta_loss * (T_out[t] - state["T2"])
        + zeta_occ * samples[k]["Occ2"]
        - zeta_cool * model.v)

    model.dyn_H = Constraint(model.K, rule=lambda model, k:
        model.H_new[k] == state["H"]
        + eta_occ * (samples[k]["Occ1"] + samples[k]["Occ2"])
        - eta_vent * model.v)

    model.dyn_vc = Constraint(expr=
        model.vc_new == state["vent_counter"] + model.v)

    price = state["price_t"]
    immediate_cost = price * (model.p1 + model.p2 + P_vent * model.v)

    if eta_next is None:
        model.obj = Objective(expr=immediate_cost, sense=minimize)
    else:
        future_cost = 0
        for k in range(K):
            vfa_k = (eta_next[0] * (model.T1_new[k] - 22) / 8
                    + eta_next[1] * (model.T2_new[k] - 22) / 8
                    + eta_next[2] * (model.H_new[k] - 30) / 70
                    + eta_next[3] * (samples[k]["Occ1"] - 20) / 30
                    + eta_next[4] * (samples[k]["Occ2"] - 10) / 20
                    + eta_next[5] * (samples[k]["price_t"]) / 12
                    + eta_next[6] * (state["price_t"]) / 12
                    + eta_next[7] * model.vc_new / 3
                    + eta_next[8] * state["low_override_r1"]
                    + eta_next[9] * state["low_override_r2"])
            future_cost += p_k * vfa_k
        model.obj = Objective(expr=immediate_cost + future_cost, sense=minimize)

    solver = SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.solve(model)

    return {
        "p1": value(model.p1),
        "p2": value(model.p2),
        "v":  int(value(model.v))
    }

def forward_pass(eta, initial_state): 
    states  = [[None] * L for _ in range(N)]
    actions = [[None] * L for _ in range(N)]
    costs   = [[0.0] * L for _ in range(N)]

    for n in range(N):
        state = initial_state.copy()
        state["Occ1"]           = np.random.uniform(25, 35)
        state["Occ2"]           = np.random.uniform(15, 25)
        state["price_t"]        = np.random.uniform(2, 8)
        state["price_previous"] = np.random.uniform(2, 8)

        for t in range(L):
            states[n][t] = state.copy()
            action = solve_forward_pass_fast(state, eta)
            cost = compute_cost(state, action)
            actions[n][t] = action
            costs[n][t] = cost

            if t < L - 1:
                exogenous = generate_exogenous(state)
                state = simulate_transition(state, action, exogenous)

    return states, actions, costs

def backward_pass(states, actions, costs, eta): # policy evaluations (with fixed actions)
    n_features = len(phi(initial_state))
    K = 10
    alpha = 1.0

    for t in reversed(range(L)):
        targets  = np.zeros(N)
        features = np.zeros((N, n_features))

        for n in range(N):
            features[n] = phi(states[n][t])
            r = costs[n][t]

            if t == L - 1:
                targets[n] = r
            else:
                future_cost = 0
                samples = [generate_exogenous(states[n][t]) for _ in range(K)]
                for k in range(K):
                    s_next = simulate_transition(states[n][t], actions[n][t], samples[k])
                    future_cost += (1.0 / K) * (eta[t + 1] @ phi(s_next))
                targets[n] = r + future_cost

        A = features.T @ features + alpha * np.eye(n_features)
        b = features.T @ targets
        eta[t] = np.linalg.solve(A, b)

    return eta

# Initialize eta
n_features = len(phi(initial_state))
eta = np.ones((L, n_features))

# Training loop
N_iterations = 100
convergence_tol = 0.01
best_eta = eta.copy()
best_error = float("inf")

for i in range(N_iterations):
    print(f"Iteration {i}")
    states, actions, costs = forward_pass(eta, initial_state)
    eta = backward_pass(states, actions, costs, eta)

    total_error = 0
    K = 10
    for t in range(L):
        for n in range(N):
            r = costs[n][t]
            if t == L - 1:
                target = r
            else:
                future_cost = 0
                samples = [generate_exogenous(states[n][t]) for _ in range(K)]
                for k in range(K):
                    s_next = simulate_transition(states[n][t], actions[n][t], samples[k])
                    future_cost += (1.0 / K) * (eta[t + 1] @ phi(s_next))
                target = r + future_cost
            prediction = eta[t] @ phi(states[n][t])
            total_error += (prediction - target)**2

    avg_error = total_error / (L * N)
    print(f"  avg fit error: {avg_error:.4f}")

    if avg_error < best_error:
        best_error = avg_error
        best_eta = eta.copy()

    if avg_error < convergence_tol: # but never comes to 0
        print(f"Converged after {i+1} iterations")
        break

eta = best_eta

# Save weights
np.save("eta_weights.npy", eta)

feature_names = ["T1", "T2", "H", "Occ1", "Occ2", "price_t", "price_previous", "vent_counter", "low_override_r1", "low_override_r2"]
df = pd.DataFrame(eta, columns=feature_names, index=[f"stage_{t}" for t in range(L)])
df.to_csv("eta_weights.csv")
print(df)