from pathlib import Path
from pyomo.environ import *
import numpy as np
import pandas as pd
from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data

data = get_fixed_data()
N               = 1000
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
        1,                             # intercept
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
        "price_t":       price_new,
        "price_previous": state["price_t"],
        "Occ1":          occ1_new,
        "Occ2":          occ2_new
    }

def apply_overrule(state, action):
    p1 = action["p1"]
    p2 = action["p2"]
    v  = action["v"]

    if state["low_override_r1"] and state["T1"] < T_ok:
        p1 = P_max
    if state["low_override_r2"] and state["T2"] < T_ok:
        p2 = P_max

    if state["T1"] > T_high:
        p1 = 0
    if state["T2"] > T_high:
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
              + zeta_occ * state["Occ1"]
              - zeta_cool * v)

    T2_new = (T2
              + zeta_conv * p2
              + zeta_exch * (T1 - T2)
              + zeta_loss * (T_out[t] - T2)
              + zeta_occ * state["Occ2"]
              - zeta_cool * v)

    H_new = H + eta_occ * (state["Occ1"] + state["Occ2"]) - eta_vent * v

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
    action = apply_overrule(state, action)
    return state["price_t"] * (
        action["p1"] + action["p2"] + P_vent * action["v"]
    )

def solve_forward_pass_fast(state, eta):
    """Analytical solution: choose p1, p2, v by sign of their cost coefficient."""
    t = state["current_time"]
    eta_next = eta[t + 1] if t < (L - 1) else None
    price = state["price_t"]

    if eta_next is None:
        # Last timestep: no future value, minimize immediate cost only
        p1, p2, v = 0, 0, 0
    else:
        # Coefficient of p1 in (immediate cost + future VFA):
        # d/dp1 [ price*p1 + eta_next[1]/8 * zeta_conv * p1 ] = price + eta_next[1]*zeta_conv/8
        coeff_p1 = price + eta_next[1] * zeta_conv / 8
        coeff_p2 = price + eta_next[2] * zeta_conv / 8

        # Coefficient of v: immediate ventilation cost + effect on T1, T2, H, vent_counter
        coeff_v = (price * P_vent
                   - zeta_cool * (eta_next[1] / 8 + eta_next[2] / 8)
                   - eta_vent  *  eta_next[3] / 70
                   +              eta_next[8] / 3)

        # Binary choice: if coefficient > 0, set to lower bound (0); else upper bound
        p1 = 0 if coeff_p1 > 0 else P_max
        p2 = 0 if coeff_p2 > 0 else P_max
        v  = 0 if coeff_v  > 0 else 1

    # Apply overrule constraints on top of the optimal choice
    vc = state["vent_counter"]
    if 0 < vc < min_up_time:
        v = 1

    if state["low_override_r1"] and state["T1"] < T_ok:
        p1 = P_max
    if state["T1"] > T_high:
        p1 = 0

    if state["low_override_r2"] and state["T2"] < T_ok:
        p2 = P_max
    if state["T2"] > T_high:
        p2 = 0

    if state["H"] > H_high:
        v = 1

    return {"p1": p1, "p2": p2, "v": v}


def forward_pass(eta, initial_state):
    """
    Forward pass (Variant B):
    Roll out N trajectories using the current policy (eta).
    Store states, actions, costs for use in the backward pass.
    """
    states  = [[None] * L for _ in range(N)]
    actions = [[None] * L for _ in range(N)]
    costs   = [[0.0]  * L for _ in range(N)]

    for n in range(N):
        state = initial_state.copy()
        # Randomize initial exogenous state across trajectories
        state["Occ1"]           = np.random.uniform(25, 35)
        state["Occ2"]           = np.random.uniform(15, 25)
        state["price_t"]        = np.random.uniform(2, 8)
        state["price_previous"] = np.random.uniform(2, 8)

        for t in range(L):
            states[n][t]  = state.copy()
            action        = solve_forward_pass_fast(state, eta)
            cost          = compute_cost(state, action)
            actions[n][t] = action
            costs[n][t]   = cost

            if t < L - 1:
                exogenous = generate_exogenous(state)
                state     = simulate_transition(state, action, exogenous)

    return states, actions, costs


def backward_pass(states, actions, costs, eta_target):
    """
    Backward pass — Policy Evaluation (Variant B).

    Actions are FIXED (from the forward pass).
    Targets are computed using eta_target, which is FROZEN before the J sweeps start.
    This implements pure policy evaluation, not policy improvement.

    Returns a NEW eta array (does not modify eta_target in place).
    """
    n_features = len(phi(initial_state))
    alpha      = 1   # Ridge regression regularization
    K          = 10    # Number of exogenous samples per (n, t) for target estimation

    eta_new = eta_target.copy()  # start from current estimate, update stage by stage

    for t in reversed(range(L)):
        targets  = np.zeros(N)
        features = np.zeros((N, n_features))

        for n in range(N):
            features[n] = phi(states[n][t])
            r = costs[n][t]

            if t == L - 1:
                # Terminal stage: no future cost
                targets[n] = r
            else:
                # Target = immediate cost + E[V̂(s_{t+1}; eta_target)]
                # Use eta_target (frozen) for all future value estimates
                future_cost = 0.0
                samples = [generate_exogenous(states[n][t]) for _ in range(K)]
                for k in range(K):
                    s_next       = simulate_transition(states[n][t], actions[n][t], samples[k])
                    future_cost += (1.0 / K) * (eta_new[t + 1] @ phi(s_next))
                targets[n] = r + future_cost

        # Ridge regression: eta_new[t] = argmin sum_n (eta^T phi(s) - target)^2 + alpha*||eta||^2
        A          = features.T @ features + alpha * np.eye(n_features)
        b          = features.T @ targets
        eta_new[t] = np.linalg.solve(A, b)

    return eta_new


def compute_avg_fit_error(states, actions, costs, eta, K=10):
    """
    Measure how well eta fits the targets on the current trajectories.
    Used to monitor convergence after each outer iteration.
    """
    total_error = 0.0
    for t in range(L):
        for n in range(N):
            r = costs[n][t]
            if t == L - 1:
                target = r
            else:
                future_cost = 0.0
                samples = [generate_exogenous(states[n][t]) for _ in range(K)]
                for k in range(K):
                    s_next       = simulate_transition(states[n][t], actions[n][t], samples[k])
                    future_cost += (1.0 / K) * (eta[t + 1] @ phi(s_next))
                target = r + future_cost
            prediction   = eta[t] @ phi(states[n][t])
            total_error += (prediction - target) ** 2
    return total_error / (L * N)


# ─────────────────────────────────────────────
# TRAINING LOOP  —  Variant B: Approximate Policy Iteration
# ─────────────────────────────────────────────

n_features     = len(phi(initial_state))
eta            = np.ones((L, n_features))   # Initialize eta
best_eta       = eta.copy()
best_error     = float("inf")
N_iterations   = 100
J              = 5                          # Number of policy evaluation sweeps per iteration
convergence_tol = 0.01

for i in range(N_iterations):
    print(f"Iteration {i+1}/{N_iterations}")

    states, actions, costs = forward_pass(eta, initial_state)

    eta_current = eta.copy()

    for j in range(J):
        eta_current = backward_pass(states, actions, costs, eta_current)

    beta = 0.2
    eta = (1 - beta) * eta + beta * eta_current

    avg_error = compute_avg_fit_error(states, actions, costs, eta)
    print(f"  avg fit error: {avg_error:.4f}")

    if avg_error < best_error:
        best_error = avg_error
        best_eta = eta.copy()

    if avg_error < convergence_tol:
        print(f"Converged at iteration {i+1}")
        break

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
eta = best_eta
np.save("eta_weights.npy", eta)

feature_names = [
    "Intercept", "T1", "T2", "H",
    "Occ1", "Occ2", "price_t", "price_previous",
    "vent_counter", "low_override_r1", "low_override_r2"
]
df = pd.DataFrame(eta, columns=feature_names, index=[f"stage_{t}" for t in range(L)])
df.to_csv("eta_weights.csv")
print(df)