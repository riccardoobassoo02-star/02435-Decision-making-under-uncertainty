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
    # 10-feature linear VFA basis.
    # price_previous is dropped: the price process is Markovian, only price_t matters.
    # An intercept is added in its place to absorb the constant component of the VFA.
    return np.array([
        1.0,                              # 0: intercept
        (state["T1"] - 22) / 8,           # 1: T1
        (state["T2"] - 22) / 8,           # 2: T2
        (state["H"]  - 30) / 70,          # 3: H
        (state["Occ1"] - 20) / 30,        # 4: Occ1
        (state["Occ2"] - 10) / 20,        # 5: Occ2
        state["price_t"] / 12,            # 6: price_t
        state["vent_counter"] / 3,        # 7: vent_counter
        state["low_override_r1"],         # 8: low_override_r1
        state["low_override_r2"]          # 9: low_override_r2
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
    """Analytical greedy action minimizing immediate_cost + eta_{t+1}^T phi(s_{t+1}).
    Used by BOTH forward and backward passes (so that the backward target
    is computed under the optimal action, i.e. a true Bellman backup)."""
    t = state["current_time"]
    eta_next = eta[t + 1] if t < (L - 1) else None
    price = state["price_t"]

    if eta_next is None:
        # Terminal stage: no successor VFA, so pure immediate-cost minimization -> all off
        p1, p2, v = 0, 0, 0
    else:
        # Marginal coefficient of each action variable on (immediate cost + future VFA).
        # Indices reflect the new phi: T1=1, T2=2, H=3, vent_counter=7.
        coeff_p1 = price + eta_next[1] * zeta_conv / 8
        coeff_p2 = price + eta_next[2] * zeta_conv / 8
        # vent_counter_{t+1} = (vc + 1) * v (discontinuous in v), so the v->1 effect
        # through the vent_counter feature is eta_next[7] * (vc + 1) / 3, not eta_next[7] / 3.
        vc_term = eta_next[7] * (state["vent_counter"] + 1) / 3
        coeff_v  = (price * P_vent
                    - zeta_cool * (eta_next[1] / 8 + eta_next[2] / 8)
                    - eta_vent * eta_next[3] / 70
                    + vc_term)

        p1 = 0 if coeff_p1 > 0 else P_max
        p2 = 0 if coeff_p2 > 0 else P_max
        v  = 0 if coeff_v  > 0 else 1

    # Hard overrule constraints (must hold regardless of the unconstrained optimum)
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

    return {"p1": p1, "p2": p2, "v": v}

def forward_pass(eta, initial_state):
    """Roll out N trajectories under the greedy policy w.r.t. the current VFA.
    Returns visited states (needed by the backward pass) plus actions/costs (for monitoring only)."""
    states  = [[None] * L for _ in range(N)]
    actions = [[None] * L for _ in range(N)]
    costs   = [[0.0] * L for _ in range(N)]

    for n in range(N):
        state = initial_state.copy()
        # Randomize initial exogenous so the visited-state distribution is diverse
        state["Occ1"]           = np.random.uniform(25, 35)
        state["Occ2"]           = np.random.uniform(15, 25)
        state["price_t"]        = np.random.uniform(2, 8)
        state["price_previous"] = np.random.uniform(2, 8)

        for t in range(L):
            states[n][t] = state.copy()
            action = solve_forward_pass_fast(state, eta)
            actions[n][t] = action
            costs[n][t] = compute_cost(state, action)

            if t < L - 1:
                exogenous = generate_exogenous(state)
                state = simulate_transition(state, action, exogenous)

    return states, actions, costs

def backward_pass(states, eta):
    """Approximate backward induction (Bellman backup, NOT policy evaluation).

    For each visited state s_{n,t}, the target V*(s_{n,t}) is obtained by
    RE-SOLVING the Bellman equation under the already-updated eta_{t+1}:

        V*(s_{n,t}) = min_u  c(s_{n,t}, u) + E_w[ eta_{t+1} . phi(f(s_{n,t}, u, w)) ]

    The minimization over u is done analytically by solve_forward_pass_fast,
    and the expectation is approximated with K Monte-Carlo exogenous samples.
    Then eta_t is fit by ridge regression on (phi(s_{n,t}), V*(s_{n,t})) pairs.

    Key fix vs. the previous version: the action used here is the optimum
    under the CURRENT VFA, not the action stored from the forward pass."""
    n_features = len(phi(initial_state))
    K = 10
    alpha = 1.0

    for t in reversed(range(L)):
        targets  = np.zeros(N)
        features = np.zeros((N, n_features))

        for n in range(N):
            s_t = states[n][t]
            features[n] = phi(s_t)

            # Re-solve Bellman at this state under the current VFA
            a_opt = solve_forward_pass_fast(s_t, eta)
            r = compute_cost(s_t, a_opt)

            if t == L - 1:
                targets[n] = r
            else:
                # Monte-Carlo estimate of the expected next-stage VFA
                future_cost = 0.0
                for _ in range(K):
                    w = generate_exogenous(s_t)
                    s_next = simulate_transition(s_t, a_opt, w)
                    future_cost += (eta[t + 1] @ phi(s_next)) / K
                targets[n] = r + future_cost

        # Ridge regression: (X'X + alpha*I) eta_t = X'y
        A = features.T @ features + alpha * np.eye(n_features)
        b = features.T @ targets
        eta[t] = np.linalg.solve(A, b)

    return eta

# === Training loop ===
n_features = len(phi(initial_state))
eta = np.ones((L, n_features))

N_iterations    = 100
convergence_tol = 1e-3
best_eta        = eta.copy()
best_cost       = float("inf")

for i in range(N_iterations):
    eta_old = eta.copy()

    states, _, costs = forward_pass(eta, initial_state)
    eta = backward_pass(states, eta)

    # Monitor convergence with two cheap signals:
    #  - avg cumulative cost over the N rollouts (policy quality)
    #  - relative change of eta (parameter convergence)
    avg_total_cost = np.mean([sum(costs[n]) for n in range(N)])
    eta_delta      = np.linalg.norm(eta - eta_old) / (np.linalg.norm(eta_old) + 1e-9)
    print(f"Iter {i:3d} | avg cost = {avg_total_cost:10.2f} | ||delta eta|| = {eta_delta:.4f}")

    if avg_total_cost < best_cost:
        best_cost = avg_total_cost
        best_eta  = eta.copy()

    if eta_delta < convergence_tol:
        print(f"Converged after {i+1} iterations")
        break

eta = best_eta

# === Save weights ===
np.save("eta_weights.npy", eta)

feature_names = ["intercept", "T1", "T2", "H", "Occ1", "Occ2",
                 "price_t", "vent_counter", "low_override_r1", "low_override_r2"]
df = pd.DataFrame(eta, columns=feature_names, index=[f"stage_{t}" for t in range(L)])
df.to_csv("eta_weights.csv")
print(df)