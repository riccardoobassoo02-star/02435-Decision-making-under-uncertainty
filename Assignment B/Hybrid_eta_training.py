# This must be placed before importing numpy, sklearn, or Hybrid_policy_30.
# It avoids the repeated KMeans/MKL warning on Windows.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL"
)

from pathlib import Path
import numpy as np
import pandas as pd

# Import the whole module, not only the functions.
# solve_hybrid() and terminal_vfa() use module-level eta_weights.
import Policies.Hybrid_policy_30 as hybrid_module

from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data


# DATA AND PARAMETERS
data = get_fixed_data()

N = 30
T = data["num_timeslots"]

P_max       = data["heating_max_power"]
zeta_exch   = data["heat_exchange_coeff"]
zeta_conv   = data["heating_efficiency_coeff"]
zeta_loss   = data["thermal_loss_coeff"]
zeta_cool   = data["heat_vent_coeff"]
zeta_occ    = data["heat_occupancy_coeff"]
T_low       = data["temp_min_comfort_threshold"]
T_ok        = data["temp_OK_threshold"]
T_high      = data["temp_max_comfort_threshold"]
T_out       = data["outdoor_temperature"]
P_vent      = data["ventilation_power"]
H_high      = data["humidity_threshold"]
eta_occ     = data["humidity_occupancy_coeff"]
eta_vent    = data["humidity_vent_coeff"]
min_up_time = data["vent_min_up_time"]

np.random.seed(42)

initial_state = {
    "T1":              data["T1"],
    "T2":              data["T2"],
    "H":               data["H"],
    "vent_counter":    data["vent_counter"],
    "low_override_r1": data["low_override_r1"],
    "low_override_r2": data["low_override_r2"],
    "current_time":    0,
    "Occ1":            data["Occ1"],
    "Occ2":            data["Occ2"],
    "price_t":         data["price_t"],
    "price_previous":  data["price_previous"],
}



# VALUE FUNCTION FEATURES
def phi(state):
    return np.array([
        1.0,
        (state["T1"] - 22) / 8,
        (state["T2"] - 22) / 8,
        (state["H"] - 30) / 70,
        (state["Occ1"] - 20) / 30,
        (state["Occ2"] - 10) / 20,
        state["price_t"] / 12,
        state["price_previous"] / 12,
        state["vent_counter"] / 3,
        float(state["low_override_r1"]),
        float(state["low_override_r2"]),
    ], dtype=float)


# EXOGENOUS PROCESS
def generate_exogenous(state):
    price_new = price_model(state["price_t"], state["price_previous"])
    occ1_new, occ2_new = next_occupancy_levels(state["Occ1"], state["Occ2"])

    return {
        "price_t":        price_new,
        "price_previous": state["price_t"],
        "Occ1":           occ1_new,
        "Occ2":           occ2_new,
    }


# SYSTEM DYNAMICS USED IN TRAINING ENVIRONMENT
def apply_overrule(state, action):
    p1 = float(action["p1"])
    p2 = float(action["p2"])
    v  = int(action["v"])

    # Low temperature overrule
    if state["low_override_r1"] and state["T1"] < T_ok:
        p1 = P_max
    if state["low_override_r2"] and state["T2"] < T_ok:
        p2 = P_max

    # High temperature overrule
    if state["T1"] > T_high:
        p1 = 0.0
    if state["T2"] > T_high:
        p2 = 0.0

    # Humidity overrule
    if state["H"] > H_high:
        v = 1

    # Ventilation inertia
    if 0 < state["vent_counter"] < min_up_time:
        v = 1

    # Safety clipping
    p1 = min(max(p1, 0.0), P_max)
    p2 = min(max(p2, 0.0), P_max)
    v  = int(v > 0.5)

    return {"p1": p1, "p2": p2, "v": v}


def update_low_override(T_new, old_override):
    if int(old_override) == 0:
        return 1 if T_new < T_low else 0
    else:
        return 0 if T_new >= T_ok else 1


def simulate_transition(state, action, exogenous):
    action = apply_overrule(state, action)

    p1 = action["p1"]
    p2 = action["p2"]
    v  = action["v"]

    t = state["current_time"]
    t_out = T_out[min(t, len(T_out) - 1)]

    T1 = state["T1"]
    T2 = state["T2"]
    H  = state["H"]

    T1_new = (
        T1
        + zeta_exch * (T2 - T1)
        - zeta_loss * (T1 - t_out)
        + zeta_conv * p1
        - zeta_cool * v
        + zeta_occ * state["Occ1"]
    )

    T2_new = (
        T2
        + zeta_exch * (T1 - T2)
        - zeta_loss * (T2 - t_out)
        + zeta_conv * p2
        - zeta_cool * v
        + zeta_occ * state["Occ2"]
    )

    H_new = (
        H
        + eta_occ * (state["Occ1"] + state["Occ2"])
        - eta_vent * v
    )

    vent_counter_new = state["vent_counter"] + 1 if v == 1 else 0

    return {
        "T1":              T1_new,
        "T2":              T2_new,
        "H":               H_new,
        "vent_counter":    vent_counter_new,
        "low_override_r1": update_low_override(T1_new, state["low_override_r1"]),
        "low_override_r2": update_low_override(T2_new, state["low_override_r2"]),
        "current_time":    t + 1,
        "Occ1":            exogenous["Occ1"],
        "Occ2":            exogenous["Occ2"],
        "price_t":         exogenous["price_t"],
        "price_previous":  state["price_t"],
    }


def compute_cost(state, action):
    action = apply_overrule(state, action)

    return state["price_t"] * (
        action["p1"] + action["p2"] + P_vent * action["v"]
    )


# HYBRID POLICY USED DURING TRAINING
def solve_hybrid_for_training(
    state,
    eta,
    L_lookahead=4,
    B=3,
    N_samples_tree=100,
):
    """
    Solves the same hybrid policy as the final policy:

        SP cost over lookahead horizon
        +
        terminal VFA at the leaf nodes.

    The only difference is that during training we inject the current eta
    into the Hybrid_policy_30 module.
    """

    # Critical fix:
    # solve_hybrid() reads eta_weights from the Hybrid_policy_30 module.
    hybrid_module.eta_weights = eta

    t = state["current_time"]
    L_eff = min(L_lookahead, T - 1 - t)

    if L_eff <= 0:
        return {"p1": 0.0, "p2": 0.0, "v": 0}

    try:
        nodes = hybrid_module.build_tree(
            state,
            L=L_eff,
            B=B,
            N_samples=N_samples_tree,
        )

        p1, p2, v = hybrid_module.solve_hybrid(state, nodes)

        return {
            "p1": float(p1),
            "p2": float(p2),
            "v":  int(v),
        }

    except Exception as e:
        print(f"[WARNING] Hybrid solve failed during training: {e}")
        return {"p1": 0.0, "p2": 0.0, "v": 0}


# FORWARD PASS
def forward_pass(eta, initial_state):
    """
    Roll out N trajectories using the current hybrid policy.
    The resulting actions are then kept fixed in the backward pass.
    """

    states  = [[None] * T for _ in range(N)]
    actions = [[None] * T for _ in range(N)]
    costs   = [[0.0]  * T for _ in range(N)]

    for n in range(N):
        state = initial_state.copy()

        # Randomize initial exogenous components for better state coverage.
        state["Occ1"] = np.random.uniform(25, 35)
        state["Occ2"] = np.random.uniform(15, 25)
        state["price_t"] = np.random.uniform(2, 8)
        state["price_previous"] = np.random.uniform(2, 8)

        for t in range(T):
            states[n][t] = state.copy()

            action = solve_hybrid_for_training(
                state,
                eta,
                L_lookahead=4,
                B=3,
                N_samples_tree=100,
            )

            action = apply_overrule(state, action)

            actions[n][t] = action
            costs[n][t] = compute_cost(state, action)

            if t < T - 1:
                exogenous = generate_exogenous(state)
                state = simulate_transition(state, action, exogenous)

    return states, actions, costs


# BACKWARD PASS
def backward_pass(states, actions, costs, eta_current, alpha=1.0, K=20):
    """
    Policy evaluation with fixed actions from the forward pass.

    For each time t, we fit eta_t to:

        target_t = cost_t + E[eta_{t+1}^T phi(s_{t+1})]

    This keeps the same training logic as the ADP training,
    but the policy used to generate the forward-pass actions is hybrid.
    """

    n_features = len(phi(initial_state))
    eta_new = eta_current.copy()

    for t in reversed(range(T)):
        features = np.zeros((N, n_features))
        targets  = np.zeros(N)

        for n in range(N):
            state = states[n][t]
            action = actions[n][t]
            r = costs[n][t]

            features[n, :] = phi(state)

            if t == T - 1:
                targets[n] = r
            else:
                future_value = 0.0

                for _ in range(K):
                    exogenous = generate_exogenous(state)
                    s_next = simulate_transition(state, action, exogenous)
                    future_value += eta_new[t + 1] @ phi(s_next)

                future_value /= K
                targets[n] = r + future_value

        # Ridge regression:
        #
        # eta_t = argmin_eta sum_n (eta^T phi(s_n,t) - target_n)^2
        #         + alpha * ||eta||^2
        #
        A = features.T @ features + alpha * np.eye(n_features)
        b = features.T @ targets

        eta_new[t, :] = np.linalg.solve(A, b)

    return eta_new



# FIT ERROR
def compute_avg_fit_error(states, actions, costs, eta, K=10):
    total_error = 0.0

    for t in range(T):
        for n in range(N):
            state = states[n][t]
            action = actions[n][t]
            r = costs[n][t]

            if t == T - 1:
                target = r
            else:
                future_value = 0.0

                for _ in range(K):
                    exogenous = generate_exogenous(state)
                    s_next = simulate_transition(state, action, exogenous)
                    future_value += eta[t + 1] @ phi(s_next)

                future_value /= K
                target = r + future_value

            prediction = eta[t] @ phi(state)
            total_error += (prediction - target) ** 2

    return total_error / (T * N)


# TRAINING LOOP
n_features = len(phi(initial_state))

# Warm start from existing ADP eta if available.
# This is better than starting from ones.
if Path("eta_weights.npy").exists():
    eta = np.load("eta_weights.npy")
    print("Loaded eta_weights.npy as warm start.")
else:
    eta = np.ones((T, n_features))
    print("No eta_weights.npy found. Starting from ones.")

best_eta = eta.copy()
best_error = float("inf")

N_iterations = 20
J = 1
beta = 0.05
convergence_tol = 30.0

for i in range(N_iterations):
    print(f"\nIteration {i + 1}/{N_iterations}")

    # 1. Forward pass using current hybrid policy.
    states, actions, costs = forward_pass(eta, initial_state)

    # 2. Backward policy evaluation.
    eta_candidate = eta.copy()

    for j in range(J):
        eta_candidate = backward_pass(
            states,
            actions,
            costs,
            eta_candidate,
            alpha=1.0,
            K=10,
        )

    # 3. Smooth update.
    eta = (1.0 - beta) * eta + beta * eta_candidate

    # 4. Diagnostics.
    avg_error = compute_avg_fit_error(states, actions, costs, eta, K=10)
    avg_policy_cost = np.mean([sum(costs[n]) for n in range(N)])

    print(f"  avg fit error:   {avg_error:.4f}")
    print(f"  avg policy cost: {avg_policy_cost:.4f}")

    if avg_error < best_error:
        best_error = avg_error
        best_eta = eta.copy()
        np.save("eta_weights_hybrid_best.npy", best_eta)

    if avg_error < convergence_tol:
        print(f"Converged at iteration {i + 1}")
        break


# SAVE RESULTS
np.save("eta_weights_hybrid.npy", eta)

feature_names = [
    "Intercept",
    "T1",
    "T2",
    "H",
    "Occ1",
    "Occ2",
    "price_t",
    "price_previous",
    "vent_counter",
    "low_override_r1",
    "low_override_r2",
]

df = pd.DataFrame(
    eta,
    columns=feature_names,
    index=[f"stage_{t}" for t in range(T)],
)

df.to_csv("eta_weights_hybrid.csv")

print("\nFinal eta:")
print(df)

print(f"\nBest fit error: {best_error:.4f}")
print("Saved:")
print("  eta_weights_hybrid.npy")
print("  eta_weights_hybrid_best.npy")
print("  eta_weights_hybrid.csv")