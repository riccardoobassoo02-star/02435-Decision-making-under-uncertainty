from pathlib import Path
from pyomo.environ import *
import numpy as np
import pandas as pd
from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data

data = get_fixed_data()
N               = 50
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
M_temp = 50    # big-M for temperature constraints
M_hum  = 100   # big-M for humidity constraints
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


def solve_forward_pass_milp(state, eta, K_policy=5):
    """
    MILP policy improvement for the forward pass (Variant B — Task 4).

    At state s_t, choose the action u_t = (p1, p2, v) by solving:

        min  immediate_cost(s_t, u_t)
             + (1/K) * sum_k eta[t+1]^T phi(s_{t+1}^k)

    The MILP uses the SAME constraint structure as the Task 1 OIH MILP:
    big-M cutoffs for temp_high / temp_low / temp_ok, an overrule controller,
    humidity-triggered ventilation, and the ventilation minimum up-time logic.

    The only difference from OIH: we optimize over a SINGLE here-and-now
    step and approximate the cost of the future via eta[t+1]^T phi(s_{t+1}).
    """
    t = state["current_time"]
    price = state["price_t"]

    # ────────────────────────────────────────────────────────────────────
    # Model
    # ────────────────────────────────────────────────────────────────────
    model = ConcreteModel()

    # Sets
    model.R = RangeSet(0, 1)               # rooms
    model.S = RangeSet(0, K_policy - 1)    # exogenous scenarios

    # ────────────────────────────────────────────────────────────────────
    # Decision variables (here-and-now, at time t)
    # ────────────────────────────────────────────────────────────────────
    model.p = Var(model.R, domain=NonNegativeReals, bounds=(0, P_max))  # heating power
    model.v = Var(domain=Binary)                                        # ventilation ON/OFF

    # ────────────────────────────────────────────────────────────────────
    # Next-state variables (at time t+1) — only the endogenous part
    # depends on (p, v); the exogenous part (Occ, price) is sampled.
    # Since dynamics are linear in (p, v), next-state is scenario-independent
    # for the endogenous variables.
    # ────────────────────────────────────────────────────────────────────
    model.temp = Var(model.R, domain=Reals)   # indoor temperature in room r at t+1
    model.hum  = Var(domain=NonNegativeReals) # indoor humidity at t+1
    model.vc   = Var(domain=NonNegativeReals) # ventilation counter at t+1

    # Overrule controller state at t+1 (needed for phi(s_{t+1}))
    model.temp_low = Var(model.R, domain=Binary)   # 1 if temp[r] < T_low
    model.temp_ok  = Var(model.R, domain=Binary)   # 1 if temp[r] > T_ok
    model.overrule = Var(model.R, domain=Binary)   # next overrule state

    # ────────────────────────────────────────────────────────────────────
    # Apply current overrule constraints (state-dependent action restrictions)
    # ────────────────────────────────────────────────────────────────────
    if 0 < state["vent_counter"] < min_up_time:
        model.v.fix(1)

    if state["H"] > H_high:
        model.v.fix(1)

    if state["T1"] > T_high:
        model.p[0].fix(0)
    elif state["low_override_r1"] and state["T1"] < T_ok:
        model.p[0].fix(P_max)

    if state["T2"] > T_high:
        model.p[1].fix(0)
    elif state["low_override_r2"] and state["T2"] < T_ok:
        model.p[1].fix(P_max)

    # ────────────────────────────────────────────────────────────────────
    # Constraints — same structure as Task 1 OIH MILP
    # ────────────────────────────────────────────────────────────────────

    # 1-2. Temperature dynamics (one-step transition from state to t+1)
    model.c_temp = Constraint(model.R, rule=lambda m, r:
        m.temp[r] == (state["T1"] if r == 0 else state["T2"])
                     + zeta_exch * ((state["T2"] if r == 0 else state["T1"])
                                    - (state["T1"] if r == 0 else state["T2"]))
                     - zeta_loss * ((state["T1"] if r == 0 else state["T2"])
                                    - T_out[t])
                     + zeta_conv * m.p[r]
                     - zeta_cool * m.v
                     + zeta_occ * (state["Occ1"] if r == 0 else state["Occ2"]))

    # 3-4. Humidity dynamics (one-step transition from state to t+1)
    model.c_hum = Constraint(expr=
        model.hum == state["H"]
                     + eta_occ * (state["Occ1"] + state["Occ2"])
                     - eta_vent * model.v)

    # 5. Ventilation counter dynamics (linearized: vc_new = v * (vc + 1))
    model.c_vc = Constraint(expr=
        model.vc == state["vent_counter"] * model.v + model.v)

    # 8-9. Detecting when temperature at t+1 is below T_low
    model.c8 = Constraint(model.R, rule=lambda m, r:
        m.temp[r] <= T_low + M_temp * (1 - m.temp_low[r]))
    model.c9 = Constraint(model.R, rule=lambda m, r:
        m.temp[r] >= T_low - M_temp * m.temp_low[r])

    # 10-11. Detecting when temperature at t+1 is above T_ok
    model.c10 = Constraint(model.R, rule=lambda m, r:
        m.temp[r] >= T_ok - M_temp * (1 - m.temp_ok[r]))
    model.c11 = Constraint(model.R, rule=lambda m, r:
        m.temp[r] <= T_ok + M_temp * m.temp_ok[r])

    # 12-16. Overrule controller update (hysteresis)
    # If current overrule = 0: next overrule = temp_low (activate if T < T_low)
    # If current overrule = 1: next overrule = 1 - temp_ok (release if T >= T_ok)
    model.c_overrule = ConstraintList()
    for r in range(2):
        current_overrule = state["low_override_r1"] if r == 0 else state["low_override_r2"]
        if int(current_overrule) == 0:
            model.c_overrule.add(model.overrule[r] == model.temp_low[r])
        else:
            model.c_overrule.add(model.overrule[r] == 1 - model.temp_ok[r])

    # 21. Humidity-triggered ventilation already enforced above via model.v.fix(1)
    # when state["H"] > H_high. No additional constraint needed.

    # ────────────────────────────────────────────────────────────────────
    # Objective: immediate cost + expected approximate future value
    # ────────────────────────────────────────────────────────────────────
    immediate_cost = price * (model.p[0] + model.p[1] + P_vent * model.v)

    if t == L - 1:
        # Terminal stage: no future value
        model.obj = Objective(expr=immediate_cost, sense=minimize)
    else:
        eta_next = eta[t + 1]
        # Sample K_policy exogenous scenarios (only exogenous part varies)
        scenarios = [generate_exogenous(state) for _ in range(K_policy)]

        future_value = 0.0
        for s in range(K_policy):
            price_next     = scenarios[s]["price_t"]
            price_prev_nxt = state["price_t"]   # price_previous at t+1 = price at t
            occ1_next      = scenarios[s]["Occ1"]
            occ2_next      = scenarios[s]["Occ2"]

            future_value += (1.0 / K_policy) * (
                  eta_next[0]  * 1.0
                + eta_next[1]  * (model.temp[0] - 22) / 8
                + eta_next[2]  * (model.temp[1] - 22) / 8
                + eta_next[3]  * (model.hum - 30) / 70
                + eta_next[4]  * (occ1_next - 20) / 30
                + eta_next[5]  * (occ2_next - 10) / 20
                + eta_next[6]  * price_next / 12
                + eta_next[7]  * price_prev_nxt / 12
                + eta_next[8]  * model.vc / 3
                + eta_next[9]  * model.overrule[0]
                + eta_next[10] * model.overrule[1]
            )

        model.obj = Objective(expr=immediate_cost + future_value, sense=minimize)

    # ────────────────────────────────────────────────────────────────────
    # Solve
    # ────────────────────────────────────────────────────────────────────
    solver = SolverFactory("gurobi")
    solver.solve(model, options={"OutputFlag": 0})

    return {
        "p1": float(value(model.p[0])),
        "p2": float(value(model.p[1])),
        "v":  int(value(model.v) > 0.5)
    }

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
            action = solve_forward_pass_milp(state, eta, K_policy=5)
            if action is None:
                raise RuntimeError(
                    f"solve_forward_pass_milp returned None. "
                    f"n={n}, t={t}, state={state}"
                )
            cost          = compute_cost(state, action)
            actions[n][t] = action
            costs[n][t]   = cost

            if t < L - 1:
                exogenous = generate_exogenous(state)
                state     = simulate_transition(state, action, exogenous)

    return states, actions, costs


def backward_pass(states, actions, costs, eta_current):
    """
    Backward pass — Policy Evaluation (Variant B).

    Actions are fixed from the forward pass.
    The value function is updated backward in time using ridge regression.

    At each stage t, the regression target is:

        cost_t + E[V_hat_{t+1}(s_{t+1})]

    where V_hat_{t+1} uses the most recent estimate eta_new[t+1].
    This is a stagewise backward update, not a policy improvement step.

    Returns a new eta array.
    """
    n_features = len(phi(initial_state))
    alpha      = 1     # Ridge regression regularization
    K          = 10    # Number of exogenous samples per (n, t) for target estimation

    eta_new = eta_current.copy()

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
                # Target = immediate cost + E[V_hat_{t+1}(s_{t+1})]
                # Since we move backward in time, eta_new[t+1] has already been updated.
                future_cost = 0.0
                samples = [generate_exogenous(states[n][t]) for _ in range(K)]

                for k in range(K):
                    s_next = simulate_transition(
                        states[n][t],
                        actions[n][t],
                        samples[k]
                    )
                    future_cost += (1.0 / K) * (eta_new[t + 1] @ phi(s_next))

                targets[n] = r + future_cost

        # Ridge regression:
        # eta_new[t] = argmin_eta sum_n (eta^T phi(s_n,t) - target_n)^2
        #              + alpha * ||eta||^2
        A = features.T @ features + alpha * np.eye(n_features)
        b = features.T @ targets
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


# TRAINING LOOP, Variant B: Approximate Policy Iteration

n_features     = len(phi(initial_state))
eta            = np.ones((L, n_features))   # Initialize eta
best_eta       = eta.copy()
best_error     = float("inf")
N_iterations   = 100
J              = 3                          # Number of policy evaluation sweeps per iteration
convergence_tol = 30

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

    if avg_error < convergence_tol:
        print(f"Converged at iteration {i+1}")
        break

# SAVE RESULTS
np.save("eta_weights.npy", eta)

feature_names = [
    "Intercept", "T1", "T2", "H",
    "Occ1", "Occ2", "price_t", "price_previous",
    "vent_counter", "low_override_r1", "low_override_r2"
]
df = pd.DataFrame(eta, columns=feature_names, index=[f"stage_{t}" for t in range(L)])
df.to_csv("eta_weights.csv")
print(df)