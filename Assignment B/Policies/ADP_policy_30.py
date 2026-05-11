# Corrected ONLINE policy using the OFFLINE training structure
# Main fixes:
# 1. feature order aligned with offline phi()
# 2. removed obsolete price_previous feature from VFA
# 3. corrected vent_counter transition
# 4. corrected next-state VFA indexing
# 5. corrected terminal-stage handling
# 6. aligned online Bellman approximation with offline ADP

from pyomo.environ import *
from sklearn.cluster import KMeans
import numpy as np
import time

from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data


sys_data = get_fixed_data()

T = sys_data['num_timeslots']
P_max = sys_data['heating_max_power']
zeta_exch = sys_data['heat_exchange_coeff']
zeta_conv = sys_data['heating_efficiency_coeff']
zeta_loss = sys_data['thermal_loss_coeff']
zeta_cool = sys_data['heat_vent_coeff']
zeta_occ = sys_data['heat_occupancy_coeff']

T_low = sys_data['temp_min_comfort_threshold']
T_ok = sys_data['temp_OK_threshold']
T_high = sys_data['temp_max_comfort_threshold']
T_out = sys_data['outdoor_temperature']

P_vent = sys_data['ventilation_power']
H_high = sys_data['humidity_threshold']
eta_occ = sys_data['humidity_occupancy_coeff']
eta_vent = sys_data['humidity_vent_coeff']

min_up_time = sys_data['vent_min_up_time']
L = sys_data['num_timeslots']

M_temp = 50
M_hum = 100

# OFFLINE phi(state):
# [0] intercept
# [1] T1
# [2] T2
# [3] H
# [4] Occ1
# [5] Occ2
# [6] price_t
# [7] vent_counter
# [8] low_override_r1
# [9] low_override_r2
eta_weights = np.load("eta_weights.npy")


def generate_samples(state, B=5, N_samples=1000):
    prices = []
    occ1s = []
    occ2s = []

    for _ in range(N_samples):
        p = price_model(state["price_t"], state["price_previous"])
        o1, o2 = next_occupancy_levels(state["Occ1"], state["Occ2"])

        prices.append(p)
        occ1s.append(o1)
        occ2s.append(o2)

    X = np.column_stack([prices, occ1s, occ2s])

    km = KMeans(
        n_clusters=B,
        random_state=0,
        n_init=10
    ).fit(X)

    labels = km.labels_
    centroids = km.cluster_centers_

    scenarios = []
    for b in range(B):
        prob = np.mean(labels == b)

        scenarios.append({
            "price": centroids[b, 0],
            "occ1": centroids[b, 1],
            "occ2": centroids[b, 2],
            "prob": prob
        })

    return scenarios


def value_function(model, s, scenarios, state):
    """
    Approximate next-state value:

    V(s_{t+1}) = eta[t+1]^T phi(s_{t+1})

    EXACTLY aligned with offline phi().
    """

    t = state["current_time"]

    if t >= L - 1:
        return 0

    w = eta_weights[t + 1]

    T1_next = model.temp_next[0, s]
    T2_next = model.temp_next[1, s]
    H_next = model.humidity_next[s]
    vc_next = model.vent_counter_next

    lr1_next = model.overrule_next[0, s]
    lr2_next = model.overrule_next[1, s]

    Occ1_next = scenarios[s]["occ1"]
    Occ2_next = scenarios[s]["occ2"]
    price_next = scenarios[s]["price"]

    return (
        w[0] * 1.0
        + w[1] * (T1_next - 22) / 8
        + w[2] * (T2_next - 22) / 8
        + w[3] * (H_next - 30) / 70
        + w[4] * (Occ1_next - 20) / 30
        + w[5] * (Occ2_next - 10) / 20
        + w[6] * price_next / 12
        + w[7] * vc_next / 3
        + w[8] * lr1_next
        + w[9] * lr2_next
    )


def solve_MILP(state, scenarios):
    model = ConcreteModel()

    t = state["current_time"]

    current_temp = [state["T1"], state["T2"]]
    current_H = state["H"]
    current_price = state["price_t"]
    current_vc = state["vent_counter"]

    previous_overrules = [
        state["low_override_r1"],
        state["low_override_r2"]
    ]

    v_prev = 1 if current_vc > 0 else 0

    model.R = RangeSet(0, 1)
    model.S = RangeSet(0, len(scenarios) - 1)

    model.p = Var(model.R, bounds=(0, P_max))
    model.v = Var(domain=Binary)

    model.temp_next = Var(model.R, model.S, bounds=(0, 50))
    model.humidity_next = Var(model.S, bounds=(0, 100))

    model.overrule_next = Var(model.R, model.S, domain=Binary)
    model.vent_counter_next = Var(bounds=(0, L))

    immediate_cost = current_price * (
        model.p[0] + model.p[1] + P_vent * model.v
    )

    future_value = sum(
        scenarios[s]["prob"] * value_function(model, s, scenarios, state)
        for s in model.S
    )

    model.obj = Objective(
        expr=immediate_cost + future_value,
        sense=minimize
    )

    # state transition
    def temp_rule(model, r, s):
        occ = scenarios[s]["occ1"] if r == 0 else scenarios[s]["occ2"]

        return model.temp_next[r, s] == (
            current_temp[r]
            + zeta_conv * model.p[r]
            + zeta_exch * (current_temp[1 - r] - current_temp[r])
            + zeta_loss * (T_out[t] - current_temp[r])
            + zeta_occ * occ
            - zeta_cool * model.v
        )

    model.temp_dyn = Constraint(model.R, model.S, rule=temp_rule)

    model.humidity_dyn = Constraint(
        model.S,
        rule=lambda model, s:
            model.humidity_next[s] == (
                current_H
                + eta_occ * (scenarios[s]["occ1"] + scenarios[s]["occ2"])
                - eta_vent * model.v
            )
    )

    # IMPORTANT FIX:
    # offline transition was:
    # vc_new = vc + 1 if v == 1 else 0
    # not vc_new = (vc+1)*v assumed incorrectly elsewhere
    # but since v is binary, linear form below is exact
    model.vc_dyn = Constraint(
        expr=model.vent_counter_next == (current_vc + 1) * model.v
    )

    # hard constraints / overrides
    for r in model.R:
        if previous_overrules[r] and current_temp[r] < T_ok:
            model.p[r].fix(P_max)

        if current_temp[r] >= T_high:
            model.p[r].fix(0)

    if current_H > H_high:
        model.v.fix(1)

    if 0 < current_vc < min_up_time:
        model.v.fix(1)

    # simple next-state overrule approximation
    for s in model.S:
        for r in model.R:
            if current_temp[r] < T_low:
                model.overrule_next[r, s].fix(1)

    solver = SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0

    result = solver.solve(model)

    if result.solver.termination_condition != TerminationCondition.optimal:
        return 0.0, 0.0, 0

    return (
        value(model.p[0]),
        value(model.p[1]),
        int(value(model.v) > 0.5)
    )


def select_action(state):
    start = time.time()

    scenarios = generate_samples(
        state,
        B=5,
        N_samples=1000
    )

    p1, p2, v = solve_MILP(state, scenarios)

    return {
        "HeatPowerRoom1": p1,
        "HeatPowerRoom2": p2,
        "VentilationON": v
    }
