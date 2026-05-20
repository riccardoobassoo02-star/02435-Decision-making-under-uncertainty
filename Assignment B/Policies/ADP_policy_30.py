from pyomo.environ import *
from sklearn.cluster import KMeans
import numpy as np
import time
import csv

from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data


# Parameters extraction from system characteristics
data        = get_fixed_data()
T           = data["num_timeslots"]
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
L           = data["num_timeslots"]

M_temp = 50
M_hum  = 100

# eta_weights = np.load("eta_weights.npy")
eta_weights = np.load("eta_weights_hybrid_best.npy")

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

    # Reduce Monte Carlo samples to B representative scenarios
    X = np.column_stack([sample_prices, sample_occ1s, sample_occ2s])

    km = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
    labels = km.labels_
    centroids = km.cluster_centers_

    clusters = []

    for b in range(B):
        cluster_prob = np.sum(labels == b) / N_samples

        data = {
            "price": float(centroids[b, 0]),
            "occ_room_0": float(centroids[b, 1]),
            "occ_room_1": float(centroids[b, 2]),
            "prob": float(cluster_prob)
        }

        clusters.append(data)

    return clusters


def value_function(model, s_idx, scenarios, state):
    """
    Approximate next-state value:

    V(s_{t+1}) = eta[t+1]^T phi(s_{t+1})

    Aligned with the offline phi().
    """

    t = state["current_time"]
    w = eta_weights[t + 1]

    # State variables for scenario s in time t+1
    T1_next             = model.temp_next[0, s_idx]
    T2_next             = model.temp_next[1, s_idx]
    H_next              = model.humidity_next[s_idx]
    Occ1_next           = scenarios[s_idx]["occ_room_0"]
    Occ2_next           = scenarios[s_idx]["occ_room_1"]
    price_next          = scenarios[s_idx]["price"]
    price_previous_next = state["price_t"]
    vc_next             = model.vent_counter_next
    lr1_next            = model.overrule_next[0, s_idx]
    lr2_next            = model.overrule_next[1, s_idx]

    return (
        w[0] * 1.0
        + w[1] * (T1_next - 22) / 8
        + w[2] * (T2_next - 22) / 8
        + w[3] * (H_next - 30) / 70
        + w[4] * (Occ1_next - 20) / 30
        + w[5] * (Occ2_next - 10) / 20
        + w[6] * price_next / 12
        + w[7] * price_previous_next / 12
        + w[8] * vc_next / 3
        + w[9] * lr1_next
        + w[10] * lr2_next
    )


def solve_MILP(state, scenarios):
    # State variables at time t
    current_temp = [state["T1"], state["T2"]]
    current_humidity = state["H"]
    current_price = state["price_t"]
    current_vent_counter = state["vent_counter"]
    current_occ = [state["Occ1"], state["Occ2"]]
    t = state["current_time"]

    # Ventilation must remain on if the minimum up-time constraint is active
    v_inertia = 1 if 0 < current_vent_counter < min_up_time else 0

    # Ventilation status in the previous time step
    v_prev = 1 if current_vent_counter > 0 else 0

    # Previous low-temperature overrule states
    overrulers_prev = [
        int(state["low_override_r1"]),
        int(state["low_override_r2"])
    ]

    model = ConcreteModel()

    # Sets
    model.R = RangeSet(0, 1)
    model.Scenarios = RangeSet(0, len(scenarios) - 1)

    # Here-and-now decision variables
    model.p = Var(model.R, domain=NonNegativeReals, bounds=(0, P_max))
    model.v = Var(domain=Binary)

    # Current overrule helper variables
    model.y_high = Var(model.R, domain=Binary)
    model.y_low  = Var(model.R, domain=Binary)
    model.y_ok   = Var(model.R, domain=Binary)
    model.overrule = Var(model.R, domain=Binary)

    # Ventilation startup at time t
    model.s = Var(domain=Binary)

    # State variables for each scenario at t+1
    model.temp_next         = Var(model.R, model.Scenarios, domain=NonNegativeReals)
    model.humidity_next     = Var(model.Scenarios)
    model.vent_counter_next = Var(domain=NonNegativeIntegers)
    model.overrule_next     = Var(model.R, model.Scenarios, domain=Binary)
    model.y_low_next        = Var(model.R, model.Scenarios, domain=Binary)
    model.y_ok_next         = Var(model.R, model.Scenarios, domain=Binary)

    # Immediate electricity cost
    immediate_cost = current_price * (
        P_vent * model.v + model.p[0] + model.p[1]
    )

    # Current high-temperature detection
    model.c1 = Constraint(
        model.R,
        rule=lambda model, r:
            current_temp[r] >= T_high - M_temp * (1 - model.y_high[r])
    )

    model.c2 = Constraint(
        model.R,
        rule=lambda model, r:
            current_temp[r] <= T_high + M_temp * model.y_high[r]
    )

    # If current temperature is too high, heater is forced to zero
    model.c3 = Constraint(
        model.R,
        rule=lambda model, r:
            model.p[r] <= P_max * (1 - model.y_high[r])
    )

    # Current low-temperature detection
    model.c4 = Constraint(
        model.R,
        rule=lambda model, r:
            current_temp[r] <= T_low + M_temp * (1 - model.y_low[r])
    )

    model.c5 = Constraint(
        model.R,
        rule=lambda model, r:
            current_temp[r] >= T_low - M_temp * model.y_low[r]
    )

    # Current OK-temperature detection
    model.c6 = Constraint(
        model.R,
        rule=lambda model, r:
            current_temp[r] >= T_ok - M_temp * (1 - model.y_ok[r])
    )

    model.c7 = Constraint(
        model.R,
        rule=lambda model, r:
            current_temp[r] <= T_ok + M_temp * model.y_ok[r]
    )

    # Current low-temperature overrule logic
    model.c8 = Constraint(
        model.R,
        rule=lambda model, r:
            model.overrule[r] >= model.y_low[r]
    )

    model.c9 = Constraint(
        model.R,
        rule=lambda model, r:
            model.overrule[r] <= overrulers_prev[r] + model.y_low[r]
    )

    model.c10 = Constraint(
        model.R,
        rule=lambda model, r:
            model.p[r] >= P_max * model.overrule[r]
    )

    model.c11 = Constraint(
        model.R,
        rule=lambda model, r:
            model.overrule[r] >= overrulers_prev[r] - model.y_ok[r]
    )

    model.c12 = Constraint(
        model.R,
        rule=lambda model, r:
            model.overrule[r] <= 1 - model.y_ok[r]
    )

    # Ventilation startup and minimum up-time logic
    if t == 0:
        model.c13 = Constraint(expr=model.s >= model.v)
    else:
        model.c13 = Constraint(expr=model.s >= model.v - v_prev)

    model.c14 = Constraint(expr=model.s <= model.v)

    if t > 0:
        model.c15 = Constraint(expr=model.s <= 1 - v_prev)

    model.c16 = Constraint(expr=model.v >= v_inertia)

    # Humidity-triggered ventilation
    model.c17 = Constraint(
        expr=current_humidity <= H_high + M_hum * model.v
    )

    if t < L - 1:
        # Temperature dynamics at t+1 for each scenario
        model.c18 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda model, r, s:
                model.temp_next[r, s] ==
                current_temp[r]
                + zeta_exch * (current_temp[1 - r] - current_temp[r])
                - zeta_loss * (current_temp[r] - T_out[t])
                + zeta_conv * model.p[r]
                - zeta_cool * model.v
                + zeta_occ * current_occ[r]
        )

        # Humidity dynamics at t+1 for each scenario
        model.c19 = Constraint(
            model.Scenarios,
            rule=lambda model, s:
                model.humidity_next[s] ==
                current_humidity
                + eta_occ * (current_occ[0] + current_occ[1])
                - eta_vent * model.v
        )

        # Ventilation counter update
        model.c20 = Constraint(
            rule=lambda model:
                model.vent_counter_next == current_vent_counter * model.v + model.v
        )

        # Next low-temperature detection
        model.c21 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda model, r, s:
                model.temp_next[r, s] <= T_low + M_temp * (1 - model.y_low_next[r, s])
        )

        model.c22 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda model, r, s:
                model.temp_next[r, s] >= T_low - M_temp * model.y_low_next[r, s]
        )

        # Next OK-temperature detection
        model.c23 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda model, r, s:
                model.temp_next[r, s] >= T_ok - M_temp * (1 - model.y_ok_next[r, s])
        )

        model.c24 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda model, r, s:
                model.temp_next[r, s] <= T_ok + M_temp * model.y_ok_next[r, s]
        )

        # Next overrule activation if temperature is low
        model.c25 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda model, r, s:
                model.overrule_next[r, s] >= model.y_low_next[r, s]
        )

        model.c26 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda model, r, s:
                model.overrule_next[r, s] <= model.overrule[r] + model.y_low_next[r, s]
        )

        # Next overrule deactivation if temperature is OK
        model.c27 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda model, r, s:
                model.overrule_next[r, s] >= model.overrule[r] - model.y_ok_next[r, s]
        )

        model.c28 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda model, r, s:
                model.overrule_next[r, s] <= 1 - model.y_ok_next[r, s]
        )

        # ADP objective: immediate cost plus expected approximate value of next state
        expected_future_value = sum(
            scenarios[s]["prob"] * value_function(model, s, scenarios, state)
            for s in range(len(scenarios))
        )

        model.obj = Objective(
            expr=immediate_cost + expected_future_value,
            sense=minimize
        )

    else:
        # Last time slot: no future value remains
        model.obj = Objective(
            expr=immediate_cost,
            sense=minimize
        )

    solver = SolverFactory("gurobi")
    result = solver.solve(model, options={"OutputFlag": 0})

    if result.solver.termination_condition != TerminationCondition.optimal:
        return 0.0, 0.0, 0

    p1 = value(model.p[0])
    p2 = value(model.p[1])
    v = int(value(model.v) > 0.5)

    return p1, p2, v


def select_action(state):
    start_time = time.time()

    state = state.copy()

    if "price_previous" not in state:
        state["price_previous"] = state["price_t"]

    t = state["current_time"]

    if t == L - 1:
        scenarios = []
    else:
        scenarios = generate_samples(state, B=5, N_samples=500)

    try:
        p1, p2, v = solve_MILP(state, scenarios)
    except Exception:
        p1, p2, v = 0.0, 0.0, 0

    HereAndNowActions = {
        "HeatPowerRoom1": p1,
        "HeatPowerRoom2": p2,
        "VentilationON": v
    }

    return HereAndNowActions