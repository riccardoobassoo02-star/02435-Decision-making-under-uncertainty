from pyomo.environ import *
from sklearn.cluster import KMeans
import numpy as np
import time

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

    # Reduce Monte Carlo samples to B representative scenarios
    X = np.column_stack([sample_prices, sample_occ1s, sample_occ2s])

    km = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
    labels = km.labels_
    centroids = km.cluster_centers_

    clusters = []

    for b in range(B):
        cluster_prob = np.sum(labels == b) / N_samples

        data = {
            "price": centroids[b, 0],
            "occ_room_0": centroids[b, 1],
            "occ_room_1": centroids[b, 2],
            "prob": cluster_prob
        }

        clusters.append(data)

    return clusters


def value_function(model, scenario_idx, scenarios, current_state):
    # Approximate value of the next state using the same features used offline
    hour = current_state["current_time"]

    temp_r1_next      = model.temp_next[0, scenario_idx]
    temp_r2_next      = model.temp_next[1, scenario_idx]
    humidity_next     = model.humidity_next[scenario_idx]
    occ_room_0_next   = scenarios[scenario_idx]["occ_room_0"]
    occ_room_1_next   = scenarios[scenario_idx]["occ_room_1"]
    price_next        = scenarios[scenario_idx]["price"]
    prev_price_next   = current_state["price_t"]
    vent_counter_next = model.vent_counter_next
    overrule_r1_next  = model.overrule_next[0, scenario_idx]
    overrule_r2_next  = model.overrule_next[1, scenario_idx]

    # Same phi(state) used in the offline training
    next_features = [
        1,
        (temp_r1_next - 22) / 8,
        (temp_r2_next - 22) / 8,
        (humidity_next - 30) / 70,
        (occ_room_0_next - 20) / 30,
        (occ_room_1_next - 10) / 20,
        price_next / 12,
        prev_price_next / 12,
        vent_counter_next / 3,
        overrule_r1_next,
        overrule_r2_next
    ]

    # Use eta of the next stage, because the objective is c_t + V_{t+1}(s_{t+1})
    value = sum(
        eta_weights[hour + 1, i] * next_features[i]
        for i in range(len(next_features))
    )

    return value


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

    # Ventilation startup variable
    model.s = Var(domain=Binary)

    # Next-state variables
    model.temp_next = Var(model.R, model.Scenarios, domain=Reals)
    model.humidity_next = Var(model.Scenarios, domain=Reals)
    model.vent_counter_next = Var(domain=NonNegativeReals)

    model.overrule_next = Var(model.R, model.Scenarios, domain=Binary)
    model.y_low_next = Var(model.R, model.Scenarios, domain=Binary)
    model.y_ok_next = Var(model.R, model.Scenarios, domain=Binary)

    # Immediate electricity cost
    immediate_cost = current_price * (
        model.p[0] + model.p[1] + P_vent * model.v
    )

    # Current high-temperature detection
    model.c1 = Constraint(
        model.R,
        rule=lambda m, r:
            current_temp[r] >= T_high - M_temp * (1 - m.y_high[r])
    )

    model.c2 = Constraint(
        model.R,
        rule=lambda m, r:
            current_temp[r] <= T_high + M_temp * m.y_high[r]
    )

    # If current temperature is too high, heater is forced to zero
    model.c3 = Constraint(
        model.R,
        rule=lambda m, r:
            m.p[r] <= P_max * (1 - m.y_high[r])
    )

    # Current low-temperature detection
    model.c4 = Constraint(
        model.R,
        rule=lambda m, r:
            current_temp[r] <= T_low + M_temp * (1 - m.y_low[r])
    )

    model.c5 = Constraint(
        model.R,
        rule=lambda m, r:
            current_temp[r] >= T_low - M_temp * m.y_low[r]
    )

    # Current OK-temperature detection
    model.c6 = Constraint(
        model.R,
        rule=lambda m, r:
            current_temp[r] >= T_ok - M_temp * (1 - m.y_ok[r])
    )

    model.c7 = Constraint(
        model.R,
        rule=lambda m, r:
            current_temp[r] <= T_ok + M_temp * m.y_ok[r]
    )

    # Current low-temperature overrule logic
    model.c8 = Constraint(
        model.R,
        rule=lambda m, r:
            m.overrule[r] >= m.y_low[r]
    )

    model.c9 = Constraint(
        model.R,
        rule=lambda m, r:
            m.overrule[r] <= overrulers_prev[r] + m.y_low[r]
    )

    model.c10 = Constraint(
        model.R,
        rule=lambda m, r:
            m.p[r] >= P_max * m.overrule[r]
    )

    model.c11 = Constraint(
        model.R,
        rule=lambda m, r:
            m.overrule[r] >= overrulers_prev[r] - m.y_ok[r]
    )

    model.c12 = Constraint(
        model.R,
        rule=lambda m, r:
            m.overrule[r] <= 1 - m.y_ok[r]
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
        # Temperature dynamics from t to t+1
        # Occupancy is the current known occupancy, consistently with the offline training
        model.c18 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda m, r, s:
                m.temp_next[r, s] ==
                current_temp[r]
                + zeta_conv * m.p[r]
                + zeta_exch * (current_temp[1 - r] - current_temp[r])
                + zeta_loss * (T_out[t] - current_temp[r])
                + zeta_occ * current_occ[r]
                - zeta_cool * m.v
        )

        # Humidity dynamics from t to t+1
        model.c19 = Constraint(
            model.Scenarios,
            rule=lambda m, s:
                m.humidity_next[s] ==
                current_humidity
                + eta_occ * (current_occ[0] + current_occ[1])
                - eta_vent * m.v
        )

        # Ventilation counter update
        # If v = 1, counter increases by 1; if v = 0, counter resets to 0
        model.c20 = Constraint(
            expr=model.vent_counter_next == current_vent_counter * model.v + model.v
        )

        # Next low-temperature detection
        model.c21 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda m, r, s:
                m.temp_next[r, s] <= T_low + M_temp * (1 - m.y_low_next[r, s])
        )

        model.c22 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda m, r, s:
                m.temp_next[r, s] >= T_low - M_temp * m.y_low_next[r, s]
        )

        # Next overrule activation if temperature is low
        model.c23 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda m, r, s:
                m.overrule_next[r, s] >= m.y_low_next[r, s]
        )

        model.c24 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda m, r, s:
                m.overrule_next[r, s] <= m.overrule[r] + m.y_low_next[r, s]
        )

        # Next OK-temperature detection
        model.c25 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda m, r, s:
                m.temp_next[r, s] >= T_ok - M_temp * (1 - m.y_ok_next[r, s])
        )

        model.c26 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda m, r, s:
                m.temp_next[r, s] <= T_ok + M_temp * m.y_ok_next[r, s]
        )

        # Next overrule deactivation if temperature is OK
        model.c27 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda m, r, s:
                m.overrule_next[r, s] >= m.overrule[r] - m.y_ok_next[r, s]
        )

        model.c28 = Constraint(
            model.R,
            model.Scenarios,
            rule=lambda m, r, s:
                m.overrule_next[r, s] <= 1 - m.y_ok_next[r, s]
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

    # Fallback in case the environment does not provide price_previous
    if "price_previous" not in state:
        state["price_previous"] = state["price_t"]

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