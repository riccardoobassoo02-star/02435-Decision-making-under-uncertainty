from pyomo.environ import *
from sklearn.cluster import KMeans
import numpy as np
import time

from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data


# Parameters extraction from system characteristics
# Renamed to sys_data to avoid shadowing inside generate_samples
sys_data    = get_fixed_data()
T           = sys_data['num_timeslots']
P_max       = sys_data['heating_max_power']
zeta_exch   = sys_data['heat_exchange_coeff']
zeta_conv   = sys_data['heating_efficiency_coeff']
zeta_loss   = sys_data['thermal_loss_coeff']
zeta_cool   = sys_data['heat_vent_coeff']
zeta_occ    = sys_data['heat_occupancy_coeff']
T_low       = sys_data['temp_min_comfort_threshold']
T_ok        = sys_data['temp_OK_threshold']
T_high      = sys_data['temp_max_comfort_threshold']
T_out       = sys_data['outdoor_temperature']
P_vent      = sys_data['ventilation_power']
H_high      = sys_data['humidity_threshold']
eta_occ     = sys_data['humidity_occupancy_coeff']
eta_vent    = sys_data['humidity_vent_coeff']
U_vent      = sys_data['vent_min_up_time']   # 3 hours
L           = sys_data['num_timeslots']       # 10 hours

M_temp = 50   # big-M constant for temperature
M_hum  = 100  # big-M constant for humidity

# Load pre-trained ADP weights: shape (T, 10)
# Weights were trained on NORMALIZED features — normalization must be applied here too
# Feature order: [T1, T2, H, Occ1, Occ2, price_t, price_previous, vent_counter, low_override_r1, low_override_r2]
eta_weights = np.load("eta_weights.npy")


def generate_samples(state, B, N_samples):
    """Sample N_samples next-step scenarios from the stochastic models,
    then reduce to B representative clusters via K-means.
    Returns a list of B cluster dictionaries.
    """
    sample_prices = []
    sample_occ1s  = []
    sample_occ2s  = []

    for _ in range(N_samples):
        p      = price_model(state["price_t"], state["price_previous"])
        o1, o2 = next_occupancy_levels(state["Occ1"], state["Occ2"])
        sample_prices.append(p)
        sample_occ1s.append(o1)
        sample_occ2s.append(o2)

    # Cluster N_samples into B representative centroids
    X         = np.column_stack([sample_prices, sample_occ1s, sample_occ2s])
    km        = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
    labels    = km.labels_
    centroids = km.cluster_centers_   # shape (B, 3)

    clusters = []
    for b in range(B):
        cluster_prob = np.sum(labels == b) / N_samples
        cluster = {                        # renamed from 'data' to avoid shadowing sys_data
            "price":      centroids[b, 0],
            "occ_room_0": centroids[b, 1],
            "occ_room_1": centroids[b, 2],
            "prob":       cluster_prob
        }
        clusters.append(cluster)

    return clusters


def value_function(model, scenario_idx, scenarios, current_state):
    """Compute the approximate value of the next state s_{t+1} for a given scenario,
    using the pre-trained linear VFA: V(s_{t+1}) = eta_{t+1}^T * phi(s_{t+1}).

    IMPORTANT: features must be normalized exactly as done during training in phi().
    Normalization: (x - mean) / range
    """
    hour = current_state["current_time"]   # current time t

    # Pyomo variables (appear in the objective — solver optimizes over these)
    temp_r1_next      = model.temp_next[0, scenario_idx]
    temp_r2_next      = model.temp_next[1, scenario_idx]
    humidity_next     = model.humidity_next[scenario_idx]
    vent_counter_next = model.vent_counter_next
    overrule_r1_next  = model.overrule_next[0, scenario_idx]
    overrule_r2_next  = model.overrule_next[1, scenario_idx]

    # Constants (known from scenario and current state — not optimization variables)
    occ_room_0_next = scenarios[scenario_idx]["occ_room_0"]
    occ_room_1_next = scenarios[scenario_idx]["occ_room_1"]
    price_next      = scenarios[scenario_idx]["price"]
    prev_price_next = current_state["price_t"]   # price at t becomes price_previous at t+1

    # Weights for t+1 (estimating value of the NEXT state)
    w = eta_weights[hour + 1]

    # Normalized VFA — normalization matches training phi() exactly:
    #   T1, T2: (x - 22) / 8
    #   H:      (x - 30) / 70
    #   Occ1:   (x - 20) / 30
    #   Occ2:   (x - 10) / 20
    #   price_t, price_previous: x / 12
    #   vent_counter: x / 3
    #   low_override_r1, low_override_r2: raw (already in {0,1})
    vfa_value = (
        w[0] * 1.0                          +   # Intercept (constant feature)
        w[1] * (temp_r1_next - 22) / 8      +   # T1 normalized
        w[2] * (temp_r2_next - 22) / 8      +   # T2 normalized
        w[3] * (humidity_next - 30) / 70    +   # H normalized
        w[4] * (occ_room_0_next - 20) / 30  +   # Occ1 normalized (constant)
        w[5] * (occ_room_1_next - 10) / 20  +   # Occ2 normalized (constant)
        w[6] * price_next / 12              +   # price_t normalized (constant)
        w[7] * prev_price_next / 12         +   # price_previous normalized (constant)
        w[8] * vent_counter_next / 3        +   # vent_counter normalized
        w[9] * overrule_r1_next             +   # low_override_r1 raw
        w[10] * overrule_r2_next                 # low_override_r2 raw
    )

    return vfa_value


def solve_MILP(state, scenarios):
    """Build and solve the one-step lookahead MILP with VFA for the future cost.
    Returns the here-and-now decisions: p1, p2, v.
    """
    current_temp         = [state["T1"], state["T2"]]
    current_humidity     = state["H"]
    current_price        = state["price_t"]
    current_vent_counter = state["vent_counter"]
    t                    = state["current_time"]

    # Ventilation inertia: must stay ON if counter is 1 or 2
    v_inertia = 1 if current_vent_counter in [1, 2] else 0

    # Whether ventilation was ON in the previous timestep
    v_prev = current_vent_counter > 0

    # Whether overrule controllers were active in the previous timestep
    overrulers_prev = [state["low_override_r1"], state["low_override_r2"]]

    # Create MILP model
    model = ConcreteModel()

    # Sets
    model.R         = RangeSet(0, 1)
    model.Scenarios = RangeSet(0, len(scenarios) - 1)

    # Here-and-now decision variables
    model.p = Var(model.R, domain=NonNegativeReals, bounds=(0, P_max))
    model.v = Var(domain=Binary)

    # Auxiliary binary variables for overrule controller logic at time t
    model.y_high   = Var(model.R, domain=Binary)
    model.y_low    = Var(model.R, domain=Binary)
    model.y_ok     = Var(model.R, domain=Binary)
    model.overrule = Var(model.R, domain=Binary)

    # Ventilation startup variable at time t
    model.s = Var(domain=Binary)

    # Next-state variables — bounded to prevent unboundedness caused by negative VFA weights
    model.temp_next         = Var(model.R, model.Scenarios, domain=Reals,
                                  bounds=(0, 50))    # physical temperature range [0°C, 50°C]
    model.humidity_next     = Var(model.Scenarios, domain=NonNegativeReals,
                                  bounds=(0, 100))   # humidity range [0%, 100%]
    model.vent_counter_next = Var(domain=NonNegativeIntegers,
                                  bounds=(0, L))     # max = min_up_time
    model.overrule_next     = Var(model.R, model.Scenarios, domain=Binary)
    model.y_low_next        = Var(model.R, model.Scenarios, domain=Binary)
    model.y_ok_next         = Var(model.R, model.Scenarios, domain=Binary)

    # Objective: immediate cost + probability-weighted VFA of next state
    # At t=9 (last hour) there is no future, so VFA term is skipped
    model.obj = Objective(
        expr = current_price * (P_vent * model.v + model.p[0] + model.p[1])
             + (sum(scenarios[s]["prob"] * value_function(model, s, scenarios, state)
                    for s in model.Scenarios) if t < 9 else 0),
        sense=minimize
    )

    # 1-2. Detect if temperature exceeds T_high
    model.c1 = Constraint(model.R, rule=lambda model, r: current_temp[r] >= T_high - M_temp * (1 - model.y_high[r]))
    model.c2 = Constraint(model.R, rule=lambda model, r: current_temp[r] <= T_high + M_temp * model.y_high[r])

    # 3. High-temp overrule: force heater to zero
    model.c3 = Constraint(model.R, rule=lambda model, r: model.p[r] <= P_max * (1 - model.y_high[r]))

    # 4-5. Detect if temperature is below T_low
    model.c4 = Constraint(model.R, rule=lambda model, r: current_temp[r] <= T_low + M_temp * (1 - model.y_low[r]))
    model.c5 = Constraint(model.R, rule=lambda model, r: current_temp[r] >= T_low - M_temp * model.y_low[r])

    # 6-7. Detect if temperature is above T_ok
    model.c6 = Constraint(model.R, rule=lambda model, r: current_temp[r] >= T_ok - M_temp * (1 - model.y_ok[r]))
    model.c7 = Constraint(model.R, rule=lambda model, r: current_temp[r] <= T_ok + M_temp * model.y_ok[r])

    # 8-9. Low-temp overrule: activate if temperature is below T_low
    model.c8 = Constraint(model.R, rule=lambda model, r: model.overrule[r] >= model.y_low[r])
    model.c9 = Constraint(model.R, rule=lambda model, r: model.overrule[r] <= int(overrulers_prev[r]) + model.y_low[r])

    # 10. Low-temp overrule: force heater to maximum
    model.c10 = Constraint(model.R, rule=lambda model, r: model.p[r] >= P_max * model.overrule[r])

    # 11-12. Low-temp overrule: deactivate only when temperature exceeds T_ok
    model.c11 = Constraint(model.R, rule=lambda model, r: model.overrule[r] >= int(overrulers_prev[r]) - model.y_ok[r])
    model.c12 = Constraint(model.R, rule=lambda model, r: model.overrule[r] <= 1 - model.y_ok[r])

    # 13-16. Ventilation startup detection and minimum up-time enforcement
    model.c13 = Constraint(rule=(model.s >= model.v) if t == 0 else (model.s >= model.v - v_prev))
    model.c14 = Constraint(rule=model.s <= model.v)
    model.c15 = Constraint(rule=model.s <= 1 - v_prev if t > 0 else Constraint.Skip)
    model.c16 = Constraint(rule=model.v >= v_inertia)

    # 17. Humidity overrule: force ventilation ON if humidity exceeds threshold
    model.c17 = Constraint(rule=current_humidity <= H_high + M_hum * model.v)

    # 18. Temperature dynamics at t+1 for each scenario
    model.c18 = Constraint(
        model.R, model.Scenarios,
        rule=lambda model, r, s: (
            model.temp_next[r, s] == current_temp[r]
            + zeta_exch * (current_temp[1 - r] - current_temp[r])
            - zeta_loss * (current_temp[r] - T_out[t])
            + zeta_conv * model.p[r]
            - zeta_cool * model.v
            + zeta_occ * scenarios[s]["occ_room_" + str(r)]
        ) if t < 9 else Constraint.Skip
    )

    # 19. Humidity dynamics at t+1 for each scenario
    model.c19 = Constraint(
        model.Scenarios,
        rule=lambda model, s: (
            model.humidity_next[s] == current_humidity
            + eta_occ * sum(scenarios[s]["occ_room_" + str(r)] for r in model.R)
            - eta_vent * model.v
        ) if t < 9 else Constraint.Skip
    )

    # 20. Ventilation counter at t+1 (linear: v is binary so v^2 = v)
    model.c20 = Constraint(rule=model.vent_counter_next == model.v * (current_vent_counter + 1))

    # 21-22. Detect if next temperature is below T_low (per scenario)
    model.c21 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.temp_next[r, s] <= T_low + M_temp * (1 - model.y_low_next[r, s]))
    model.c22 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.temp_next[r, s] >= T_low - M_temp * model.y_low_next[r, s])

    # 23-24. Next overrule: activate if next temperature is below T_low
    model.c23 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.overrule_next[r, s] >= model.y_low_next[r, s])
    model.c24 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.overrule_next[r, s] <= model.overrule[r] + model.y_low_next[r, s])

    # 25-26. Next overrule: deactivate if next temperature exceeds T_ok
    model.c25 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.overrule_next[r, s] >= model.overrule[r] - model.y_ok_next[r, s])
    model.c26 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.overrule_next[r, s] <= 1 - model.y_ok_next[r, s])

    # 27-28. Detect if next temperature is above T_ok (per scenario)
    model.c27 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.temp_next[r, s] >= T_ok - M_temp * (1 - model.y_ok_next[r, s]))
    model.c28 = Constraint(model.R, model.Scenarios, rule=lambda model, r, s: model.temp_next[r, s] <= T_ok + M_temp * model.y_ok_next[r, s])

    # Solve
    solver = SolverFactory('gurobi')
    result  = solver.solve(model, options={"OutputFlag": 0})

    if result.solver.termination_condition != TerminationCondition.optimal:
        print("[WARNING] MILP did not solve to optimality — returning zeros")
        return 0.0, 0.0, 0

    p1 = value(model.p[0])
    p2 = value(model.p[1])
    v  = int(value(model.v) > 0.5)

    return p1, p2, v


# ENTRY POINT — called by the environment at each hour
def select_action(state):
    start_time = time.time()

    scenarios = generate_samples(state, B=5, N_samples=1_000)
    p1, p2, v = solve_MILP(state, scenarios)

    # print(f"Policy time: {time.time() - start_time:.2f} s")

    return {
        "HeatPowerRoom1": p1,
        "HeatPowerRoom2": p2,
        "VentilationON":  v
    }