import time

import numpy as np
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Binary, ConstraintList, Objective, SolverFactory, value, \
    Reals, Set, RangeSet, minimize, TerminationCondition
from sklearn.cluster import KMeans

from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.PriceProcessRestaurant import price_model
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

M_temp = 50
M_hum  = 100

# The state will be provided by the environment as the following dictionary
# state = {
#     "T1": ...,              # Temperature of room 1
#     "T2": ...,              # Temperature of room 2
#     "H": ...,               # Humidity
#     "Occ1": ...,            # Occupancy of room 1
#     "Occ2": ...,            # Occupancy of room 2
#     "price_t": ...,         # Price
#     "price_previous": ...,  # Previous Price
#     "vent_counter": ...,    # Consecutive hours ventilation has been ON
#     "low_override_r1": ..., # Is low-temp overrule controller of room 1 active
#     "low_override_r2": ..., # Is low-temp overrule controller of room 2 active
#     "current_time": ...     # Hour of the day (0-9)
# }


# ─────────────────────────────────────────────────────────────────────────────
# VALUE FUNCTION APPROXIMATION
# Hardcoded initial linear VFA for testing. Replace coefficients after training.
#
# Inputs: full estimated state at a leaf node — a dict with keys:
#   T1, T2, H, Occ1, Occ2, price_t
# Output: scalar estimated cost-to-go from that leaf state onward.
#
# Sign intuition:
#   T1, T2 negative  → warmer rooms mean less future heating cost
#   H      positive  → higher humidity means more future ventilation cost
#   price  positive  → higher future price means more future cost
#   Occ    positive  → more people raise humidity, increasing future vent cost
# ─────────────────────────────────────────────────────────────────────────────
def value_function(leaf_state):
    """
    Hardcoded initial linear value function approximation (VFA).
    Approximates the expected future cost-to-go from a given leaf state.
    Replace the coefficients with fitted ones after training.

    Args:
        leaf_state: dict with keys T1, T2, H, Occ1, Occ2, price_t

    Returns:
        scalar cost-to-go estimate
    """
    #These are the $\eta$ that will come from the parametrization of the Value Function

    w_T1    = -0.5   # warmer room 1 → less future heating needed
    w_T2    = -0.5   # warmer room 2 → less future heating needed
    w_H     =  0.3   # higher humidity → more future ventilation needed
    w_Occ1  =  0.1   # more occupancy → more humidity generation
    w_Occ2  =  0.1
    w_price =  1.0   # higher future price → higher future cost
    #bias    =  5.0  #TO BE REVIEWED, MAYBE WE CAN DELETE THIS LINE

    return (
        w_T1   * leaf_state["T1"]      +
        w_T2   * leaf_state["T2"]      +
        w_H    * leaf_state["H"]       +
        w_Occ1 * leaf_state["Occ1"]   +
        w_Occ2 * leaf_state["Occ2"]   +
        w_price* leaf_state["price_t"]
#         + bias
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO TREE BUILDER — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
def build_tree(state, L, B, N_samples=100):
    """
    Builds scenario tree using iterative Branch & Cluster.

    Args:
        state:     current state dictionary from environment
        L:         lookahead horizon (number of future steps)
        B:         branching factor (number of clusters per node)
        N_samples: raw samples generated per node before clustering

    Returns:
        list of node dictionaries representing the full scenario tree
    """
    root = {
        "id":         0,
        "tau":        0,
        "parent_id":  None,
        "price":      state["price_t"],
        "price_prev": state["price_previous"],
        "occ1":       state["Occ1"],
        "occ2":       state["Occ2"],
        "prob":       1.0
    }

    nodes   = [root]
    queue   = [root]
    next_id = 1

    while queue:
        parent = queue.pop(0)

        if parent["tau"] >= L:
            continue

        sample_prices = []
        sample_occ1s  = []
        sample_occ2s  = []

        for _ in range(N_samples):
            p = price_model(parent["price"], parent["price_prev"])
            o1, o2 = next_occupancy_levels(parent["occ1"], parent["occ2"])
            sample_prices.append(p)
            sample_occ1s.append(o1)
            sample_occ2s.append(o2)

        X         = np.column_stack([sample_prices, sample_occ1s, sample_occ2s])
        km        = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
        labels    = km.labels_
        centroids = km.cluster_centers_

        for b in range(B):
            cluster_prob = np.sum(labels == b) / N_samples

            child = {
                "id":         next_id,
                "tau":        parent["tau"] + 1,
                "parent_id":  parent["id"],
                "price":      centroids[b, 0],
                "price_prev": parent["price"],
                "occ1":       centroids[b, 1],
                "occ2":       centroids[b, 2],
                "prob":       parent["prob"] * cluster_prob
            }

            nodes.append(child)
            queue.append(child)
            next_id += 1

    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# ADP MILP SOLVER
# ─────────────────────────────────────────────────────────────────────────────
def ADP_solve(state, nodes, value_function):
    """
    Builds and solves the multi-stage ADP MILP on the scenario tree.
    Identical to the SP formulation except that leaf nodes contribute an
    additional probability-weighted terminal cost V(leaf_state) to the
    objective, approximating the cost-to-go beyond the lookahead horizon.

    V is evaluated on analytically estimated leaf states (not Pyomo variables)
    so the added term is a scalar constant — the MILP remains linear.

    Args:
        state:          current state dictionary from the environment
        nodes:          scenario tree from build_tree()
        value_function: callable V(leaf_state_dict) -> scalar cost-to-go

    Returns:
        p1, p2, v  (here-and-now decisions at tau=0)
    """
    model = ConcreteModel()

    # SETUP
    node_by_id   = {n["id"]: n for n in nodes}
    nodes_future = [n for n in nodes if n["tau"] >= 1]

    # leaf nodes: future nodes that have no children in the tree
    ids_with_children = {n["parent_id"] for n in nodes if n["parent_id"] is not None}
    leaf_nodes        = [n for n in nodes_future if n["id"] not in ids_with_children]

    t_now = state["current_time"]

    low_override_init = {1: state["low_override_r1"],
                         2: state["low_override_r2"]}

    vent_counter     = state["vent_counter"]
    remaining_forced = max(0, min_up_time - vent_counter) if vent_counter > 0 else 0
    v_prev           = 1 if vent_counter > 0 else 0

    # SETS
    model.R     = RangeSet(1, 2)
    model.NODES = Set(initialize=[n["id"] for n in nodes_future])

    # VARIABLES
    model.p0 = Var(model.R, within=NonNegativeReals, bounds=(0, P_max))
    model.v0 = Var(within=Binary)
    model.s0 = Var(within=Binary)
    model.p    = Var(model.R, model.NODES, within=NonNegativeReals, bounds=(0, P_max))
    model.v    = Var(model.NODES, within=Binary)
    model.s    = Var(model.NODES, within=Binary)
    model.temp = Var(model.R, model.NODES, within=Reals)
    model.hum  = Var(model.NODES, within=NonNegativeReals)
    model.y_low  = Var(model.R, model.NODES, within=Binary)
    model.y_ok   = Var(model.R, model.NODES, within=Binary)
    model.u      = Var(model.R, model.NODES, within=Binary)
    model.y_high = Var(model.R, model.NODES, within=Binary)

    # HELPER FUNCTIONS — unchanged
    def v_par(node):
        return model.v0 if node["tau"] == 1 else model.v[node["parent_id"]]

    def p_par(r, node):
        return model.p0[r] if node["tau"] == 1 else model.p[r, node["parent_id"]]

    def temp_par(r, node):
        if node["tau"] == 1:
            return state["T1"] if r == 1 else state["T2"]
        return model.temp[r, node["parent_id"]]

    def temp_other_par(r, node):
        r_other = 3 - r
        if node["tau"] == 1:
            return state["T1"] if r_other == 1 else state["T2"]
        return model.temp[r_other, node["parent_id"]]

    def hum_par(node):
        return state["H"] if node["tau"] == 1 else model.hum[node["parent_id"]]

    def occ_par(r, node):
        if node["tau"] == 1:
            return state["Occ1"] if r == 1 else state["Occ2"]
        parent = node_by_id[node["parent_id"]]
        return parent["occ1"] if r == 1 else parent["occ2"]

    def u_par(r, node):
        if node["tau"] == 1:
            return int(low_override_init[r])
        return model.u[r, node["parent_id"]]


    # ── OBJECTIVE FUNCTION ──────────────────────────────────────────────────
    # Part 1: stage costs over the whole tree (identical to SP)
    obj_expr = state["price_t"] * (
        model.p0[1] + model.p0[2] + P_vent * model.v0
    )

    #Part 2: THIS CODE needs to be reviewed. Here Should go the value function     
    for n in leaf_nodes:
        t_idx = min(t_now, len(T_out) - 1)
        t_out_val = T_out[t_idx]

        # one-step temperature estimate from current state toward this leaf
        t1_est = (
            state["T1"]
            + zeta_exch * (state["T2"] - state["T1"])
            - zeta_loss * (state["T1"] - t_out_val)
            + zeta_occ  * n["occ1"]
        )
        t2_est = (
            state["T2"]
            + zeta_exch * (state["T1"] - state["T2"])
            - zeta_loss * (state["T2"] - t_out_val)
            + zeta_occ  * n["occ2"]
        )
        h_est = (
            state["H"]
            + eta_occ * (n["occ1"] + n["occ2"])
        )

        # build the full estimated leaf state dict (matches value_function signature)
        leaf_state_est = {
            "T1":      t1_est,
            "T2":      t2_est,
            "H":       h_est,
            "Occ1":    n["occ1"],
            "Occ2":    n["occ2"],
            "price_t": n["price"]
        }

        vfa_value = value_function(leaf_state_est)  # scalar
        obj_expr += n["prob"] * vfa_value            # weighted constant term

    model.obj = Objective(expr=obj_expr, sense=minimize)


    # ── CONSTRAINTS — identical to SP ───────────────────────────────────────
    model.c = ConstraintList()

    # HERE-AND-NOW OVERRULE CONSTRAINTS
    for r in [1, 2]:
        temp_now = state["T1"] if r == 1 else state["T2"]
        if low_override_init[r]:
            model.p0[r].fix(P_max)
        if temp_now >= T_high:
            model.p0[r].fix(0)

    if state["H"] > H_high:
        model.v0.fix(1)

    # HERE-AND-NOW VENTILATION CONSTRAINTS
    model.c.add(model.s0 >= model.v0 - v_prev)
    model.c.add(model.s0 <= model.v0)
    model.c.add(model.s0 <= 1 - v_prev)

    if remaining_forced >= 1:
        model.v0.fix(1)
    for n in nodes_future:
        if n["tau"] == 1 and remaining_forced >= 2:
            model.v[n["id"]].fix(1)

    # FUTURE NODE CONSTRAINTS
    for n in nodes_future:
        nid      = n["id"]
        tau      = n["tau"]
        t_parent = t_now + tau - 1
        t_out    = T_out[min(t_parent, len(T_out) - 1)]

        for r in [1, 2]:

            # TEMPERATURE DYNAMICS
            model.c.add(
                model.temp[r, nid] == temp_par(r, n)
                    + zeta_exch * (temp_other_par(r, n) - temp_par(r, n))
                    - zeta_loss * (temp_par(r, n) - t_out)
                    + zeta_conv * p_par(r, n)
                    - zeta_cool * v_par(n)
                    + zeta_occ  * occ_par(r, n)
            )

            # LOW-TEMP OVERRULE CONTROLLER
            model.c.add(model.temp[r, nid] <= T_low + M_temp * (1 - model.y_low[r, nid]))
            model.c.add(model.temp[r, nid] >= T_low - M_temp * model.y_low[r, nid])
            model.c.add(model.temp[r, nid] >= T_ok  - M_temp * (1 - model.y_ok[r, nid]))
            model.c.add(model.temp[r, nid] <= T_ok  + M_temp * model.y_ok[r, nid])
            model.c.add(model.u[r, nid] >= model.y_low[r, nid])
            model.c.add(model.u[r, nid] <= u_par(r, n) + model.y_low[r, nid])
            model.c.add(model.p[r, nid] >= P_max * model.u[r, nid])
            model.c.add(model.u[r, nid] >= u_par(r, n) - model.y_ok[r, nid])
            model.c.add(model.u[r, nid] <= 1 - model.y_ok[r, nid])

            # HIGH-TEMP OVERRULE CONTROLLER
            model.c.add(model.temp[r, nid] >= T_high - M_temp * (1 - model.y_high[r, nid]))
            model.c.add(model.temp[r, nid] <= T_high + M_temp * model.y_high[r, nid])
            model.c.add(model.p[r, nid] <= P_max * (1 - model.y_high[r, nid]))

        # HUMIDITY DYNAMICS
        model.c.add(
            model.hum[nid] ==
                hum_par(n)
                + eta_occ * (occ_par(1, n) + occ_par(2, n))
                - eta_vent * v_par(n)
        )

        # HUMIDITY OVERRULE CONTROLLER
        model.c.add(model.hum[nid] <= H_high + M_hum * model.v[nid])

        # VENTILATION INERTIA
        model.c.add(model.s[nid] >= model.v[nid] - v_par(n))
        model.c.add(model.s[nid] <= model.v[nid])
        model.c.add(model.s[nid] <= 1 - v_par(n))
        ancestor = n
        for depth in range(1, min_up_time):
            if ancestor["parent_id"] is None:
                break
            ancestor = node_by_id[ancestor["parent_id"]]
            if ancestor["tau"] == 0:
                model.c.add(model.v[nid] >= model.s0)
                break
            else:
                model.c.add(model.v[nid] >= model.s[ancestor["id"]])

    # SOLVE
    solver = SolverFactory('gurobi')
    result = solver.solve(model, options={"OutputFlag": 0})

    if result.solver.termination_condition != TerminationCondition.optimal:
        print("[WARNING] ADP did not solve to optimality — returning zeros")
        return 0.0, 0.0, 0

    p1 = value(model.p0[1])
    p2 = value(model.p0[2])
    v  = int(value(model.v0) > 0.5)

    return p1, p2, v


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT (called by the environment)
# ─────────────────────────────────────────────────────────────────────────────
def select_action(state):
    """Selects an action given the current state by building and solving a
    multi-stage ADP MILP on a scenario tree, with a value function
    approximation added as terminal cost at leaf nodes."""

    try:
        start     = time.time()
        L, B      = min(4, 9 - state["current_time"]), 3
        nodes     = build_tree(state, L=L, B=B, N_samples=100)
        p1, p2, v = ADP_solve(state, nodes, value_function)
        end       = time.time()
        print(f"Total policy time: {end - start:.2f} s")
        HereAndNowActions = {
            "HeatPowerRoom1": p1,
            "HeatPowerRoom2": p2,
            "VentilationON":  v
        }
    except Exception as e:
        print(f"[ERROR] ADP policy failed: {e}")
        HereAndNowActions = {
            "HeatPowerRoom1": 0,
            "HeatPowerRoom2": 0,
            "VentilationON":  0
        }
    return HereAndNowActions