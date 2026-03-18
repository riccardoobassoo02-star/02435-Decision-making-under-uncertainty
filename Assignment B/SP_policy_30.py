# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:14:31 2025

@author: geots
"""

from pyomo.environ import *
from sklearn.cluster import KMeans
import numpy as np
from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.SystemCharacteristics import get_fixed_data

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

M = 1000   # big-M constant for linearization
# Note: initial conditions (T0, H0) are not extracted here because
# they are provided at runtime by the environment via the state dictionary

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


# ------------------------------------------------------------------
# SCENARIO TREE BUILDER (iterative Branch & Cluster)
# ------------------------------------------------------------------
def build_tree(state, H, B, N_samples=100):
    """
    Builds scenario tree using iterative Branch & Cluster.

    Args:
        state:     current state dictionary from environment
        H:         lookahead horizon (number of future steps)
        B:         branching factor (number of clusters per node)
        N_samples: raw samples generated per node before clustering

    Returns:
        list of node dictionaries representing the full scenario tree
    """

    # root node (tau=0) — current state, no uncertainty
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

    nodes = [root]
    queue = [root]   # BFS queue
    next_id = 1

    while queue:
        parent = queue.pop(0)

        if parent["tau"] >= H:
            continue   # leaf node — no children

        # --- BRANCH: generate N_samples random children from this parent ---
        sample_prices = []
        sample_occ1s  = []
        sample_occ2s  = []

        for _ in range(N_samples):
            p      = price_model(parent["price"], parent["price_prev"])
            o1, o2 = next_occupancy_levels(parent["occ1"], parent["occ2"])
            sample_prices.append(p)
            sample_occ1s.append(o1)
            sample_occ2s.append(o2)

        # --- CLUSTER: reduce N_samples to B representative centroids ---
        # feature matrix: each row is one sample [price, occ1, occ2]
        X         = np.column_stack([sample_prices, sample_occ1s, sample_occ2s])
        km        = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
        labels    = km.labels_
        centroids = km.cluster_centers_   # shape (B, 3)

        # --- CREATE B child nodes from centroids ---
        for b in range(B):
            cluster_prob = np.sum(labels == b) / N_samples   # conditional probability

            child = {
                "id":         next_id,
                "tau":        parent["tau"] + 1,
                "parent_id":  parent["id"],
                "price":      centroids[b, 0],          # centroid price
                "price_prev": parent["price"],           # parent price becomes prev
                "occ1":       centroids[b, 1],          # centroid occ1
                "occ2":       centroids[b, 2],          # centroid occ2
                "prob":       parent["prob"] * cluster_prob   # chain rule
            }

            nodes.append(child)
            queue.append(child)
            next_id += 1

    return nodes


# ------------------------------------------------------------------
# SP MILP SOLVER (correct model from Solution to Assignment Part A)
# ------------------------------------------------------------------
def solve_sp(state, nodes):
    """
    Builds and solves the multi-stage SP MILP on the scenario tree.
    Uses the correct model from the Solution to Assignment Part A.
    Returns the here-and-now decisions (p1, p2, v) for tau=0.
    """

    model = ConcreteModel()

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    node_by_id   = {n["id"]: n for n in nodes}
    nodes_future = [n for n in nodes if n["tau"] >= 1]

    t_now = state["current_time"]

    # low override status at tau=0 (known from environment)
    low_override_init = {1: state["low_override_r1"],
                         2: state["low_override_r2"]}

    # ventilation inertia carried over from past decisions
    vent_counter     = state["vent_counter"]
    remaining_forced = max(0, min_up_time - vent_counter) if vent_counter > 0 else 0
    v_prev           = 1 if vent_counter > 0 else 0

    # ------------------------------------------------------------------
    # SETS
    # ------------------------------------------------------------------
    model.R     = RangeSet(1, 2)
    model.NODES = Set(initialize=[n["id"] for n in nodes_future])

    # ------------------------------------------------------------------
    # VARIABLES
    # ------------------------------------------------------------------

    # here-and-now (tau=0) — single decision shared across all scenarios
    model.p0 = Var(model.R, within=NonNegativeReals, bounds=(0, P_max))
    model.v0 = Var(within=Binary)
    model.s0 = Var(within=Binary)   # ventilation startup indicator at tau=0

    # future nodes (tau >= 1) — one variable per node
    model.p          = Var(model.R, model.NODES, within=NonNegativeReals, bounds=(0, P_max))
    model.v          = Var(model.NODES, within=Binary)
    model.s          = Var(model.NODES, within=Binary)   # ventilation startup indicator
    model.temp       = Var(model.R, model.NODES, within=Reals)
    model.hum        = Var(model.NODES, within=NonNegativeReals)

    # low-temp overrule: 3 separate binary variables (solution model eq. 8-16)
    model.y_low = Var(model.R, model.NODES, within=Binary)   # 1 if temp < T_low
    model.y_ok  = Var(model.R, model.NODES, within=Binary)   # 1 if temp > T_ok
    model.u     = Var(model.R, model.NODES, within=Binary)   # 1 if overrule active

    # high-temp overrule and humidity overrule
    model.delta_high = Var(model.R, model.NODES, within=Binary)
    model.delta_hum  = Var(model.NODES, within=Binary)

    # ------------------------------------------------------------------
    # HELPER FUNCTIONS — return parent value (variable or known parameter)
    # ------------------------------------------------------------------
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
            return low_override_init[r]   # known parameter from state
        return model.u[r, node["parent_id"]]

    # ------------------------------------------------------------------
    # OBJECTIVE FUNCTION — minimize expected cost over lookahead horizon
    # ------------------------------------------------------------------
    obj_expr = state["price_t"] * (
        model.p0[1] + model.p0[2] + P_vent * model.v0
    )
    for n in nodes_future:
        obj_expr += n["prob"] * n["price"] * (
            model.p[1, n["id"]] + model.p[2, n["id"]] + P_vent * model.v[n["id"]]
        )
    model.obj = Objective(expr=obj_expr, sense=minimize)

    # ------------------------------------------------------------------
    # CONSTRAINTS
    # ------------------------------------------------------------------
    model.c = ConstraintList()

    for n in nodes_future:
        nid   = n["id"]
        tau   = n["tau"]
        t_abs = t_now + tau - 1
        t_out = T_out[min(t_abs, len(T_out) - 1)]

        for r in [1, 2]:

            # C1 — temperature dynamics (solution eq. 2)
            model.c.add(
                model.temp[r, nid] ==
                    temp_par(r, n)
                    + zeta_exch * (temp_other_par(r, n) - temp_par(r, n))
                    - zeta_loss * (temp_par(r, n) - t_out)
                    + zeta_conv * p_par(r, n)
                    - zeta_cool * v_par(n)
                    + zeta_occ  * occ_par(r, n)
            )

            # LOW-TEMP OVERRULE — solution model eq. 8-16
            # detect temp < T_low (eq. 8-9)
            model.c.add(model.temp[r, nid] <= T_low + M * (1 - model.y_low[r, nid]))
            model.c.add(model.temp[r, nid] >= T_low - M * model.y_low[r, nid])
            # detect temp > T_ok (eq. 10-11)
            model.c.add(model.temp[r, nid] >= T_ok - M * (1 - model.y_ok[r, nid]))
            model.c.add(model.temp[r, nid] <= T_ok + M * model.y_ok[r, nid])
            # activation: temp < T_low → u=1 (eq. 12)
            model.c.add(model.u[r, nid] >= model.y_low[r, nid])
            # memory: u stays ON only if was ON before (eq. 13)
            model.c.add(model.u[r, nid] <= u_par(r, n) + model.y_low[r, nid])
            # force power to max when overrule active (eq. 14)
            model.c.add(model.p[r, nid] >= P_max * model.u[r, nid])
            # deactivation: temp > T_ok → u=0 (eq. 15-16)
            model.c.add(model.u[r, nid] >= u_par(r, n) - model.y_ok[r, nid])
            model.c.add(model.u[r, nid] <= 1 - model.y_ok[r, nid])

            # HIGH-TEMP OVERRULE — solution model eq. 5-7
            # detect temp >= T_high (eq. 5-6)
            model.c.add(model.temp[r, nid] >= T_high - M * (1 - model.delta_high[r, nid]))
            model.c.add(model.temp[r, nid] <= T_high + M * model.delta_high[r, nid])
            # force power to zero (eq. 7)
            model.c.add(model.p[r, nid] <= P_max * (1 - model.delta_high[r, nid]))

        # C2 — humidity dynamics (solution eq. 3)
        model.c.add(
            model.hum[nid] ==
                hum_par(n)
                + eta_occ * (occ_par(1, n) + occ_par(2, n))
                - eta_vent * v_par(n)
        )

        # HUMIDITY OVERRULE — solution eq. 21
        model.c.add(M * model.delta_hum[nid] >= model.hum[nid] - H_high)
        model.c.add(M * (1 - model.delta_hum[nid]) >= H_high - model.hum[nid])
        model.c.add(model.v[nid] >= model.delta_hum[nid])

        # VENTILATION INERTIA — solution eq. 17-20
        # startup detection at this node
        model.c.add(model.s[nid] >= model.v[nid] - v_par(n))
        model.c.add(model.s[nid] <= model.v[nid])
        model.c.add(model.s[nid] <= 1 - v_par(n))
        # minimum uptime: walk up ancestors within min_up_time-1 steps
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

    # ------------------------------------------------------------------
    # HERE-AND-NOW VENTILATION CONSTRAINTS (tau=0)
    # ------------------------------------------------------------------
    model.c.add(model.s0 >= model.v0 - v_prev)
    model.c.add(model.s0 <= model.v0)
    model.c.add(model.s0 <= 1 - v_prev)

    if remaining_forced >= 1:
        model.v0.fix(1)
    for n in nodes_future:
        if n["tau"] == 1 and remaining_forced >= 2:
            model.v[n["id"]].fix(1)

    # ------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------
    solver = SolverFactory('gurobi')
    result = solver.solve(model)

    if result.solver.termination_condition != TerminationCondition.optimal:
        print("[WARNING] SP did not solve to optimality — returning zeros")
        return 0.0, 0.0, 0

    p1 = value(model.p0[1])
    p2 = value(model.p0[2])
    v  = int(value(model.v0) > 0.5)

    return p1, p2, v


# ------------------------------------------------------------------
# ENTRY POINT (called by the environment)
# ------------------------------------------------------------------
def select_action(state):
    try:
        H, B = 3, 3
        nodes = build_tree(state, H=H, B=B, N_samples=100)
        p1, p2, v = solve_sp(state, nodes)
        HereAndNowActions = {
            "HeatPowerRoom1": p1,
            "HeatPowerRoom2": p2,
            "VentilationON":  v
        }
    except Exception as e:
        print(f"[ERROR] SP policy failed: {e}")
        HereAndNowActions = {
            "HeatPowerRoom1": 0,
            "HeatPowerRoom2": 0,
            "VentilationON":  0
        }
    return HereAndNowActions