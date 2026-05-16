# TASK 6: Two-Stage Stochastic Programming Policy
"""
Two-Stage Stochastic Programming Policy (Task 6)
Adapted from the multi-stage SP policy (Task 3).

Key difference — scenario tree structure:
  - Multi-stage (Task 3): branching at EVERY node (full tree)
  - Two-stage  (Task 6): branching ONLY at the root (fan shape)

E.g. : fan structure with S=5 scenarios and L=3 lookahead:

         ┌─ node 1 ── node 6  ── node 11
         ├─ node 2 ── node 7  ── node 12
root ────┼─ node 3 ── node 8  ── node 13
         ├─ node 4 ── node 9  ── node 14
         └─ node 5 ── node 10 ── node 15

Stage 1 (tau=0): here-and-now decision -> p0[r], v0  (single, shared across ALL scenarios)
Stage 2 (tau=1..L): wait-and-see recourse-> p[r,node], v[node]  (one set per scenario chain)

Non-anticipativity is implicit: there is only one Stage-1 variable, no need for explicit
non-anticipativity constraints (unlike multi-stage trees where intermediate nodes must share
decisions across branches that have not yet diverged).
"""

import time
from pyomo.environ import *
from sklearn.cluster import KMeans
import numpy as np
from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data

# System parameters
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

M_temp = 50   # big-M constant for temperature constraints
M_hum  = 100  # big-M constant for humidity constraints


# FAN TREE BUILDER 
def build_fan_tree(state, L, S, N_samples=100):
    """
    Builds a fan-shaped scenario tree for two-stage SP.

    Branching happens ONLY at the root (tau=0 -> tau=1).
    Each of the S branches then continues as a linear chain up to tau=L.
    This enforces the two-stage structure: one here-and-now decision at tau=0,
    and S independent recourse chains for tau=1..L.

    Args:
        state:     current state dictionary from the environment
        L:         lookahead horizon (number of future steps)
        S:         number of scenarios (fan width, branching factor at root only)
        N_samples: raw samples generated at the root before clustering into S

    Returns:
        list of node dictionaries representing the fan-shaped scenario tree
    """

    # Root node (tau=0) — current state, no uncertainty
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
    next_id = 1

    # STAGE 1: branch root into S scenarios via sampling + KMeans
    sample_prices, sample_occ1s, sample_occ2s = [], [], []
    for _ in range(N_samples):
        p        = price_model(root["price"], root["price_prev"])
        o1, o2   = next_occupancy_levels(root["occ1"], root["occ2"])
        sample_prices.append(p)
        sample_occ1s.append(o1)
        sample_occ2s.append(o2)

    X         = np.column_stack([sample_prices, sample_occ1s, sample_occ2s])
    km        = KMeans(n_clusters=S, random_state=0, n_init=10).fit(X)
    labels    = km.labels_
    centroids = km.cluster_centers_   # shape (S, 3)

    # Create S first-level children (one per scenario) — these are the Stage-2 roots
    scenario_heads = []  # the tau=1 node of each scenario chain
    for s in range(S):
        cluster_prob = np.sum(labels == s) / N_samples  # probability of this scenario

        child = {
            "id":         next_id,
            "tau":        1,
            "parent_id":  0,          # parent is root
            "price":      centroids[s, 0],
            "price_prev": root["price"],
            "occ1":       centroids[s, 1],
            "occ2":       centroids[s, 2],
            "prob":       cluster_prob  # P(scenario s)
        }
        nodes.append(child)
        scenario_heads.append(child)
        next_id += 1

    # STAGE 2: extend each scenario as a LINEAR chain (no more branching)
    # Each chain samples one next step deterministically from the scenario head.
    # We use the centroid values as the starting point and propagate forward.
    for head in scenario_heads:
        current = head
        for tau in range(2, L + 1):
            # Sample one representative next step from the current node
            # (single sample — no clustering needed, chain is linear)
            p_next      = price_model(current["price"], current["price_prev"])
            o1_next, o2_next = next_occupancy_levels(current["occ1"], current["occ2"])

            next_node = {
                "id":         next_id,
                "tau":        tau,
                "parent_id":  current["id"],
                "price":      p_next,
                "price_prev": current["price"],
                "occ1":       o1_next,
                "occ2":       o2_next,
                "prob":       current["prob"]  # same probability as the scenario head (chain rule, single branch)
            }
            nodes.append(next_node)
            current = next_node
            next_id += 1

    return nodes


# SP MILP SOLVER (identical to Task 3) 
def solve_sp(state, nodes):
    """
    Builds and solves the two-stage SP MILP on the fan-shaped scenario tree.
    Returns the here-and-now decisions (p1, p2, v) for tau=0.

    This function is intentionally identical to the multi-stage SP solver:
    the two-stage structure is entirely encoded in the tree topology built
    by build_fan_tree — the MILP formulation does not change.
    """
    model = ConcreteModel()

    # Setup
    node_by_id   = {n["id"]: n for n in nodes}
    nodes_future = [n for n in nodes if n["tau"] >= 1]

    t_now = state["current_time"]

    low_override_init = {1: state["low_override_r1"],
                         2: state["low_override_r2"]}

    vent_counter     = state["vent_counter"]
    remaining_forced = max(0, min_up_time - vent_counter) if vent_counter > 0 else 0
    v_prev           = 1 if vent_counter > 0 else 0

    # Sets
    model.R     = RangeSet(1, 2)
    model.NODES = Set(initialize=[n["id"] for n in nodes_future])

    # Variables — here-and-now (tau=0)
    model.p0 = Var(model.R, within=NonNegativeReals, bounds=(0, P_max))
    model.v0 = Var(within=Binary)
    model.s0 = Var(within=Binary)

    # Variables — future nodes (tau>=1)
    model.p      = Var(model.R, model.NODES, within=NonNegativeReals, bounds=(0, P_max))
    model.v      = Var(model.NODES, within=Binary)
    model.s      = Var(model.NODES, within=Binary)
    model.temp   = Var(model.R, model.NODES, within=Reals)
    model.hum    = Var(model.NODES, within=NonNegativeReals)
    model.y_low  = Var(model.R, model.NODES, within=Binary)
    model.y_ok   = Var(model.R, model.NODES, within=Binary)
    model.u      = Var(model.R, model.NODES, within=Binary)
    model.y_high = Var(model.R, model.NODES, within=Binary)

    # Helper functions — return parent value (variable or known parameter)
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

    # Objective — Stage 1 cost (certain) + expected Stage 2 cost
    obj_expr = state["price_t"] * (
        model.p0[1] + model.p0[2] + P_vent * model.v0
    )
    for n in nodes_future:
        obj_expr += n["prob"] * n["price"] * (
            model.p[1, n["id"]] + model.p[2, n["id"]] + P_vent * model.v[n["id"]]
        )
    model.obj = Objective(expr=obj_expr, sense=minimize)

    # Constraints
    model.c = ConstraintList()

    # Here-and-now overrule constraints (tau=0)
    for r in [1, 2]:
        temp_now = state["T1"] if r == 1 else state["T2"]
        if low_override_init[r]:
            model.p0[r].fix(P_max)   # low-temp override active: force max power
        if temp_now >= T_high:
            model.p0[r].fix(0)       # high-temp override active: force zero power

    if state["H"] > H_high:
        model.v0.fix(1)              # humidity override: force ventilation ON

    # Here-and-now ventilation inertia
    model.c.add(model.s0 >= model.v0 - v_prev)
    model.c.add(model.s0 <= model.v0)
    model.c.add(model.s0 <= 1 - v_prev)

    if remaining_forced >= 1:
        model.v0.fix(1)              # minimum uptime not yet satisfied: force ON

    for n in nodes_future:
        if n["tau"] == 1 and remaining_forced >= 2:
            model.v[n["id"]].fix(1)  # force tau=1 ventilation ON as well

    # Future nodes constraints (tau>=1)
    for n in nodes_future:
        nid      = n["id"]
        tau      = n["tau"]
        t_parent = t_now + tau - 1
        t_out_val = T_out[min(t_parent, len(T_out) - 1)]

        for r in [1, 2]:

            # Temperature dynamics
            model.c.add(
                model.temp[r, nid] ==
                    temp_par(r, n)
                    + zeta_exch * (temp_other_par(r, n) - temp_par(r, n))
                    - zeta_loss * (temp_par(r, n) - t_out_val)
                    + zeta_conv * p_par(r, n)
                    - zeta_cool * v_par(n)
                    + zeta_occ  * occ_par(r, n)
            )

            # Low-temp overrule controller
            model.c.add(model.temp[r, nid] <= T_low + M_temp * (1 - model.y_low[r, nid]))
            model.c.add(model.temp[r, nid] >= T_low - M_temp * model.y_low[r, nid])
            model.c.add(model.temp[r, nid] >= T_ok  - M_temp * (1 - model.y_ok[r, nid]))
            model.c.add(model.temp[r, nid] <= T_ok  + M_temp * model.y_ok[r, nid])
            model.c.add(model.u[r, nid] >= model.y_low[r, nid])
            model.c.add(model.u[r, nid] <= u_par(r, n) + model.y_low[r, nid])
            model.c.add(model.p[r, nid] >= P_max * model.u[r, nid])
            model.c.add(model.u[r, nid] >= u_par(r, n) - model.y_ok[r, nid])
            model.c.add(model.u[r, nid] <= 1 - model.y_ok[r, nid])

            # High-temp overrule controller
            model.c.add(model.temp[r, nid] >= T_high - M_temp * (1 - model.y_high[r, nid]))
            model.c.add(model.temp[r, nid] <= T_high + M_temp * model.y_high[r, nid])
            model.c.add(model.p[r, nid] <= P_max * (1 - model.y_high[r, nid]))

        # Humidity dynamics
        model.c.add(
            model.hum[nid] ==
                hum_par(n)
                + eta_occ  * (occ_par(1, n) + occ_par(2, n))
                - eta_vent * v_par(n)
        )

        # Humidity overrule controller
        model.c.add(model.hum[nid] <= H_high + M_hum * model.v[nid])

        # Ventilation inertia
        model.c.add(model.s[nid] >= model.v[nid] - v_par(n))
        model.c.add(model.s[nid] <= model.v[nid])
        model.c.add(model.s[nid] <= 1 - v_par(n))

        # Minimum uptime: walk up ancestors within min_up_time-1 steps
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

    # Solve
    solver = SolverFactory('gurobi')
    result = solver.solve(model, options={"OutputFlag": 0})

    if result.solver.termination_condition != TerminationCondition.optimal:
        print("[WARNING] Two-stage SP did not solve to optimality — returning zeros")
        return 0.0, 0.0, 0

    p1 = value(model.p0[1])
    p2 = value(model.p0[2])
    v  = int(value(model.v0) > 0.5)

    return p1, p2, v


# ENTRY POINT (called by the environment)
def select_action(state):
    """
    Selects an action given the current state by building and solving a
    two-stage SP MILP on a fan-shaped scenario tree.

    The fan tree branches ONLY at the root into S scenarios, then each
    scenario continues as a linear chain — enforcing the two-stage structure.
    """
    try:
        start = time.time()

        L = min(4, 9 - state["current_time"])  # lookahead horizon
        S = 9                                   # number of fan scenarios (Stage-2 branches)

        nodes    = build_fan_tree(state, L=L, S=S, N_samples=150)
        p1, p2, v = solve_sp(state, nodes)

        end = time.time()
        # print(f"Two-stage SP time: {end - start:.2f} s")

        HereAndNowActions = {
            "HeatPowerRoom1": p1,
            "HeatPowerRoom2": p2,
            "VentilationON":  v
        }

    except Exception as e:
        print(f"[ERROR] Two-stage SP policy failed: {e}")
        HereAndNowActions = {
            "HeatPowerRoom1": 0,
            "HeatPowerRoom2": 0,
            "VentilationON":  0
        }

    return HereAndNowActions