from pyomo.environ import *
from sklearn.cluster import KMeans
import numpy as np
from Utils.PriceProcessRestaurant import price_model
from Utils.OccupancyProcessRestaurant import next_occupancy_levels
from Utils.v2_SystemCharacteristics import get_fixed_data

# Parameter extraction from system characteristics
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

M_temp = 50                    # big-M for temperature constraints
M_hum  = 100                   # big-M for humidity constraints
M_vc   = min_up_time + 1       # upper bound on vent_counter (resets to 0 when v=0)

# Load offline-trained ADP value function weights, shape (T, 11)
eta_weights = np.load("eta_weights.npy")


# SCENARIO TREE BUILDER (iterative Branch & Cluster)
def build_tree(state, L, B, N_samples=100):
    """
    Builds the multi-stage scenario tree via iterative Branch & Cluster.
    At each parent node, N_samples raw children are sampled from the exogenous
    process, then reduced to B representative scenarios via KMeans clustering.

    Args:
        state:     current state dictionary from the environment
        L:         lookahead horizon (number of future steps)
        B:         branching factor (number of clusters per node)
        N_samples: raw samples generated per node before clustering

    Returns:
        list of node dictionaries representing the full scenario tree
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
    queue   = [root]
    next_id = 1

    while queue:
        parent = queue.pop(0)

        # Stop branching at the lookahead horizon — these are leaf nodes
        if parent["tau"] >= L:
            continue

        # BRANCHING: generate N_samples raw children from this parent
        sample_prices = []
        sample_occ1s  = []
        sample_occ2s  = []
        for _ in range(N_samples):
            p      = price_model(parent["price"], parent["price_prev"])
            o1, o2 = next_occupancy_levels(parent["occ1"], parent["occ2"])
            sample_prices.append(p)
            sample_occ1s.append(o1)
            sample_occ2s.append(o2)

        # CLUSTERING: reduce N_samples to B representative centroids
        X         = np.column_stack([sample_prices, sample_occ1s, sample_occ2s])
        km        = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
        labels    = km.labels_
        centroids = km.cluster_centers_

        # Create B child nodes from centroids
        for b in range(B):
            cluster_prob = np.sum(labels == b) / N_samples

            child = {
                "id":         next_id,
                "tau":        parent["tau"] + 1,
                "parent_id":  parent["id"],
                "price":      float(centroids[b, 0]),
                "price_prev": parent["price"],
                "occ1":       float(centroids[b, 1]),
                "occ2":       float(centroids[b, 2]),
                "prob":       parent["prob"] * cluster_prob  # chain rule
            }

            nodes.append(child)
            queue.append(child)
            next_id += 1

    return nodes


# TERMINAL VFA: phi(s_leaf)^T eta_{t+L}
def terminal_vfa(model, n, node_by_id, t_now, L_horizon):
    """
    Returns the ADP terminal cost phi(s_leaf)^T eta_{t+L} as a Pyomo expression,
    appended to the SP objective at every leaf node (tau=L).

    model.vc[nid] is used directly — supports arbitrary L without special-casing,
    because vc is propagated explicitly through the tree via McCormick linearization.

    price_previous at the leaf is the price of the leaf's parent node (known from
    the tree topology, not a decision variable).
    """
    t_leaf = min(t_now + L_horizon, T - 1)
    w      = eta_weights[t_leaf]
    nid    = n["id"]

    parent          = node_by_id[n["parent_id"]]
    price_prev_leaf = parent["price"]

    return (
          w[0]  * 1.0
        + w[1]  * (model.temp[1, nid] - 22) / 8
        + w[2]  * (model.temp[2, nid] - 22) / 8
        + w[3]  * (model.hum[nid]     - 30) / 70
        + w[4]  * (n["occ1"]          - 20) / 30
        + w[5]  * (n["occ2"]          - 10) / 20
        + w[6]  *  n["price"]              / 12
        + w[7]  *  price_prev_leaf         / 12
        + w[8]  *  model.vc[nid]           / 3
        + w[9]  *  model.u[1, nid]
        + w[10] *  model.u[2, nid]
    )


# HYBRID MILP SOLVER: multi-stage SP over [tau=0, tau=L-1] + VFA terminal cost at tau=L
def solve_hybrid(state, nodes):
    """
    Builds and solves the hybrid MILP:
        min  E[ sum_{tau=0}^{L-1} c(u_tau, x_tau) ]   +   E[ V_hat(s_L) ]
             \________________________/                  \____________/
                  multi-stage SP                          ADP terminal cost

    Multi-stage SP propagates the system dynamics over L steps on the scenario tree.
    At each leaf node (tau=L), the offline-trained value function V_hat(s) = phi(s)^T eta
    approximates the remaining cost from t+L to T.

    Returns the here-and-now decisions (p1, p2, v) for tau=0, or zeros if the solver fails.
    """
    model = ConcreteModel()

    # Tree topology
    node_by_id    = {n["id"]: n for n in nodes}
    nodes_future  = [n for n in nodes if n["tau"] >= 1]
    L_horizon     = max((n["tau"] for n in nodes), default=0)
    # When L_horizon=0 (last timeslot), there are no future nodes — no VFA applied
    leaf_nodes    = [n for n in nodes if n["tau"] == L_horizon and n["tau"] > 0]
    tau_ge2_nodes = [n for n in nodes_future if n["tau"] >= 2]

    # State at tau=0
    t_now             = state["current_time"]
    low_override_init = {1: state["low_override_r1"], 2: state["low_override_r2"]}
    vent_counter     = state["vent_counter"]
    remaining_forced = max(0, min_up_time - vent_counter) if vent_counter > 0 else 0
    v_prev           = 1 if vent_counter > 0 else 0

    # SETS
    model.R     = RangeSet(1, 2)
    model.NODES = Set(initialize=[n["id"] for n in nodes_future])

    # HERE-AND-NOW VARIABLES (tau=0)
    model.p0 = Var(model.R, within=NonNegativeReals, bounds=(0, P_max))
    model.v0 = Var(within=Binary)
    model.s0 = Var(within=Binary)

    # FUTURE NODE VARIABLES (tau>=1)
    model.p      = Var(model.R, model.NODES, within=NonNegativeReals, bounds=(0, P_max))
    model.v      = Var(model.NODES, within=Binary)
    model.s      = Var(model.NODES, within=Binary)
    model.temp   = Var(model.R, model.NODES, within=Reals)
    model.hum    = Var(model.NODES, within=NonNegativeReals)
    model.y_low  = Var(model.R, model.NODES, within=Binary)
    model.y_ok   = Var(model.R, model.NODES, within=Binary)
    model.u      = Var(model.R, model.NODES, within=Binary)
    model.y_high = Var(model.R, model.NODES, within=Binary)

    # VENT COUNTER as explicit MILP variable at every future node
    # Propagation: vc[nid] = (vc_parent + 1) * v[nid]
    #   tau=1  : vc_parent = vent_counter (known scalar) → linear expression
    #   tau>=2 : vc_parent = model.vc[parent_id] (variable) → McCormick linearization
    model.vc = Var(model.NODES, within=NonNegativeReals, bounds=(0, M_vc))

    # AUXILIARY: vc_prod[nid] = vc[parent_id] * v[nid] for tau>=2
    # Linearization of continuous × binary product (McCormick envelopes)
    if tau_ge2_nodes:
        model.VC_GE2  = Set(initialize=[n["id"] for n in tau_ge2_nodes])
        model.vc_prod = Var(model.VC_GE2, within=NonNegativeReals, bounds=(0, M_vc))

    # HELPER FUNCTIONS — return parent value (variable or known parameter)
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

    # OBJECTIVE: expected SP cost over [tau=0, tau=L-1] + expected VFA at tau=L
    # Leaf nodes are excluded from the SP sum — V_hat(s_leaf) already covers the
    # cost from t+L onwards (Bellman convention: V(s_t) = c(s_t,u_t) + E[V(s_{t+1})])
    obj_expr = state["price_t"] * (model.p0[1] + model.p0[2] + P_vent * model.v0)
    for n in nodes_future:
        if n["tau"] < L_horizon:
            obj_expr += n["prob"] * n["price"] * (
                model.p[1, n["id"]] + model.p[2, n["id"]] + P_vent * model.v[n["id"]]
            )
    for n in leaf_nodes:
        obj_expr += n["prob"] * terminal_vfa(model, n, node_by_id, t_now, L_horizon)

    model.obj = Objective(expr=obj_expr, sense=minimize)

    # CONSTRAINTS
    model.c = ConstraintList()

    # Here-and-now overrule fixings
    for r in [1, 2]:
        temp_now = state["T1"] if r == 1 else state["T2"]
        if low_override_init[r]:
            model.p0[r].fix(P_max)
        if temp_now >= T_high:
            model.p0[r].fix(0)
    if state["H"] > H_high:
        model.v0.fix(1)

    # Here-and-now ventilation startup detection
    model.c.add(model.s0 >= model.v0 - v_prev)
    model.c.add(model.s0 <= model.v0)
    model.c.add(model.s0 <= 1 - v_prev)

    # Minimum uptime carryover from past decisions
    if remaining_forced >= 1:
        model.v0.fix(1)
    for n in nodes_future:
        if n["tau"] == 1 and remaining_forced >= 2:
            model.v[n["id"]].fix(1)

    # FUTURE NODE CONSTRAINTS
    for n in nodes_future:
        nid      = n["id"]
        t_parent = t_now + n["tau"] - 1
        t_out    = T_out[min(t_parent, len(T_out) - 1)]

        # Vent counter transition: vc[nid] = (vc_parent + 1) * v[nid]
        if n["tau"] == 1:
            # vc_parent = vent_counter (known scalar) → linear
            model.c.add(model.vc[nid] == (vent_counter + 1) * model.v0)
        else:
            # vc_parent = model.vc[parent_id] (variable)
            # McCormick: vc_prod[nid] = vc[parent_id] * v[nid]
            #   vc_prod >= 0
            #   vc_prod <= vc_parent
            #   vc_prod <= M_vc * v
            #   vc_prod >= vc_parent - M_vc * (1 - v)
            # Then vc[nid] = vc_prod[nid] + v[nid]
            vc_par = model.vc[n["parent_id"]]
            v_cur  = model.v[nid]
            model.c.add(model.vc_prod[nid] >= 0)
            model.c.add(model.vc_prod[nid] <= vc_par)
            model.c.add(model.vc_prod[nid] <= M_vc * v_cur)
            model.c.add(model.vc_prod[nid] >= vc_par - M_vc * (1 - v_cur))
            model.c.add(model.vc[nid] == model.vc_prod[nid] + v_cur)

        for r in [1, 2]:
            # Temperature dynamics
            model.c.add(
                model.temp[r, nid] ==
                    temp_par(r, n)
                    + zeta_exch * (temp_other_par(r, n) - temp_par(r, n))
                    - zeta_loss * (temp_par(r, n) - t_out)
                    + zeta_conv * p_par(r, n)
                    - zeta_cool * v_par(n)
                    + zeta_occ  * occ_par(r, n)
            )

            # Low-temp overrule controller (detect, activate, deactivate)
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
                + eta_occ * (occ_par(1, n) + occ_par(2, n))
                - eta_vent * v_par(n)
        )

        # Humidity-triggered ventilation
        model.c.add(model.hum[nid] <= H_high + M_hum * model.v[nid])

        # Ventilation startup detection at this node
        model.c.add(model.s[nid] >= model.v[nid] - v_par(n))
        model.c.add(model.s[nid] <= model.v[nid])
        model.c.add(model.s[nid] <= 1 - v_par(n))

        # Minimum uptime: walk up ancestors within min_up_time-1 steps
        ancestor = n
        for _ in range(1, min_up_time):
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
    result = solver.solve(model, options={
        "OutputFlag": 0,
        "TimeLimit":  10.0,
        "MIPGap":     0.01,
    })

    tc = result.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.maxTimeLimit):
        print(f"[WARNING] Hybrid did not solve (tc={tc}) — returning zeros")
        return 0.0, 0.0, 0

    p1 = value(model.p0[1])
    p2 = value(model.p0[2])
    v  = int(value(model.v0) > 0.5)

    return p1, p2, v


def calculate_number_of_active_overrides(state):
    count = 0
    if state["low_override_r1"]:
        count += 1
    if state["low_override_r2"]:
        count += 1
    if 0 < state["vent_counter"] < min_up_time:
        count += 1
    return count


# ENTRY POINT (called by the environment)
def select_action(state):
    """
    Entry point called by the environment at each timestep.
    Builds the scenario tree and solves the hybrid SP+ADP MILP to obtain
    the here-and-now actions (p1, p2, v) for tau=0.
    """
    try:
        if "price_previous" not in state:
            state = state.copy()
            state["price_previous"] = state["price_t"]

        # L=4 captures the full ventilation inertia window (min_up_time=3) and one
        # step beyond, giving Gurobi visibility of the full commitment cost.
        # Near the end of the day the lookahead is shortened to avoid empty trees.
        # B=3 keeps the tree manageable at L=4 (3+9+27+81 = 120 future nodes).
        L = min(4, T - 1 - state["current_time"])
        B = 3

        nodes = build_tree(state, L=L, B=B, N_samples=100)

        p1, p2, v = solve_hybrid(state, nodes)

        return {
            "HeatPowerRoom1": p1,
            "HeatPowerRoom2": p2,
            "VentilationON":  v
        }

    except Exception as e:
        print(f"[ERROR] Hybrid policy failed: {e}")
        return {
            "HeatPowerRoom1": 0.0,
            "HeatPowerRoom2": 0.0,
            "VentilationON":  0
        }