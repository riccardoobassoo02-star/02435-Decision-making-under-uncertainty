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

M_temp = 50  # big-M constant for temperature ()
M_hum = 150   # big-M constant for humidity
epsilon = 0.1 # small constant for strict inequalities for overrule controllers 

# Note: initial conditions (T0, H0) are not extracted here because they are provided at runtime by the environment via the state dictionary 

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

def build_tree(state, H, B, N_samples = 100):
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

    # root node (tau=0) - current state (no uncertainty)
    root = {
        "id":         0,
        "tau":        0,
        "parent_id":  None,
        "price":      state["price_t"],
        "price_prev": state["price_previous"],
        "occ1":       state["Occ1"],
        "occ2":       state["Occ2"],
        "prob":       1.0 # certain
    }

    nodes = [root]   # list of dictionaries, each representing a node in the scenario tree with its features and probability
    queue = [root]   # BFS queue (breadth-first search)
    next_id = 1      # initialization of the next_id counter

    while queue: # branching until the queue is empty (all nodes have been processed)
        parent = queue.pop(0) # takes the first element of the queue list (parent node) and removes it (so the queue is updated to contain only the children)

        if parent["tau"] >= H: # when we go beyond the lookahead horizon, we don't create children. When H becomes 0 (hour 10), we don't create any children
            continue   # leaf node - no children

        # BRANCHING: generate N_samples random children from this parent
        # define the features of the child nodes: price, occ1, occ2
        sample_prices = []
        sample_occ1s  = []
        sample_occ2s  = []

        for _ in range(N_samples):
            p = price_model(parent["price"], parent["price_prev"]) # price model needs the current price and the previous to generate a hypothetical price for the child node
            o1, o2 = next_occupancy_levels(parent["occ1"], parent["occ2"]) # occupancy model needs the current values of room occupancies
            sample_prices.append(p)
            sample_occ1s.append(o1)
            sample_occ2s.append(o2)

        # CLUSTERING: reduce N_samples to B representative centroids
        X         = np.column_stack([sample_prices, sample_occ1s, sample_occ2s]) # feature matrix with all the N samples rows
        km        = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
        labels    = km.labels_
        centroids = km.cluster_centers_   # reduced matrix of shape (B, 3)

        # CREATING B child nodes from centroids
        for b in range(B):
            cluster_prob = np.sum(labels == b) / N_samples     # conditional probability

            child = {
                "id":         next_id,                         # unique ID for the child node
                "tau":        parent["tau"] + 1,               # deep level in the tree
                "parent_id":  parent["id"],                    # ID of the parent node
                "price":      centroids[b, 0],                 # centroid price
                "price_prev": parent["price"],                 # parent price becomes prev
                "occ1":       centroids[b, 1],                 # centroid occ1
                "occ2":       centroids[b, 2],                 # centroid occ2
                "prob":       parent["prob"] * cluster_prob    # probability of the child according to the chain rule
            }

            nodes.append(child) # update the full list of nodes with the new child
            queue.append(child) # updated the queue list with the new child, which will become the next parent for the next iteration of the while loop
            next_id += 1

    return nodes

# ------------------------------------------------------------------
# SP MILP SOLVER
# ------------------------------------------------------------------ 

def solve_sp(state, nodes): # 2 dictionaries as inputs
    """
    Builds and solves the multi-stage SP MILP on the scenario tree.
    Returns the here-and-now decisions (p1, p2, v) for tau=0.
    """
    model = ConcreteModel()

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    node_by_id   = {n["id"]: n for n in nodes}                  # dictionary with node IDs as keys for easy node lookup
    nodes_future = [n for n in nodes if n["tau"] >= 1]          # list of only future nodes (tau>=1) lookup

    t_now = state["current_time"] # current hour (0-9)

    # low temperature overrule controller status at tau = 0 (known from environment)
    low_override_init = {1: state["low_override_r1"],           # room 1
                         2: state["low_override_r2"]}           # room 2

    # ventilation inertia carried over from past decisions
    vent_counter     = state["vent_counter"]                    # how many consecutive hours the ventilation has been ON until now
    remaining_forced = max(0, min_up_time - vent_counter) if vent_counter > 0 else 0 # how many more hours the ventilation must be forced ON to satisfy the minimum uptime constraint
    v_prev           = 1 if vent_counter > 0 else 0             # 1 if the ventilation was ON in the previous hour

    # ------------------------------------------------------------------
    # SETS
    # ------------------------------------------------------------------
    model.R     = RangeSet(1, 2) # 2 rooms. Automatically creates the set {1, 2} since indexes are numbers
    model.NODES = Set(initialize=[n["id"] for n in nodes_future]) # pyomo set of node IDs for future nodes (tau>=1). Using set and initialize since the indexes are not numerical

    # ------------------------------------------------------------------
    # VARIABLES
    # ------------------------------------------------------------------

    # here-and-now (tau=0) 
    model.p0 = Var(model.R, within=NonNegativeReals, bounds=(0, P_max)) # heating power per room
    model.v0 = Var(within=Binary) # ventilation ON/OFF
    model.s0 = Var(within=Binary)   # ventilation startup indicator at tau = 0 ( 1 if the ventilation is turned ON at tau=0, 0 otherwise)

    # future nodes (tau>=1)
    model.p          = Var(model.R, model.NODES, within=NonNegativeReals, bounds=(0, P_max)) # heating power per room
    model.v          = Var(model.NODES, within=Binary)   # ventilation ON/OFF
    model.s          = Var(model.NODES, within=Binary)   # ventilation startup indicator
    model.temp       = Var(model.R, model.NODES, within=Reals) # temperature per room
    model.hum        = Var(model.NODES, within=NonNegativeReals) # humidity

    # low-temp overrule: 3 separate binary variables (solution model eq. 8-16)
    model.y_low = Var(model.R, model.NODES, within=Binary)   # 1 if temp < T_low
    model.y_ok  = Var(model.R, model.NODES, within=Binary)   # 1 if temp > T_ok
    model.u     = Var(model.R, model.NODES, within=Binary)   # 1 if overrule active

    # high-temp overrule and humidity overrule
    model.y_high = Var(model.R, model.NODES, within=Binary)  # 1 if the temperature of the room exceeds the high threshold

    # ------------------------------------------------------------------
    # HELPER FUNCTIONS — return parent value (variable or known parameter)
    # ------------------------------------------------------------------
    def v_par(node): # ventilation of the parent
        return model.v0 if node["tau"] == 1 else model.v[node["parent_id"]]

    def p_par(r, node): # heating of the power of room r
        return model.p0[r] if node["tau"] == 1 else model.p[r, node["parent_id"]]

    def temp_par(r, node): # temperature of room r at the parent node
        if node["tau"] == 1:
            return state["T1"] if r == 1 else state["T2"]
        return model.temp[r, node["parent_id"]]

    def temp_other_par(r, node): # temperature of the other room at the parent node
        r_other = 3 - r
        if node["tau"] == 1:
            return state["T1"] if r_other == 1 else state["T2"]
        return model.temp[r_other, node["parent_id"]]

    def hum_par(node): # humidity at the parent node
        return state["H"] if node["tau"] == 1 else model.hum[node["parent_id"]]

    def occ_par(r, node): # occupancy of room r at the parent node
        if node["tau"] == 1:
            return state["Occ1"] if r == 1 else state["Occ2"]
        parent = node_by_id[node["parent_id"]]
        return parent["occ1"] if r == 1 else parent["occ2"]

    def u_par(r, node): # status of low-temp overrule controller of room r at the parent node
        if node["tau"] == 1:
            return int(low_override_init[r])   # int converts boolean (true or false)to 0/1 (in the environment is defined as true/false)
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
    # HERE AND NOW CONSTRAINTS (tau=0) 

    # HERE-AND-NOW OVERRULE CONSTRAINTS
    for r in [1, 2]:
        if low_override_init[r]:
            model.p0[r].fix(P_max) # fix the heating power to max if the low-temp overrule controller is already active for that room at the current time step, to satisfy the low-temp overrule constraint (eq. 14)
        temp_now = state["T1"] if r == 1 else state["T2"]
        if temp_now >= T_high:
            model.p0[r].fix(0) # fix the heating power to zero if the temperature is already above the high threshold at the current time step, to satisfy the high-temp overrule constraint (eq. 7)

    if state["H"] > H_high:
        model.v0.fix(1) # fix the ventilation to ON if the humidity is already above the threshold at the current time step, to satisfy the humidity overrule constraint (eq. 21)
    
    # HERE-AND-NOW VENTILATION CONSTRAINTS
    # startup detection at tau=0
    model.c.add(model.s0 >= model.v0 - v_prev)
    model.c.add(model.s0 <= model.v0)
    model.c.add(model.s0 <= 1 - v_prev)

    # if minimum uptime is not yet satisfied by past decisions, force ventilation ON
    if remaining_forced >= 1:
        model.v0.fix(1)
    for n in nodes_future:
        if n["tau"] == 1 and remaining_forced >= 2:
            model.v[n["id"]].fix(1)


    # FUTURE NODES CONSTRAINTS
    for n in nodes_future:
        nid   = n["id"]
        tau   = n["tau"]
        t_parent = t_now + tau - 1 # hour of the parent node
        t_out = T_out[min(t_parent, len(T_out) - 1)] # external temperature a the parent node. the min is used as a safety measure, but USELESS

        for r in [1, 2]:

            # TEMPERATURE DYNAMICS (eq. 2)
            model.c.add(
                model.temp[r, nid] ==
                    temp_par(r, n) # temperature value in the present node (parent)
                    + zeta_exch * (temp_other_par(r, n) - temp_par(r, n)) # heat exchange with the other room
                    - zeta_loss * (temp_par(r, n) - t_out) # thermal loss to the outside
                    + zeta_conv * p_par(r, n) # heating power contribution (of the previous node/hour)
                    - zeta_cool * v_par(n) # cooling effect of the ventilation
                    + zeta_occ  * occ_par(r, n) # heating effect of the occupancy (more people generate more heat)
            )

            # LOW-TEMP OVERRULE CONTROLLER (eq. 8-16)
            # detect temp < T_low (eq. 8-9)---> SHOULD WE IMPLEMENT EPSILON?
            model.c.add(model.temp[r, nid] <= T_low + M_temp * (1 - model.y_low[r, nid]))
            model.c.add(model.temp[r, nid] >= T_low - M_temp * model.y_low[r, nid])
            # detect temp > T_ok (eq. 10-11)---> SHOULD WE IMPLEMENT EPSILON?
            model.c.add(model.temp[r, nid] >= T_ok - M_temp * (1 - model.y_ok[r, nid]))
            model.c.add(model.temp[r, nid] <= T_ok + M_temp  * model.y_ok[r, nid])
            # activation: temp < T_low → u=1 (eq. 12)
            model.c.add(model.u[r, nid] >= model.y_low[r, nid])
            # memory: u stays ON only if was ON before (eq. 13)
            model.c.add(model.u[r, nid] <= u_par(r, n) + model.y_low[r, nid])
            # force power to max when overrule active (eq. 14)
            model.c.add(model.p[r, nid] >= P_max * model.u[r, nid])
            # deactivation: temp > T_ok → u=0 (eq. 15-16)
            model.c.add(model.u[r, nid] >= u_par(r, n) - model.y_ok[r, nid])
            model.c.add(model.u[r, nid] <= 1 - model.y_ok[r, nid])

            # HIGH-TEMP OVERRULE CONTROLLER (eq. 5-7)
            # detect temp >= T_high (eq. 5-6)
            model.c.add(model.temp[r, nid] >= T_high - M_temp * (1 - model.y_high[r, nid]))
            model.c.add(model.temp[r, nid] <= T_high + M_temp * model.y_high[r, nid])
            # force power to zero (eq. 7)
            model.c.add(model.p[r, nid] <= P_max * (1 - model.y_high[r, nid]))

        # HUMIDITY DYNAMICS (solution eq. 3)
        model.c.add(
            model.hum[nid] ==
                hum_par(n) # humidity value in the present node (parent)
                + eta_occ * (occ_par(1, n) + occ_par(2, n)) # humidity increase due to occupancy
                - eta_vent * v_par(n) # humidity decrease due to ventilation
        )

        # HUMIDITY OVERRULE CONTROLLER(eq. 21)
        model.c.add(model.hum[nid] <= H_high + M_hum * model.v[nid])

        # VENTILATION INERTIA (eq. 17-20)
        # startup detection at this node
        model.c.add(model.s[nid] >= model.v[nid] - v_par(n))
        model.c.add(model.s[nid] <= model.v[nid])
        model.c.add(model.s[nid] <= 1 - v_par(n))
        # minimum uptime: walk up ancestors within min_up_time-1 steps
        ancestor = n # starting point is the current node
        for depth in range(1, min_up_time):
            if ancestor["parent_id"] is None: # if the current node hasn't any parent, that means it is the root node. it doesn't have any ancestor
                break
            ancestor = node_by_id[ancestor["parent_id"]]
            if ancestor["tau"] == 0: # root node is reached, use the here-and-now variable v0 as ancestor value
                model.c.add(model.v[nid] >= model.s0)
                break
            else:
                model.c.add(model.v[nid] >= model.s[ancestor["id"]]) # if startup, then s = 1  and forces v to be 1

    
    # ------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------
    solver = SolverFactory('gurobi')
    result = solver.solve(model)

    if result.solver.termination_condition != TerminationCondition.optimal:
        print("[WARNING] SP did not solve to optimality — returning zeros")
        return 0.0, 0.0, 0

    p1 = value(model.p0[1])           # heating power of room 1 at tau=0
    p2 = value(model.p0[2])           # heating power of room 2 at tau=0
    v  = int(value(model.v0) > 0.5)   # ventilation ON/OFF at tau=0 (binary variable, thresholded at 0.5 for the solver tollerance)

    return p1, p2, v


# ------------------------------------------------------------------
# ENTRY POINT (called by the environment)
# ------------------------------------------------------------------
def select_action(state):
    try:
        H, B = min(3, 9-state["current_time"]), 2
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