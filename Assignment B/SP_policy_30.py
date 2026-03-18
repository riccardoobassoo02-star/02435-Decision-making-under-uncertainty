# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:14:31 2025

@author: geots
""" 

from pyomo.environ import * 
from sklearn.cluster import KMeans
import numpy as np 
from Utils.PriceProcessRestaurant import price_model 
from OccupancyProcessRestaurant import next_occupancy_levels
from SystemCharacteristics import get_fixed_data

# parameters extraction from system characteristics
data = get_fixed_data()
T           = data['num_timeslots'] # number of hours (10)
P_max       = data['heating_max_power'] # maximum heating power (kW)
zeta_exch   = data['heat_exchange_coeff'] # heat exchange coefficient between rooms
zeta_conv   = data['heating_efficiency_coeff'] # heating efficiency: increase in room temperature per kW of heating power
zeta_loss   = data['thermal_loss_coeff'] # thermal loss coefficient: fraction of indoor-outdoor temperature difference lost per hour    
zeta_cool   = data['heat_vent_coeff'] # ventilation cooling effect: temperature decrease in the room for each hour that ventilation is ON (°C)
zeta_occ    = data['heat_occupancy_coeff'] # occupancy heat gain: temperature increase per hour per person in the room (°C)
T_low       = data['temp_min_comfort_threshold'] # minimum comfortable temperature threshold (°C)
T_ok        = data['temp_OK_threshold'] # comfortable temperature threshold (°C)
T_high      = data['temp_max_comfort_threshold'] # maximum comfortable temperature threshold (°C)
T_out       = data['outdoor_temperature'] # outdoor temperature (°C)
P_vent      = data['ventilation_power'] # power consumption of ventilation system when ON (kW)
H_high      = data['humidity_threshold'] # maximum comfortable humidity threshold (%)
eta_occ     = data['humidity_occupancy_coeff'] # humidity increase per hour per person in the room (%)
eta_vent    = data['humidity_vent_coeff'] # humidity decrease per hour when ventilation is ON (%)
min_up_time = data['vent_min_up_time'] # minimum number of consecutive hours that ventilation must be ON once turned ON (hours)

M = 1000 # big M constant for linearization of logical conditions in the overrule controllers (should be sufficiently large to not cut off any feasible solution, but not too large to avoid numerical issues)
epsilon = 0.01 # small value to express the strict inequality for the controllers rules



# The state will be provided by the environment as the following dictionary

# state = {
#     "T1": ..., #Temperature of room 1
#     "T2": ..., #Temperature of room 2
#     "H": ..., #Humidity
#     "Occ1": ..., #Occupancy of room 1
#     "Occ2": ..., #Occupancy of room 2
#     "price_t": ..., #Price
#     "price_previous": ..., #Previous Price
#     "vent_counter": ..., #For how many consecutive hours has the ventilation been on 
#     "low_override_r1": ..., #Is the low-temperature overrule controller of room 1 active 
#     "low_override_r2": ..., #Is the low-temperature overrule controller of room 2 active 
#     "current_time": ... #What is the hour of the day
# }

# CREATE SCENARIO TREE WITH K-MEANS CLUSTERING -- CORRECT APPROACH
def build_tree(state, H, B, N_samples=100):
    """
    Builds scenario tree using iterative Branch & Cluster.
    
    Args:
        state: current state from environment
        H: lookahead horizon
        B: branching factor (clusters per node)
        N_samples: raw samples generated per node before clustering
    """
    
    # root node (tau=0) — known values from state, no uncertainty
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
    queue = [root]
    next_id = 1
    
    while queue:
        parent = queue.pop(0)
        
        if parent["tau"] >= H:
            continue  # leaf node, no children
        
        # --- BRANCH: generate N_samples children from this parent ---
        sample_prices  = []
        sample_prevs   = []
        sample_occ1s   = []
        sample_occ2s   = []
        
        for _ in range(N_samples):
            p  = price_model(parent["price"], parent["price_prev"])
            o1, o2 = next_occupancy_levels(parent["occ1"], parent["occ2"])
            sample_prices.append(p)
            sample_prevs.append(parent["price"])  # same for all samples
            sample_occ1s.append(o1)
            sample_occ2s.append(o2)
        
        # --- CLUSTER: reduce N_samples to B representatives ---
        # feature matrix: each row is one sample [price, occ1, occ2]
        X = np.column_stack([sample_prices, sample_occ1s, sample_occ2s])
        
        km = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
        labels    = km.labels_
        centroids = km.cluster_centers_  # shape (B, 3)
        
        # --- CREATE B child nodes from centroids ---
        for b in range(B):
            cluster_prob = np.sum(labels == b) / N_samples
            
            child = {
                "id":         next_id,
                "tau":        parent["tau"] + 1,
                "parent_id":  parent["id"],
                "price":      centroids[b, 0],   # centroid price
                "price_prev": parent["price"],    # parent price becomes prev
                "occ1":       centroids[b, 1],   # centroid occ1
                "occ2":       centroids[b, 2],   # centroid occ2
                "prob":       parent["prob"] * cluster_prob  # chain rule
            }
            
            nodes.append(child)
            queue.append(child)
            next_id += 1
    
    return nodes







# CREATE SCENARIO TREE - NO CLUSTER ----WRONG
# def build_tree(state, H, B):
#     # state=state variables
#     # H = lookahead horizon
#     # B = branching factor
#     # root node (τ=0) — known values from state, no uncertainty
#     root = {
#         "id":         0,
#         "tau":        0,
#         "parent_id":  None,
#         "price":      state["price_t"],
#         "price_prev": state["price_previous"],
#         "occ1":       state["Occ1"],
#         "occ2":       state["Occ2"],
#         "prob":       1.0
#     }
    
#     nodes = [root]
#     queue = [root]  # BFS queue
#     next_id = 1
    
#     while queue: # goes on until the queue list is empty
#         parent = queue.pop(0)  # take the first element of the list, removes it and returns it 
    
#         if parent["tau"] >= H:
#             continue  # if leaf node, skip the for loop
    
#         for b in range(B):
#         # generates random values
#             child_price = price_model(parent["price"], parent["price_prev"]) # for each branch, generates a random value for the price
#             child_o1, child_o2 = next_occupancy_levels(parent["occ1"], parent["occ2"]) # for each branch, generates a random value for the occupancies
        
#             child = {
#                 "id":         next_id,
#                 "tau":        parent["tau"] + 1,
#                 "parent_id":  parent["id"],
#                 "price":      child_price,
#                 "price_prev": parent["price"],
#                 "occ1":       child_o1,
#                 "occ2":       child_o2,
#                 "prob":       parent["prob"] / B
#         }
        
#             nodes.append(child)
#             queue.append(child)
#             next_id += 1
#     return nodes

# ALTERNATIVE SCENARIO BUILDER (WITH THE FAN) --- WRONG, BUT KEPT FOR COMPARISON
# def build_fan(state, H, N):
#     """
#     Generates N raw scenarios (trajectories) from the current state.
#     Each scenario is a dictionary with H future values for price, occ1, occ2.
    
#     Args:
#         state: current state dictionary from environment
#         H: lookahead horizon
#         N: number of raw scenarios to generate
    
#     Returns:
#         list of N dictionaries, each with keys "price", "occ1", "occ2"
#         each value is a list of H future values (tau=1,...,H)
#     """
#     scenarios = []
    
#     for i in range(N):
#         # initialize trajectory with current known values
#         price_traj  = [state["price_t"], state["price_previous"]]
#         occ1_traj   = [state["Occ1"]]
#         occ2_traj   = [state["Occ2"]]
        
#         # simulate H steps forward
#         for tau in range(H):
#             p_next = price_model(price_traj[-1], price_traj[-2])
#             o1_next, o2_next = next_occupancy_levels(occ1_traj[-1], occ2_traj[-1])
#             price_traj.append(p_next)
#             occ1_traj.append(o1_next)
#             occ2_traj.append(o2_next)
        
#         # store only future values (tau=1,...,H) — tau=0 is already known
#         scenarios.append({
#             "price": price_traj[2:],   # H values: lambda_1, ..., lambda_H
#             "occ1":  occ1_traj[1:],    # H values: kappa1_1, ..., kappa1_H
#             "occ2":  occ2_traj[1:]     # H values: kappa2_1, ..., kappa2_H
#         })
    
#     return scenarios


# SOLVE SP
def solve_sp(state, nodes):
    """
    Builds and solves the multi-stage SP MILP on the scenario tree.
    Returns the here-and-now decisions (p1, p2, v) for tau=0.
    """
 
    model = ConcreteModel()
 
    # ------------------------------------------------------------------
    # SETUP: node lookups and useful groupings
    # ------------------------------------------------------------------
    node_by_id    = {n["id"]: n for n in nodes}
    nodes_future  = [n for n in nodes if n["tau"] >= 1]  # tau=1,2,3
 
    # current absolute time (for T_out indexing)
    t_now = state["current_time"]
 
    # low override initial state (parameters from environment)
    low_override_init = {1: state["low_override_r1"],
                         2: state["low_override_r2"]}
 
    # ventilation inertia from past decisions
    vent_counter    = state["vent_counter"]
    remaining_forced = max(0, min_up_time - vent_counter) if vent_counter > 0 else 0
    v_prev          = 1 if vent_counter > 0 else 0  # was vent ON before tau=0?
 
    # ------------------------------------------------------------------
    # SETS
    # ------------------------------------------------------------------
    model.R     = RangeSet(1, 2)
    model.NODES = Set(initialize=[n["id"] for n in nodes_future])
 
    # ------------------------------------------------------------------
    # VARIABLES
    # ------------------------------------------------------------------
 
    # here-and-now (tau=0) — shared across all scenarios
    model.p0 = Var(model.R, within=NonNegativeReals, bounds=(0, P_max))
    model.v0 = Var(within=Binary)
    model.s0 = Var(within=Binary)  # startup indicator at tau=0
 
    # future nodes (tau >= 1) — one variable per node
    model.p          = Var(model.R, model.NODES, within=NonNegativeReals, bounds=(0, P_max))
    model.v          = Var(model.NODES, within=Binary)
    model.s          = Var(model.NODES, within=Binary)   # startup indicator
    model.temp       = Var(model.R, model.NODES, within=Reals)
    model.hum        = Var(model.NODES, within=NonNegativeReals)
    model.delta_low  = Var(model.R, model.NODES, within=Binary)
    model.delta_high = Var(model.R, model.NODES, within=Binary)
    model.delta_hum  = Var(model.NODES, within=Binary)
 
    # ------------------------------------------------------------------
    # HELPER FUNCTIONS
    # return the parent's variable or state value for each quantity
    # ------------------------------------------------------------------
    def v_par(node):
        """ventilation at parent node"""
        return model.v0 if node["tau"] == 1 else model.v[node["parent_id"]]
 
    def p_par(r, node):
        """heating power at parent node"""
        return model.p0[r] if node["tau"] == 1 else model.p[r, node["parent_id"]]
 
    def temp_par(r, node):
        """temperature at parent node"""
        if node["tau"] == 1:
            return state["T1"] if r == 1 else state["T2"]
        return model.temp[r, node["parent_id"]]
 
    def temp_other_par(r, node):
        """temperature of the OTHER room at parent node"""
        r_other = 3 - r
        if node["tau"] == 1:
            return state["T1"] if r_other == 1 else state["T2"]
        return model.temp[r_other, node["parent_id"]]
 
    def hum_par(node):
        """humidity at parent node"""
        return state["H"] if node["tau"] == 1 else model.hum[node["parent_id"]]
 
    def occ_par(r, node):
        """occupancy at parent node (parameter from tree or state)"""
        if node["tau"] == 1:
            return state["Occ1"] if r == 1 else state["Occ2"]
        parent = node_by_id[node["parent_id"]]
        return parent["occ1"] if r == 1 else parent["occ2"]
 
    def delta_low_par(r, node):
        """low override status at parent — parameter at tau=1, variable otherwise"""
        if node["tau"] == 1:
            return low_override_init[r]  # known from state
        return model.delta_low[r, node["parent_id"]]
 
    # ------------------------------------------------------------------
    # OBJECTIVE FUNCTION
    # minimize expected cost over the lookahead horizon
    # ------------------------------------------------------------------
 
    # certain cost at tau=0
    obj_expr = state["price_t"] * (
        model.p0[1] + model.p0[2] + P_vent * model.v0
    )
 
    # expected cost at future nodes, weighted by probability
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
        nid = n["id"]
        tau = n["tau"]
 
        # outdoor temperature at parent's absolute time
        t_abs = t_now + tau - 1
        t_out = T_out[min(t_abs, len(T_out) - 1)]
 
        for r in [1, 2]:
 
            # C1 — temperature dynamics
            model.c.add(
                model.temp[r, nid] ==
                    temp_par(r, n)
                    + zeta_exch * (temp_other_par(r, n) - temp_par(r, n))
                    - zeta_loss * (temp_par(r, n) - t_out)
                    + zeta_conv * p_par(r, n)
                    - zeta_cool * v_par(n)
                    + zeta_occ  * occ_par(r, n)
            )
 
            # C3 — low temp overrule: activation
            model.c.add(M * model.delta_low[r, nid] >= T_low - model.temp[r, nid])
 
            # C4 — low temp overrule: memory (hysteresis)
            model.c.add(
                M * model.delta_low[r, nid] >=
                (T_ok + epsilon - model.temp[r, nid]) - M * (1 - delta_low_par(r, n))
            )
 
            # C5 — low temp overrule: deactivation above T_ok
            model.c.add(M * (1 - model.delta_low[r, nid]) >= model.temp[r, nid] - T_ok - epsilon)
 
            # C6 — low temp overrule: force power to max
            model.c.add(model.p[r, nid] >= P_max * model.delta_low[r, nid])
 
            # C7 — high temp overrule: activation
            model.c.add(M * model.delta_high[r, nid] >= model.temp[r, nid] - T_high)
 
            # C8 — high temp overrule: deactivation
            model.c.add(M * (1 - model.delta_high[r, nid]) >= T_high - model.temp[r, nid])
 
            # C9 — high temp overrule: force power to zero
            model.c.add(model.p[r, nid] <= P_max * (1 - model.delta_high[r, nid]))
 
        # C2 — humidity dynamics
        model.c.add(
            model.hum[nid] ==
                hum_par(n)
                + eta_occ * (occ_par(1, n) + occ_par(2, n))
                - eta_vent * v_par(n)
        )
 
        # C10 — humidity overrule: activation and deactivation
        model.c.add(M * model.delta_hum[nid] >= model.hum[nid] - H_high)
        model.c.add(M * (1 - model.delta_hum[nid]) >= H_high - model.hum[nid])
 
        # C11 — humidity overrule: force ventilation ON
        model.c.add(model.v[nid] >= model.delta_hum[nid])
 
        # C12 — ventilation startup definition at this node
        model.c.add(model.s[nid] >= model.v[nid] - v_par(n))
        model.c.add(model.s[nid] <= model.v[nid])
        model.c.add(model.s[nid] <= 1 - v_par(n))
 
        # C13 — minimum uptime: if startup at any ancestor within
        # min_up_time-1 steps, this node must have v=1
        ancestor = n
        for depth in range(1, min_up_time):
            if ancestor["parent_id"] is None:
                break
            ancestor = node_by_id[ancestor["parent_id"]]
            if ancestor["tau"] == 0:
                # ancestor is root: startup at tau=0 forces v[nid]=1
                model.c.add(model.v[nid] >= model.s0)
                break
            else:
                model.c.add(model.v[nid] >= model.s[ancestor["id"]])
 
    # ------------------------------------------------------------------
    # HERE-AND-NOW VENTILATION CONSTRAINTS
    # ------------------------------------------------------------------
 
    # startup at tau=0
    model.c.add(model.s0 >= model.v0 - v_prev)
    model.c.add(model.s0 <= model.v0)
    model.c.add(model.s0 <= 1 - v_prev)
 
    # forced ventilation from vent_counter (past decisions)
    if remaining_forced >= 1:
        model.v0.fix(1)  # must be ON at tau=0
 
    for n in nodes_future:
        if n["tau"] == 1 and remaining_forced >= 2:
            model.v[n["id"]].fix(1)  # must be ON at tau=1 for all nodes
 
    # ------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------
    solver = SolverFactory('gurobi')
    result = solver.solve(model)
 
    if result.solver.termination_condition != TerminationCondition.optimal:
        print("[WARNING] SP did not solve to optimality — returning zeros")
        return 0.0, 0.0, 0
 
    # ------------------------------------------------------------------
    # EXTRACT HERE-AND-NOW DECISIONS (tau=0 only)
    # ------------------------------------------------------------------
    p1 = value(model.p0[1])
    p2 = value(model.p0[2])
    v  = int(value(model.v0) > 0.5)
 
    return p1, p2, v 

# ENTRY POINT (called by the environment)
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:14:31 2025

@author: geots
""" 

from pyomo.environ import * 
from sklearn.cluster import KMeans
import numpy as np 
from Utils.PriceProcessRestaurant import price_model 
from OccupancyProcessRestaurant import next_occupancy_levels
from SystemCharacteristics import get_fixed_data

# parameters extraction from system characteristics
data = get_fixed_data()
T           = data['num_timeslots'] # number of hours (10)
T0_prev = data['previous_initial_temperature'] # every day's initial temperature
T0          = data['initial_temperature'] # initial indoor temperature (°C)
H0          = data['initial_humidity'] # initial indoor humidity (%)
P_max       = data['heating_max_power'] # maximum heating power (kW)
zeta_exch   = data['heat_exchange_coeff'] # heat exchange coefficient between rooms
zeta_conv   = data['heating_efficiency_coeff'] # heating efficiency: increase in room temperature per kW of heating power
zeta_loss   = data['thermal_loss_coeff'] # thermal loss coefficient: fraction of indoor-outdoor temperature difference lost per hour    
zeta_cool   = data['heat_vent_coeff'] # ventilation cooling effect: temperature decrease in the room for each hour that ventilation is ON (°C)
zeta_occ    = data['heat_occupancy_coeff'] # occupancy heat gain: temperature increase per hour per person in the room (°C)
T_low       = data['temp_min_comfort_threshold'] # minimum comfortable temperature threshold (°C)
T_ok        = data['temp_OK_threshold'] # comfortable temperature threshold (°C)
T_high      = data['temp_max_comfort_threshold'] # maximum comfortable temperature threshold (°C)
T_out       = data['outdoor_temperature'] # outdoor temperature (°C)
P_vent      = data['ventilation_power'] # power consumption of ventilation system when ON (kW)
H_high      = data['humidity_threshold'] # maximum comfortable humidity threshold (%)
eta_occ     = data['humidity_occupancy_coeff'] # humidity increase per hour per person in the room (%)
eta_vent    = data['humidity_vent_coeff'] # humidity decrease per hour when ventilation is ON (%)
min_up_time = data['vent_min_up_time'] # minimum number of consecutive hours that ventilation must be ON once turned ON (hours)

M = 1000 # big M constant for linearization of logical conditions in the overrule controllers (should be sufficiently large to not cut off any feasible solution, but not too large to avoid numerical issues)
epsilon = 0.01 # small value to express the strict inequality for the controllers rules



# The state will be provided by the environment as the following dictionary

# state = {
#     "T1": ..., #Temperature of room 1
#     "T2": ..., #Temperature of room 2
#     "H": ..., #Humidity
#     "Occ1": ..., #Occupancy of room 1
#     "Occ2": ..., #Occupancy of room 2
#     "price_t": ..., #Price
#     "price_previous": ..., #Previous Price
#     "vent_counter": ..., #For how many consecutive hours has the ventilation been on 
#     "low_override_r1": ..., #Is the low-temperature overrule controller of room 1 active 
#     "low_override_r2": ..., #Is the low-temperature overrule controller of room 2 active 
#     "current_time": ... #What is the hour of the day
# }

# CREATE SCENARIO TREE WITH K-MEANS CLUSTERING -- CORRECT APPROACH
def build_tree(state, H, B, N_samples=100):
    """
    Builds scenario tree using iterative Branch & Cluster.
    
    Args:
        state: current state from environment
        H: lookahead horizon
        B: branching factor (clusters per node)
        N_samples: raw samples generated per node before clustering
    """
    
    # root node (tau=0) — known values from state, no uncertainty
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
    queue = [root]
    next_id = 1
    
    while queue:
        parent = queue.pop(0)
        
        if parent["tau"] >= H:
            continue  # leaf node, no children
        
        # --- BRANCH: generate N_samples children from this parent ---
        sample_prices  = []
        sample_prevs   = []
        sample_occ1s   = []
        sample_occ2s   = []
        
        for _ in range(N_samples):
            p  = price_model(parent["price"], parent["price_prev"])
            o1, o2 = next_occupancy_levels(parent["occ1"], parent["occ2"])
            sample_prices.append(p)
            sample_prevs.append(parent["price"])  # same for all samples
            sample_occ1s.append(o1)
            sample_occ2s.append(o2)
        
        # --- CLUSTER: reduce N_samples to B representatives ---
        # feature matrix: each row is one sample [price, occ1, occ2]
        X = np.column_stack([sample_prices, sample_occ1s, sample_occ2s])
        
        km = KMeans(n_clusters=B, random_state=0, n_init=10).fit(X)
        labels    = km.labels_
        centroids = km.cluster_centers_  # shape (B, 3)
        
        # --- CREATE B child nodes from centroids ---
        for b in range(B):
            cluster_prob = np.sum(labels == b) / N_samples
            
            child = {
                "id":         next_id,
                "tau":        parent["tau"] + 1,
                "parent_id":  parent["id"],
                "price":      centroids[b, 0],   # centroid price
                "price_prev": parent["price"],    # parent price becomes prev
                "occ1":       centroids[b, 1],   # centroid occ1
                "occ2":       centroids[b, 2],   # centroid occ2
                "prob":       parent["prob"] * cluster_prob  # chain rule
            }
            
            nodes.append(child)
            queue.append(child)
            next_id += 1
    
    return nodes







# CREATE SCENARIO TREE - NO CLUSTER ----WRONG
# def build_tree(state, H, B):
#     # state=state variables
#     # H = lookahead horizon
#     # B = branching factor
#     # root node (τ=0) — known values from state, no uncertainty
#     root = {
#         "id":         0,
#         "tau":        0,
#         "parent_id":  None,
#         "price":      state["price_t"],
#         "price_prev": state["price_previous"],
#         "occ1":       state["Occ1"],
#         "occ2":       state["Occ2"],
#         "prob":       1.0
#     }
    
#     nodes = [root]
#     queue = [root]  # BFS queue
#     next_id = 1
    
#     while queue: # goes on until the queue list is empty
#         parent = queue.pop(0)  # take the first element of the list, removes it and returns it 
    
#         if parent["tau"] >= H:
#             continue  # if leaf node, skip the for loop
    
#         for b in range(B):
#         # generates random values
#             child_price = price_model(parent["price"], parent["price_prev"]) # for each branch, generates a random value for the price
#             child_o1, child_o2 = next_occupancy_levels(parent["occ1"], parent["occ2"]) # for each branch, generates a random value for the occupancies
        
#             child = {
#                 "id":         next_id,
#                 "tau":        parent["tau"] + 1,
#                 "parent_id":  parent["id"],
#                 "price":      child_price,
#                 "price_prev": parent["price"],
#                 "occ1":       child_o1,
#                 "occ2":       child_o2,
#                 "prob":       parent["prob"] / B
#         }
        
#             nodes.append(child)
#             queue.append(child)
#             next_id += 1
#     return nodes

# ALTERNATIVE SCENARIO BUILDER (WITH THE FAN) --- WRONG, BUT KEPT FOR COMPARISON
# def build_fan(state, H, N):
#     """
#     Generates N raw scenarios (trajectories) from the current state.
#     Each scenario is a dictionary with H future values for price, occ1, occ2.
    
#     Args:
#         state: current state dictionary from environment
#         H: lookahead horizon
#         N: number of raw scenarios to generate
    
#     Returns:
#         list of N dictionaries, each with keys "price", "occ1", "occ2"
#         each value is a list of H future values (tau=1,...,H)
#     """
#     scenarios = []
    
#     for i in range(N):
#         # initialize trajectory with current known values
#         price_traj  = [state["price_t"], state["price_previous"]]
#         occ1_traj   = [state["Occ1"]]
#         occ2_traj   = [state["Occ2"]]
        
#         # simulate H steps forward
#         for tau in range(H):
#             p_next = price_model(price_traj[-1], price_traj[-2])
#             o1_next, o2_next = next_occupancy_levels(occ1_traj[-1], occ2_traj[-1])
#             price_traj.append(p_next)
#             occ1_traj.append(o1_next)
#             occ2_traj.append(o2_next)
        
#         # store only future values (tau=1,...,H) — tau=0 is already known
#         scenarios.append({
#             "price": price_traj[2:],   # H values: lambda_1, ..., lambda_H
#             "occ1":  occ1_traj[1:],    # H values: kappa1_1, ..., kappa1_H
#             "occ2":  occ2_traj[1:]     # H values: kappa2_1, ..., kappa2_H
#         })
    
#     return scenarios


def solve_sp(state, nodes):
    """
    Builds and solves the multi-stage SP MILP on the scenario tree.
    Uses the correct model from the Solution to Assignment Part A.
    Returns the here-and-now decisions (p1, p2, v) for tau=0.
    """

    model = ConcreteModel()

    # ------------------------------------------------------------------
    # SETUP: node lookups and useful groupings
    # ------------------------------------------------------------------
    node_by_id   = {n["id"]: n for n in nodes}
    nodes_future = [n for n in nodes if n["tau"] >= 1]  # tau=1,2,3

    # current absolute time (for T_out indexing)
    t_now = state["current_time"]

    # low override initial state from environment (tau=0)
    low_override_init = {1: state["low_override_r1"],
                         2: state["low_override_r2"]}

    # ventilation inertia from past decisions
    vent_counter     = state["vent_counter"]
    remaining_forced = max(0, min_up_time - vent_counter) if vent_counter > 0 else 0
    v_prev           = 1 if vent_counter > 0 else 0  # was vent ON before tau=0?

    # ------------------------------------------------------------------
    # SETS
    # ------------------------------------------------------------------
    model.R     = RangeSet(1, 2)
    model.NODES = Set(initialize=[n["id"] for n in nodes_future])

    # ------------------------------------------------------------------
    # VARIABLES
    # ------------------------------------------------------------------

    # here-and-now (tau=0) — shared across all scenarios
    model.p0 = Var(model.R, within=NonNegativeReals, bounds=(0, P_max))
    model.v0 = Var(within=Binary)
    model.s0 = Var(within=Binary)  # ventilation startup indicator at tau=0

    # future nodes (tau >= 1) — one variable per node
    model.p    = Var(model.R, model.NODES, within=NonNegativeReals, bounds=(0, P_max))
    model.v    = Var(model.NODES, within=Binary)
    model.s    = Var(model.NODES, within=Binary)   # ventilation startup indicator
    model.temp = Var(model.R, model.NODES, within=Reals)
    model.hum  = Var(model.NODES, within=NonNegativeReals)

    # --- solution model: 3 separate variables for low-temp overrule ---
    model.y_low = Var(model.R, model.NODES, within=Binary)  # 1 if temp < T_low
    model.y_ok  = Var(model.R, model.NODES, within=Binary)  # 1 if temp > T_ok
    model.u     = Var(model.R, model.NODES, within=Binary)  # 1 if overrule active

    # high-temp and humidity overrule
    model.delta_high = Var(model.R, model.NODES, within=Binary)
    model.delta_hum  = Var(model.NODES, within=Binary)

    # ------------------------------------------------------------------
    # HELPER FUNCTIONS
    # return parent's variable or known state value for each quantity
    # ------------------------------------------------------------------
    def v_par(node):
        """ventilation at parent node"""
        return model.v0 if node["tau"] == 1 else model.v[node["parent_id"]]

    def p_par(r, node):
        """heating power at parent node"""
        return model.p0[r] if node["tau"] == 1 else model.p[r, node["parent_id"]]

    def temp_par(r, node):
        """temperature at parent node"""
        if node["tau"] == 1:
            return state["T1"] if r == 1 else state["T2"]
        return model.temp[r, node["parent_id"]]

    def temp_other_par(r, node):
        """temperature of the OTHER room at parent node"""
        r_other = 3 - r
        if node["tau"] == 1:
            return state["T1"] if r_other == 1 else state["T2"]
        return model.temp[r_other, node["parent_id"]]

    def hum_par(node):
        """humidity at parent node"""
        return state["H"] if node["tau"] == 1 else model.hum[node["parent_id"]]

    def occ_par(r, node):
        """occupancy at parent node — parameter from tree or state"""
        if node["tau"] == 1:
            return state["Occ1"] if r == 1 else state["Occ2"]
        parent = node_by_id[node["parent_id"]]
        return parent["occ1"] if r == 1 else parent["occ2"]

    def u_par(r, node):
        """low-temp overrule status at parent — parameter at tau=1, variable otherwise"""
        if node["tau"] == 1:
            return low_override_init[r]  # known from state
        return model.u[r, node["parent_id"]]

    # ------------------------------------------------------------------
    # OBJECTIVE FUNCTION
    # minimize expected cost over the lookahead horizon
    # ------------------------------------------------------------------

    # certain cost at tau=0 (price is known)
    obj_expr = state["price_t"] * (
        model.p0[1] + model.p0[2] + P_vent * model.v0
    )

    # expected cost at future nodes, weighted by probability of reaching them
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
        nid = n["id"]
        tau = n["tau"]

        # outdoor temperature at the absolute time of this node's parent
        t_abs = t_now + tau - 1
        t_out = T_out[min(t_abs, len(T_out) - 1)]

        for r in [1, 2]:

            # C1 — temperature dynamics (from solution eq. 2)
            model.c.add(
                model.temp[r, nid] ==
                    temp_par(r, n)
                    + zeta_exch * (temp_other_par(r, n) - temp_par(r, n))
                    - zeta_loss * (temp_par(r, n) - t_out)
                    + zeta_conv * p_par(r, n)
                    - zeta_cool * v_par(n)
                    + zeta_occ  * occ_par(r, n)
            )

            # ----------------------------------------------------------
            # LOW-TEMP OVERRULE — solution model (eq. 8-16)
            # three separate binary variables: y_low, y_ok, u
            # ----------------------------------------------------------

            # C3a — detect if temp < T_low (eq. 8-9)
            model.c.add(model.temp[r, nid] <= T_low + M * (1 - model.y_low[r, nid]))
            model.c.add(model.temp[r, nid] >= T_low - M * model.y_low[r, nid])

            # C3b — detect if temp > T_ok (eq. 10-11)
            model.c.add(model.temp[r, nid] >= T_ok - M * (1 - model.y_ok[r, nid]))
            model.c.add(model.temp[r, nid] <= T_ok + M * model.y_ok[r, nid])

            # C4 — activation: if temp < T_low → u=1 (eq. 12)
            model.c.add(model.u[r, nid] >= model.y_low[r, nid])

            # C5 — memory: u can only stay ON if it was ON before (eq. 13)
            model.c.add(model.u[r, nid] <= u_par(r, n) + model.y_low[r, nid])

            # C6 — force power to max when overrule is active (eq. 14)
            model.c.add(model.p[r, nid] >= P_max * model.u[r, nid])

            # C7 — deactivation: u must turn OFF if temp > T_ok (eq. 15-16)
            model.c.add(model.u[r, nid] >= u_par(r, n) - model.y_ok[r, nid])
            model.c.add(model.u[r, nid] <= 1 - model.y_ok[r, nid])

            # ----------------------------------------------------------
            # HIGH-TEMP OVERRULE (eq. 5-7)
            # ----------------------------------------------------------

            # C8 — detect if temp >= T_high
            model.c.add(model.temp[r, nid] >= T_high - M * (1 - model.delta_high[r, nid]))
            model.c.add(model.temp[r, nid] <= T_high + M * model.delta_high[r, nid])

            # C9 — force power to zero when high-temp overrule is active
            model.c.add(model.p[r, nid] <= P_max * (1 - model.delta_high[r, nid]))

        # C2 — humidity dynamics (eq. 3)
        model.c.add(
            model.hum[nid] ==
                hum_par(n)
                + eta_occ * (occ_par(1, n) + occ_par(2, n))
                - eta_vent * v_par(n)
        )

        # ----------------------------------------------------------
        # HUMIDITY OVERRULE (eq. 21)
        # ----------------------------------------------------------

        # C10 — detect if humidity > H_high
        model.c.add(M * model.delta_hum[nid] >= model.hum[nid] - H_high)
        model.c.add(M * (1 - model.delta_hum[nid]) >= H_high - model.hum[nid])

        # C11 — force ventilation ON when humidity overrule is active
        model.c.add(model.v[nid] >= model.delta_hum[nid])

        # ----------------------------------------------------------
        # VENTILATION INERTIA (eq. 17-20)
        # ----------------------------------------------------------

        # C12 — startup detection at this node
        model.c.add(model.s[nid] >= model.v[nid] - v_par(n))
        model.c.add(model.s[nid] <= model.v[nid])
        model.c.add(model.s[nid] <= 1 - v_par(n))

        # C13 — minimum uptime: if any ancestor within min_up_time-1
        # steps had a startup, this node must have v=1
        ancestor = n
        for depth in range(1, min_up_time):
            if ancestor["parent_id"] is None:
                break
            ancestor = node_by_id[ancestor["parent_id"]]
            if ancestor["tau"] == 0:
                # ancestor is root: startup at tau=0 forces v[nid]=1
                model.c.add(model.v[nid] >= model.s0)
                break
            else:
                model.c.add(model.v[nid] >= model.s[ancestor["id"]])

    # ------------------------------------------------------------------
    # HERE-AND-NOW VENTILATION CONSTRAINTS (tau=0)
    # ------------------------------------------------------------------

    # startup definition at tau=0
    model.c.add(model.s0 >= model.v0 - v_prev)
    model.c.add(model.s0 <= model.v0)
    model.c.add(model.s0 <= 1 - v_prev)

    # forced ventilation from vent_counter (past decisions carry over)
    if remaining_forced >= 1:
        model.v0.fix(1)  # tau=0 must be ON

    for n in nodes_future:
        if n["tau"] == 1 and remaining_forced >= 2:
            model.v[n["id"]].fix(1)  # tau=1 must be ON for all nodes

    # ------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------
    solver = SolverFactory('gurobi')
    result = solver.solve(model)

    if result.solver.termination_condition != TerminationCondition.optimal:
        print("[WARNING] SP did not solve to optimality — returning zeros")
        return 0.0, 0.0, 0

    # ------------------------------------------------------------------
    # EXTRACT HERE-AND-NOW DECISIONS (tau=0 only)
    # ------------------------------------------------------------------
    p1 = value(model.p0[1])
    p2 = value(model.p0[2])
    v  = int(value(model.v0) > 0.5)

    return p1, p2, v 

# ENTRY POINT (called by the environment)
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

