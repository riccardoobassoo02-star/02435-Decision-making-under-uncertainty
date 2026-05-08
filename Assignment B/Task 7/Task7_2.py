# imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import csv
from pyomo.environ import *
from scipy.optimize import minimize as scipy_minimize 
from scipy.optimize import LinearConstraint, Bounds
from pathlib import Path

# Data loading 
FILE_DIR = Path(__file__).parent

def fetch_task7_data():
    num_timeslots = 10
    return {
        'num_timeslots'           : num_timeslots,
        'P_mall'                  : 45,
        'Temperature_reference'   : 21,
        'initial_temperature'     : 21.0,
        'heating_max_power'       : 3.0,
        'heat_exchange_coeff'     : 0.6,
        'heating_efficiency_coeff': 1.0,
        'thermal_loss_coeff'      : 0.1,
        'heat_vent_coeff'         : 0.7,   # ventilation always ON in Task 7
        'heat_occupancy_coeff'    : 0.02,
        'outdoor_temperature'     : [
            3 * np.sin(2 * np.pi * t / num_timeslots - np.pi / 2)
            for t in range(num_timeslots)
        ],
    }

data      = fetch_task7_data()
T         = data['num_timeslots']            # number of decision epochs (10)
P_mall    = data['P_mall']                   # mall-wide power cap (kW)
T_ref     = data['Temperature_reference']    # reference temperature (°C)
T0        = data['initial_temperature']      # initial temperature, all rooms (°C)
P_max     = data['heating_max_power']        # max heater power per room (kW)
zeta_exch = data['heat_exchange_coeff']      # heat exchange between rooms
zeta_conv = data['heating_efficiency_coeff'] # heating efficiency
zeta_loss = data['thermal_loss_coeff']       # thermal loss to outdoors
zeta_cool = data['heat_vent_coeff']          # ventilation cooling (always ON)
zeta_occ  = data['heat_occupancy_coeff']     # occupancy heat gain
T_out     = data['outdoor_temperature']      # outdoor temperature profile (list)

N_stores  = 15   # total number of stores in the mall

# Occupancy data — same for all stores as per problem statement
occ_r1 = np.zeros(T)
occ_r2 = np.zeros(T)
with open(FILE_DIR / 'Task7Occupancies.csv', 'r') as f:
    for row in csv.reader(f):
        if not row:
            continue
        if row[-1].strip() == 'room 1':
            occ_r1 = np.array([float(x) for x in row[:-1] if x.strip()])
        elif row[-1].strip() == 'room 2':
            occ_r2 = np.array([float(x) for x in row[:-1] if x.strip()])

def _build_temperature_matrices():
    """
    Unroll the linear temperature dynamics into matrix form:
        temp_vec = M_mat @ [p1; p2] + b_free
    where temp_vec in R^{2T} collects post-decision temperatures for both rooms
    (time indices 1 ... T, matching the objective).
    """
    a  = 1 - zeta_exch - zeta_loss
    b  = zeta_exch
    c1 = zeta_loss * np.array(T_out) - zeta_cool + zeta_occ * occ_r1
    c2 = zeta_loss * np.array(T_out) - zeta_cool + zeta_occ * occ_r2

    A_state = np.array([[a, b], [b, a]])
    B_ctrl  = np.array([[zeta_conv, 0.0], [0.0, zeta_conv]])

    A_pows = [np.eye(2)]
    for _ in range(T):
        A_pows.append(A_pows[-1] @ A_state)

    M_mat  = np.zeros((2 * T, 2 * T))
    b_free = np.zeros(2 * T)

    x0 = np.array([T0, T0])
    for t in range(T):
        xf = A_pows[t + 1] @ x0
        for s in range(t + 1):
            xf += A_pows[t - s] @ np.array([c1[s], c2[s]])
        b_free[t]     = xf[0]
        b_free[T + t] = xf[1]

    for s in range(T):
        for t in range(s, T):
            Be = A_pows[t - s] @ B_ctrl
            M_mat[t,     s]     = Be[0, 0]
            M_mat[t,     T + s] = Be[0, 1]
            M_mat[T + t, s]     = Be[1, 0]
            M_mat[T + t, T + s] = Be[1, 1]

    return M_mat, b_free

M_MAT, B_FREE = _build_temperature_matrices()


def solve_centralized():
    """
    Solve the centralized QP (all stores, shared power cap).
    Returns: obj_val (float), p_opt dict {(n, r, t): power}  (n is 1-indexed).
    """
    BLOCK = 2 * T
    x_dim = N_stores * BLOCK

    H_total = np.zeros((x_dim, x_dim))
    g_total = np.zeros(x_dim)
    for n in range(N_stores):
        w_n = n + 2              # 0-based n → 1-based store n+1 → weight n+2
        sl  = slice(n * BLOCK, (n + 1) * BLOCK)
        H_total[sl, sl] = 2 * w_n * M_MAT.T @ M_MAT
        g_total[sl]     = 2 * w_n * M_MAT.T @ (B_FREE - T_ref)

    def obj_and_grad(x):
        return 0.5 * x @ H_total @ x + g_total @ x, H_total @ x + g_total

    # shared power cap: sum_n (p1_n,t + p2_n,t) <= P_mall for each t
    A_con = np.zeros((T, x_dim))
    for t in range(T):
        for n in range(N_stores):
            A_con[t, n * BLOCK + t]     = 1.0
            A_con[t, n * BLOCK + T + t] = 1.0

    result = scipy_minimize(
        obj_and_grad, np.zeros(x_dim), jac=True, method='trust-constr',
        bounds=Bounds(0, P_max),
        constraints=LinearConstraint(A_con, -np.inf, P_mall),
        options={'maxiter': 3000, 'gtol': 1e-9, 'verbose': 0}
    )

    x_opt  = result.x
    p1_all = np.array([x_opt[n * BLOCK : n * BLOCK + T]       for n in range(N_stores)])
    p2_all = np.array([x_opt[n * BLOCK + T : (n + 1) * BLOCK] for n in range(N_stores)])

    obj_val = sum(
        (n + 2) * np.sum(
            (M_MAT @ np.concatenate([p1_all[n], p2_all[n]]) + B_FREE - T_ref) ** 2
        )
        for n in range(N_stores)
    )
    p_opt = {
        (n + 1, r, t): (p1_all[n, t] if r == 1 else p2_all[n, t])
        for n in range(N_stores) for r in [1, 2] for t in range(T)
    }
    return obj_val, p_opt


def solve_store_subproblem(n, lambdas):
    """
    n       : store index, 1-based (1 ... 15)
    lambdas : np.array of shape (T,) — current dual variables for the power cap
    Returns : p_opt {(r, t)}, temp_opt {(r, t)}, comfort cost (float)
    """
    w_n = n + 1   # w_1=2, w_2=3, ..., w_15=16

    model = ConcreteModel()

    # sets
    model.R   = RangeSet(1, 2)
    model.T_d = RangeSet(0, T - 1)   # decision epochs  t = 0, ..., T-1
    model.T_s = RangeSet(0, T)       # temperature time points  t = 0, ..., T

    # decision variables: heater power for each room at each decision epoch
    model.p    = Var(model.R, model.T_d, within=NonNegativeReals, bounds=(0, P_max))

    # state variables: room temperature at each time point
    model.temp = Var(model.R, model.T_s, within=Reals)

    # objective: local comfort cost + Lagrangian dual penalty
    model.obj = Objective(
        expr=sum(
            w_n * (model.temp[r, t] - T_ref) ** 2
            for r in model.R for t in range(1, T + 1)
        ) + sum(
            lambdas[t] * (model.p[1, t] + model.p[2, t])
            for t in model.T_d
        ),
        sense=minimize
    )

    # 1. initial temperature conditions — same pattern as Task 1
    model.temp_init = ConstraintList()
    for r in model.R:
        model.temp_init.add(model.temp[r, 0] == T0)

    # 2. temperature dynamics — recycled directly from Task 1
    #    only change vs Task 1: v[t] = 1 always (ventilation always ON)
    model.temp_dyn = ConstraintList()
    for t in model.T_d:
        for r in model.R:
            occ = occ_r1[t] if r == 1 else occ_r2[t]
            model.temp_dyn.add(
                model.temp[r, t + 1] == model.temp[r, t]
                                      + zeta_conv * model.p[r, t]
                                      - zeta_loss * (model.temp[r, t] - T_out[t])
                                      + zeta_exch * (model.temp[3 - r, t] - model.temp[r, t])
                                      - zeta_cool * 1          # ventilation always ON
                                      + zeta_occ * occ
            )

    # solver call
    solver = SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    result = solver.solve(model)

    if result.solver.termination_condition != TerminationCondition.optimal:
        print(f'Warning: store {n} subproblem did not solve to optimality')

    p_opt    = {(r, t): value(model.p[r, t])    for r in [1, 2] for t in range(T)}
    temp_opt = {(r, t): value(model.temp[r, t]) for r in [1, 2] for t in range(T + 1)}

    # comfort cost without lambda penalty (tracks true primal objective)
    comfort_cost = sum(
        w_n * (temp_opt[r, t] - T_ref) ** 2
        for r in [1, 2] for t in range(1, T + 1)
    )
    return p_opt, temp_opt, comfort_cost


def run_dual_decomposition(alpha_func, max_iter=100, verbose=True):
    """
    alpha_func(k) -> step size at iteration k (0-indexed).
    Returns:
        obj_history    (max_iter,)    primal comfort objective per iteration
        lambda_history (max_iter, T)  dual variables per iteration
        viol_history   (max_iter, T)  constraint violation per iteration
        p_final        dict {(n,r,t)} heater powers at last iteration
    """
    lambdas = np.zeros(T)

    obj_history    = np.zeros(max_iter)
    lambda_history = np.zeros((max_iter, T))
    viol_history   = np.zeros((max_iter, T))

    for k in range(max_iter):
        total_power   = np.zeros(T)
        total_comfort = 0.0
        p_all         = {}

        for n in range(1, N_stores + 1):
            p_opt, _, comfort = solve_store_subproblem(n, lambdas)
            total_comfort += comfort
            for t in range(T):
                total_power[t] += p_opt[1, t] + p_opt[2, t]
            for (r, t), val in p_opt.items():
                p_all[n, r, t] = val

        # subgradient: positive means constraint is violated (over-consumption)
        subgradient = total_power - P_mall

        obj_history[k]    = total_comfort
        lambda_history[k] = lambdas.copy()
        viol_history[k]   = subgradient.copy()

        # projected subgradient dual update
        alpha   = alpha_func(k)
        lambdas = np.maximum(0.0, lambdas + alpha * subgradient)

        if verbose and (k + 1) % 25 == 0:
            print(f'    k={k + 1:3d}:  obj={total_comfort:,.1f}  '
                  f'max_viol={np.max(np.abs(subgradient)):.4f}  '
                  f'alpha={alpha:.5f}')

    return obj_history, lambda_history, viol_history, p_all


# Execute

print('=' * 65)
print('Solving centralized problem (reference)...')
obj_central, p_central = solve_centralized()
power_central = np.array([
    sum(p_central[n, r, t] for n in range(1, N_stores + 1) for r in [1, 2])
    for t in range(T)
])
print(f'  Centralized objective : {obj_central:,.4f}')
print(f'  Max power used        : {power_central.max():.3f} kW  (limit: {P_mall})')
print()

alphas_fixed  = [0.001, 0.01, 0.1, 1, 10]
alpha_configs = {
    **{f'α={a}': (lambda k, a=a: a) for a in alphas_fixed},
    'Adaptive α₀=5': lambda k: 5.0 / (1 + k)
}

results = {}
for name, alpha_func in alpha_configs.items():
    print(f'Running distributed decomposition: {name}')
    obj_h, lam_h, viol_h, p_f = run_dual_decomposition(alpha_func, max_iter=100)
    results[name] = {'obj': obj_h, 'lambda': lam_h, 'violation': viol_h, 'p': p_f}
    print(f'  -> Final obj={obj_h[-1]:,.1f}   '
          f'gap={obj_h[-1] - obj_central:+,.1f}   '
          f'max_viol={np.max(np.abs(viol_h[-1])):.4f}\n')


# Plotting
OUTPUT_DIR = FILE_DIR / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

iters       = np.arange(1, 101)
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
adapt_color = '#8c564b'
t_colors    = plt.cm.tab10(np.linspace(0, 1, T))

# Figure 1: Objective value vs iterations
fig1, ax1 = plt.subplots(figsize=(10, 6))
for (a, color) in zip(alphas_fixed, line_colors):
    ax1.plot(iters, results[f'α={a}']['obj'], label=f'α={a}', color=color, linewidth=1.8)
ax1.plot(iters, results['Adaptive α₀=5']['obj'],
         label='Adaptive α₀=5', color=adapt_color, linewidth=2.2, linestyle='--')
ax1.axhline(obj_central, color='k', linestyle=':', linewidth=2,
            label=f'Centralized optimal ({obj_central:,.0f})')
ax1.set_xlabel('Iteration', fontsize=13)
ax1.set_ylabel('Primal Comfort Objective', fontsize=13)
ax1.set_title('Dual Decomposition — Objective Value vs. Iterations', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(OUTPUT_DIR / 'task7_fig1_objective.png', dpi=150)
plt.close(fig1)
print('Saved: task7_fig1_objective.png')

# Figure 2: Multiplier evolution
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
for idx, name in enumerate(alpha_configs.keys()):
    ax = axes2.flatten()[idx]
    for t in range(T):
        ax.plot(iters, results[name]['lambda'][:, t],
                color=t_colors[t], linewidth=1.5, label=f't={t + 1}')
    ax.set_title(f'Multipliers lambda_t   [{name}]', fontsize=10)
    ax.set_xlabel('Iteration', fontsize=9)
    ax.set_ylabel('lambda_t', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, ncol=2, title='Hour t')
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / 'task7_fig2_lambdas.png', dpi=150)
plt.close(fig2)
print('Saved: task7_fig2_lambdas.png')

# Figure 3: Constraint violation
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
for idx, name in enumerate(alpha_configs.keys()):
    ax = axes3.flatten()[idx]
    for t in range(T):
        ax.plot(iters, results[name]['violation'][:, t],
                color=t_colors[t], linewidth=1.5, label=f't={t + 1}')
    ax.axhline(0, color='k', linestyle='--', linewidth=1.2)
    ax.set_title(f'sum_n p_n,t - P_mall   [{name}]', fontsize=10)
    ax.set_xlabel('Iteration', fontsize=9)
    ax.set_ylabel('Violation (kW)', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, ncol=2, title='Hour t')
plt.tight_layout()
fig3.savefig(OUTPUT_DIR / 'task7_fig3_violations.png', dpi=150)
plt.close(fig3)
print('Saved: task7_fig3_violations.png')

# Figure 4: Energy per store
p_adap      = results['Adaptive α₀=5']['p']
energy_dist = np.array([sum(p_adap[n, r, t]    for r in [1, 2] for t in range(T)) for n in range(1, N_stores + 1)])
energy_cent = np.array([sum(p_central[n, r, t] for r in [1, 2] for t in range(T)) for n in range(1, N_stores + 1)])
weights      = np.array([n + 1 for n in range(1, N_stores + 1)])
store_labels = [f'S{n}' for n in range(1, N_stores + 1)]
x_pos        = np.arange(N_stores)

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
axes4[0].bar(x_pos - 0.2, energy_cent, 0.4, label='Centralized', color='orange', alpha=0.85)
axes4[0].bar(x_pos + 0.2, energy_dist, 0.4, label='Adaptive α₀=5 (distributed)', color='steelblue', alpha=0.85)
axes4[0].set_xticks(x_pos)
axes4[0].set_xticklabels(store_labels, fontsize=9)
axes4[0].set_ylabel('Total Energy Consumed (kWh)', fontsize=12)
axes4[0].set_title('Energy per Store: Centralized vs Distributed', fontsize=12)
axes4[0].legend(fontsize=10)
axes4[0].grid(True, alpha=0.3, axis='y')
axes4[1].bar(x_pos, weights, color='coral', edgecolor='darkred', alpha=0.85)
axes4[1].set_xticks(x_pos)
axes4[1].set_xticklabels(store_labels, fontsize=9)
axes4[1].set_ylabel('Comfort Weight  w_n = n+1', fontsize=12)
axes4[1].set_title('Comfort Weights per Store', fontsize=12)
axes4[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig4.savefig(OUTPUT_DIR / 'task7_fig4_energy_per_store.png', dpi=150)
plt.close(fig4)
print('Saved: task7_fig4_energy_per_store.png')

# Summary table
print()
print('=' * 70)
print(f"{'Config':<22} {'Final Obj':>12} {'Gap to Opt':>12} {'Max|Viol|':>12}")
print('-' * 70)
for name in alpha_configs.keys():
    obj_f = results[name]['obj'][-1]
    gap   = obj_f - obj_central
    max_v = np.max(np.abs(results[name]['violation'][-1]))
    print(f'{name:<22} {obj_f:>12,.1f} {gap:>+12,.1f} {max_v:>12.4f}')
print(f'\n  Centralized optimal: {obj_central:,.4f}')

print()
print(f"{'Store':<8} {'Weight':>8} {'Cent kWh':>12} {'Dist kWh':>12}")
for n in range(1, N_stores + 1):
    print(f'Store {n:<2}  {weights[n-1]:>8}  {energy_cent[n-1]:>12.3f}  {energy_dist[n-1]:>12.3f}')

pd.DataFrame({
    'Store'      : list(range(1, N_stores + 1)),
    'Weight'     : weights.tolist(),
    'Energy_Cent': energy_cent.tolist(),
    'Energy_Dist': energy_dist.tolist(),
}).to_csv(OUTPUT_DIR / 'task7_energy_per_store.csv', index=False)

print('\nAll done. Results saved to results/')
