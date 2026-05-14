from Policies import SP_policy_30, SP_policy_30_v2, ADP_policy_30, DUMMY_policy_30
from Environment import run_environment
from importlib import import_module

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
import time
import pandas as pd
import os



def save_results_to_csv(results, policy_name):
    rows = []
    for day_entry in results["logs"]:
        day = day_entry["day"]
        log = day_entry["log"]
        for i in range(len(log["hour"])):
            rows.append({
                "Day": day,
                "Hour": log["hour"][i],
                "Price": log["price"][i],
                "Occupancy_R1": log["occ1"][i],
                "Occupancy_R2": log["occ2"][i],
                "Temp_Room1": log["T1"][i],
                "Temp_Room2": log["T2"][i],
                "Power_Heater1": log["P1"][i],
                "Power_Heater2": log["P2"][i],
                "Ventilation_On": log["V"][i],
                "Humidity": log["H"][i]
            })

    df = pd.DataFrame(rows)

    # Determine filename with version if necessary
    base_name = policy_name[9:] + "_logs"
    filename = f"{base_name}.csv"
    counter = 0
    while os.path.exists(f"Policy_log_files/{filename}"):
        counter += 1
        filename = f"{base_name}_{counter}.csv"

    saving_path = f"Policy_log_files/{filename}"
    df.to_csv(saving_path, index=False)
    print(f"\nLogs saved to {saving_path}")


def import_policy_module(policy_name):
    if isinstance(policy_name, str):
        return import_module(f"{policy_name}")
    return policy_name


def run_environment_with_policy_name(policy_name, start, end, plot=False):
    policy_module = import_policy_module(policy_name)
    return run_environment(policy_module, start, end, plot)


def run_environment_in_parallel(policy, n_experiments, n_workers):    
    # Calculate chunk size for each worker
    chunk_size = n_experiments // n_workers
    day_ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_workers)]
    
    # Handle any remaining days (if N_EXPERIMENTS not divisible by n_workers)
    if n_experiments % n_workers != 0:
        day_ranges[-1] = (day_ranges[-1][0], n_experiments)
    
    print(f"Day ranges for each worker: {day_ranges}")

    # Runs environment in parallel with workers handling different day ranges
    policy_name = policy.__name__ if hasattr(policy, '__name__') else str(policy)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(
            executor.map(
                run_environment_with_policy_name,
                repeat(policy_name),
                [start for start, end in day_ranges],
                [end for start, end in day_ranges]
            )
        )

    # Combine results from all workers (each worker processed a chunk of days)
    all_objectives = []
    all_logs = []

    for avg_value, worker_result in results:
        all_objectives.extend(worker_result["objectives"])
        all_logs.extend(worker_result["logs"])

    avg_objective_value = np.mean(all_objectives)

    return avg_objective_value, {
        "objectives": all_objectives,
        "logs": all_logs
    }



# Variables to set before running the environment:
POLICY          = ADP_policy_30
N_EXPERIMENTS   = 100
PLOT_RESULTS    = False
RUN_IN_PARALLEL = True
N_WORKERS       = 8


if __name__ == "__main__":
    print("Running policy: ", POLICY.__name__)

    start_time = time.time()

    if RUN_IN_PARALLEL:
        print("Running in parallel with", N_WORKERS, "workers")
        avg_objective_value, results = run_environment_in_parallel(
            policy=POLICY,
            n_experiments=N_EXPERIMENTS,
            n_workers=N_WORKERS
        )

    else:
        print("Running sequentially")
        avg_objective_value, results = run_environment(
            policy=POLICY,
            n_experiments=N_EXPERIMENTS,
            plot=PLOT_RESULTS
        )


    all_objectives = np.array(results["objectives"])

    # mean_cost_per_day = np.mean(all_objectives, axis=0)

    # print("\nMean cost per day:")
    # for day, cost in enumerate(mean_cost_per_day):
    #     print(f"Day {day + 1}: {cost:.2f}")

    print("\nSummary:")
    print(f"Number of days evaluated: {len(all_objectives)}")
    print(f"Average daily cost over all days: {avg_objective_value:.2f}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


    # Save logs and objectives to a file for later analysis
    save_results_to_csv(results, POLICY.__name__)

