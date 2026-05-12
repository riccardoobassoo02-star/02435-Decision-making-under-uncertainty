from Utils import v2_SystemCharacteristics, Checks
from Policies import SP_policy_30, ADP_policy_30
import numpy as np
import matplotlib.pyplot as plt
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import repeat
import os
import time
warnings.filterwarnings("ignore")

# Runs the environment in parallel
#   - Each thread runs all the experiments (days)
#   - Number of threads = Number of repetitions per experiment

# Choose number of experiments and repetitions
N_EXPERIMENTS = 100
N_WORKERS = 8
N_REPETITIONS = 1  # Each worker runs once, handling multiple days
POLICY = ADP_policy_30 #SP_policy_30_v2


def update_overrule_controler_state(overrule_state, temperature, data):
    """
    Turns on the overrule controller if the temperature is below the minimum comfort threshold,
    and turns it off if the temperature is above the OK threshold.
    Inputs:
    - overrule_state: current state of the overrule controller (True/False)
    - temperature: current temperature
    - data: system data dictionary containing the thresholds
    """
    if (overrule_state == False) and (temperature < data["temp_min_comfort_threshold"]):
        overrule_state = True

    elif (overrule_state == True) and (temperature > data["temp_OK_threshold"]):
        overrule_state = False

    return overrule_state


def calculate_room_temperature(P, occupancy, prev_temperature, other_room_prev_temp, data, V, outside_temperature):
    """Calculate the new temperature of a room based on the previous temperature, the heating power, occupancy, 
    ventilation, and outdoor temperature.
    Inputs:
    - P: heating power applied to the room (here-and-now decision p1 or p2)
    - occupancy: number of people in the room
    - prev_temperature: previous temperature of the room
    - other_room_prev_temp: previous temperature of the other room
    - data: system characteristics dictionary
    - V: ventilation system status (here-and-now decision v)
    - outside_temperature: current outdoor temperature
    """
    return (
        prev_temperature + 
        data["heat_exchange_coeff"] * (other_room_prev_temp - prev_temperature) -
        data["thermal_loss_coeff"] * (prev_temperature - outside_temperature) +
        data["heating_efficiency_coeff"] * P - 
        data["heat_vent_coeff"] * V + 
        data["heat_occupancy_coeff"] * occupancy
    )


def run_environment(start_day, end_day, plot=False):  
    """Run the environment simulation for a given policy, number of experiments (days) and repetitions.
    Inputs:
    - start_day: starting day index (inclusive)
    - end_day: ending day index (exclusive)
    - plot: whether to plot the results of each experiment (day)
    """
    # Import data
    data = v2_SystemCharacteristics.get_fixed_data()
    occupancy1_matrix = np.genfromtxt("Data/OccupancyRoom1.csv", delimiter=",", skip_header=1)
    occupancy2_matrix = np.genfromtxt("Data/OccupancyRoom2.csv", delimiter=",", skip_header=1)
    price_data        = np.genfromtxt("Data/v2_PriceData.csv",   delimiter=",", skip_header=1)

    # Vector with initial previous value of electricity for each day (t-1)
    initial_previous_prices = price_data[:, 0] # takes only the first column of the price data

    # Matrix with rest of price data
    price_matrix = price_data[:, 1:] # takes all columns except the first one

    NUM_TIMESLOTS     = data["num_timeslots"]
    HEATING_MAX_POWER = data["heating_max_power"]
    VENT_MIN_UP_TIME  = data["vent_min_up_time"] # 3 consecutive hours

    log_objectives = [] # store objective values for all repetitions and experiments (days)
    max_time_logs = [] # logs max time for each day
    avg_time_logs = [] # logs avg time for each day


    outside_temperature_vector = data["outdoor_temperature"]

    # Simulation
    for day in range(start_day, end_day): 
        print(f"Day {day + 1}")
        vent_counter = 0           # consectutive ventilation tracker has to be reset at the beggining of each day
        is_override_room1 = False  # overrule controllers state also has to be reset at the beggining of each day
        is_override_room2 = False
        objective_value = 0        # objective value has to be reset at the beggining of each day

        hourly_time_log = []

        for hour in range(0, NUM_TIMESLOTS):
            #print("Hour: ", hour)
            if hour == 0:
                previous_price    = initial_previous_prices[day]
                temperature_room1 = data["T1"]
                temperature_room2 = data["T2"]
                humidity          = data["H"]

            else:
                previous_price = price_matrix[day][hour-1]
                old_T1, old_T2 = temperature_room1, temperature_room2 # store old temperatures to calculate the new ones based on them (before updating them)
                temperature_room1 = calculate_room_temperature(P1, occupancy1_matrix[day][hour-1], old_T1, old_T2, data, V, outside_temperature_vector[hour-1])
                temperature_room2 = calculate_room_temperature(P2, occupancy2_matrix[day][hour-1], old_T2, old_T1, data, V, outside_temperature_vector[hour-1])
                humidity = (
                    humidity +
                    data["humidity_occupancy_coeff"] * (occupancy1_matrix[day][hour-1] + occupancy2_matrix[day][hour-1]) -
                    data["humidity_vent_coeff"] * V
                )

            # update overrule controlers state
            is_override_room1 = update_overrule_controler_state(is_override_room1, temperature_room1, data)
            is_override_room2 = update_overrule_controler_state(is_override_room2, temperature_room2, data)

            # Update state
            state = {
                "T1": temperature_room1,
                "T2": temperature_room2,
                "H": humidity,
                "Occ1": occupancy1_matrix[day][hour],
                "Occ2": occupancy2_matrix[day][hour],
                "price_t": price_matrix[day][hour],
                "price_previous": previous_price,
                "vent_counter": vent_counter,
                "low_override_r1": is_override_room1,
                "low_override_r2": is_override_room2,
                "current_time": hour
            }

            # Evaluate policy's decision
            POWER_MAX = {1 : HEATING_MAX_POWER, 2 : HEATING_MAX_POWER}

            start_time = time.time()
            decision = Checks.check_and_sanitize_action(POLICY, state, POWER_MAX)
            elapsed_time = time.time() - start_time

            # Update decision variables
            V  = 1 if (humidity > data["humidity_threshold"]) or (0 < vent_counter < VENT_MIN_UP_TIME) else decision["VentilationON"] # if humidity is above the threshold or the min up time is still going, the ventilation is turned ON regardless of the policy's decision. .
            P1 = decision["HeatPowerRoom1"] if not is_override_room1 else HEATING_MAX_POWER # if overrule controller is ON, the heating power is set to the maximum, regardless of the policy's decision
            P2 = decision["HeatPowerRoom2"] if not is_override_room2 else data["heating_max_power"] # if overrule controller is ON, the heating power is set to the maximum, regardless of the policy's decision

            # Update consecutive ventilation usage counter
            if V == 0:
                vent_counter = 0
            else:
                vent_counter += 1

            # Calculate objective function
            objective_value += price_matrix[day][hour] * (V * data["ventilation_power"] + P1 + P2)

            # Save hour log for elapsed time
            hourly_time_log.append(elapsed_time)

            # print(f"Time for {hour}: {elapsed_time:.2f}s", is_override_room1, is_override_room2, vent_counter)


        # Save day logs
        log_objectives.append(float(objective_value))
        max_time_logs.append(max(hourly_time_log))
        avg_time_logs.append(sum(hourly_time_log) / NUM_TIMESLOTS)
        # print(max(hourly_time_log))

    return log_objectives, max_time_logs, avg_time_logs
    
# def run_single_experiment(_):
    # """Helper function for multiprocessing - ignores the input parameter and runs the experiment"""
    # return run_environment(POLICY, N_EXPERIMENTS, False)




if __name__ == "__main__": # executes the following block only if this file is run directly (not imported as a module)
    print("Running policy: ", POLICY.__name__)
    print(f"Distributing {N_EXPERIMENTS} days across {N_WORKERS} workers")

    # Calculate chunk size for each worker
    chunk_size = N_EXPERIMENTS // N_WORKERS
    day_ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(N_WORKERS)]
    
    # Handle any remaining days (if N_EXPERIMENTS not divisible by N_WORKERS)
    if N_EXPERIMENTS % N_WORKERS != 0:
        last_start = day_ranges[-1][1]
        day_ranges[-1] = (day_ranges[-1][0], N_EXPERIMENTS)
    
    print(f"Day ranges for each worker: {day_ranges}")

    # Runs environment in parallel with workers handling different day ranges
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(executor.map(run_environment, [start for start, end in day_ranges], [end for start, end in day_ranges]))

    # Combine results from all workers (each worker processed a chunk of days)
    all_objectives = []
    all_max_times = []
    all_avg_times = []
    
    for worker_results in results:
        worker_objectives, worker_max_times, worker_avg_times = worker_results
        all_objectives.extend(worker_objectives)
        all_max_times.extend(worker_max_times)
        all_avg_times.extend(worker_avg_times)
    
    # Convert to numpy arrays
    all_objectives = np.array(all_objectives)
    all_max_times = np.array(all_max_times)
    all_avg_times = np.array(all_avg_times)
    
    # For single run per day, mean and std are the value and 0
    means = all_objectives
    stds = np.zeros_like(all_objectives)
    
    final_results = np.column_stack((means, stds, all_max_times, all_avg_times))


    # Save file with out overwriting
    base_filename = f"Policy_log_files/{POLICY.__name__}_log_{N_EXPERIMENTS}_exp_{N_REPETITIONS}_reps"
    filename = base_filename + ".csv"

    counter = 1
    while os.path.exists(filename):
        filename = f"{base_filename}_{counter}.csv"
        counter += 1

    np.savetxt(
        filename,
        final_results,
        delimiter=",",
        header="mean,std,max_time,avg_time",
        comments="",
        fmt="%.2f"
    )

    print(f"Results saved to: {filename}")

