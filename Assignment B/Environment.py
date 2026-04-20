from Utils import v2_SystemCharacteristics, Checks
from Policies import SP_policy_30
import numpy as np
import matplotlib.pyplot as plt
import warnings
from Utils.v2_SystemCharacteristics import get_fixed_data
warnings.filterwarnings("ignore")

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

def _plot_experiment(rep, day, day_log, price_series=None, temp_thresholds=None, humidity_threshold=None):
    """Plot (for an experiment=day) two-room view with power vs temperature (twin axes) and thresholds.
    Inputs: 
    - rep: replication index (for title)
    - day: day index (for title)
    - day_log: dictionary containing the logged data for the day (hour, P1, P2, T1, T2, H, V)
    - price_series: series of electricity prices to plot on the lower graph
    - temp_thresholds: tuple of (min_comfort_threshold, OK_threshold) to plot as horizontal lines on the temperature graphs
    - humidity_threshold: humidity threshold to plot as a horizontal line on the humidity graph
    """
    hours = day_log["hour"]
    P1 = day_log["P1"] # here-and-now decision p1 (uppercase because it's a constant for the environment)
    P2 = day_log["P2"] # here-and-now decision p2 (uppercase because it's a constant for the environment)
    T1 = day_log["T1"]
    T2 = day_log["T2"]
    H = day_log["H"]
    V = day_log["V"]   # here-and-now decision v (uppercase because it's a constant for the environment)

    fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    fig.suptitle(f"Replication {rep+1} - Day {day+1}")

    # Room 1: Power and temperature
    ax1 = axs[0]
    ax1.plot(hours, P1, '-o', color='red', label='Heating Power R1')
    ax1.set_ylabel('Power R1 [kW]', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)

    ax1b = ax1.twinx()
    ax1b.plot(hours, T1, '-s', color='green', label='Temp R1')
    ax1b.set_ylabel('Temp R1 [°C]', color='green')
    ax1b.tick_params(axis='y', labelcolor='green')
    if temp_thresholds is not None:
        min_thr, ok_thr = temp_thresholds
        ax1b.axhline(min_thr, color='cyan', linestyle='--', label='Min comfort temp')
        ax1b.axhline(ok_thr, color='magenta', linestyle='--', label='OK temp')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(handles1 + handles1b, labels1 + labels1b, loc='upper right')
    ax1.set_title('Room 1: Power and Temperature')

    # Room 2: Power and temperature
    ax2 = axs[1]
    ax2.plot(hours, P2, '-o', color='red', label='Heating Power R2')
    ax2.set_ylabel('Power R2 [kW]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True)

    ax2b = ax2.twinx()
    ax2b.plot(hours, T2, '-s', color='green', label='Temp R2')
    ax2b.set_ylabel('Temp R2 [°C]', color='green')
    ax2b.tick_params(axis='y', labelcolor='green')
    if temp_thresholds is not None:
        min_thr, ok_thr = temp_thresholds
        ax2b.axhline(min_thr, color='cyan', linestyle='--', label='Min comfort temp')
        ax2b.axhline(ok_thr, color='magenta', linestyle='--', label='OK temp')

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(handles2 + handles2b, labels2 + labels2b, loc='upper right')
    ax2.set_title('Room 2: Power and Temperature')

    # Lower plot: humidity, ventilation and electricity price
    ax3 = axs[2]
    ax3.plot(hours, H, '-o', color='green', label='Humidity')
    ax3.set_ylabel('Humidity [%]', color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.grid(True)
    if humidity_threshold is not None:
        ax3.axhline(humidity_threshold, color='gray', linestyle='--', label='Humidity Threshold')

    if price_series is not None:
        ax3b = ax3.twinx()
        ax3b.plot(hours, price_series, '-o', color='blue', label='Electricity Price')
        ax3b.set_ylabel('Price [€/kWh]', color='blue')
        ax3b.tick_params(axis='y', labelcolor='blue')

        # Add ventilation to the same twin axis or separate
        ax3c = ax3.twinx()
        ax3c.plot(hours, V, '-s', color='red', label='Ventilation')
        ax3c.set_ylabel('Ventilation', color='red')
        ax3c.tick_params(axis='y', labelcolor='red')
        ax3c.set_ylim(-0.1, 1.1)
        # Position ax3c to the right of ax3b
        ax3c.spines["right"].set_position(("axes", 1.1))

    handles3, labels3 = ax3.get_legend_handles_labels()
    if price_series is not None:
        handles3b, labels3b = ax3b.get_legend_handles_labels()
        handles3c, labels3c = ax3c.get_legend_handles_labels()
        ax3.legend(handles3 + handles3b + handles3c, labels3 + labels3b + labels3c, loc='upper right')
    else:
        ax3.legend(handles3, labels3, loc='upper right')

    ax3.set_xlabel('Hour')
    ax3.set_title('Humidity, Ventilation and Electricity Price')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def run_environment(policy, n_experiments=1, n_repetitions=1, plot=False):  
    """Run the environment simulation for a given policy, number of experiments (days) and repetitions.
    Inputs:
    - policy: function that takes the current state and returns a decision dictionary with keys "HeatPowerRoom1", "HeatPowerRoom2", "VentilationON"
    - n_experiments: number of days to simulate
    - n_repetitions: number of times to repeat the whole set of experiments (for statistical significance)
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

    all_objectives = [] # store objective values for all repetitions and experiments (days)
    all_logs = [] # store detailed logs for all repetitions and experiments (days)

    for rep in range(n_repetitions):
        outside_temperature_vector = data["outdoor_temperature"]
        objective_value_record = []

        # Simulation
        for day in range(0, n_experiments): 
            vent_counter = 0           # consectutive ventilation tracker has to be reset at the beggining of each day (false=OFF, true=ON)
            is_override_room1 = False  # overrule controllers state also has to be reset at the beggining of each day (false=OFF, true=ON)
            is_override_room2 = False
            objective_value = 0        # objective value has to be reset at the beggining of each day

            day_log = {
                "hour": [],
                "V": [],
                "P1": [],
                "P2": [],
                "T1": [],
                "T2": [],
                "H": []
            }

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
                decision = Checks.check_and_sanitize_action(policy, state, POWER_MAX)
                # Update decision variables
                V  = 1 if (humidity > data["humidity_threshold"]) or (0 < vent_counter < VENT_MIN_UP_TIME) else decision["VentilationON"] # if humidity is above the threshold or the min up time is still going, the ventilation is turned ON regardless of the policy's decision. .
                P1 = decision["HeatPowerRoom1"] if not is_override_room1 else HEATING_MAX_POWER # if overrule controller is ON, the heating power is set to the maximum, regardless of the policy's decision
                P2 = decision["HeatPowerRoom2"] if not is_override_room2 else data["heating_max_power"] # if overrule controller is ON, the heating power is set to the maximum, regardless of the policy's decision
                # High-temp overrule: force heating OFF if temperature is at or above max threshold
                if temperature_room1 >= data["temp_max_comfort_threshold"]:
                    P1 = 0
                if temperature_room2 >= data["temp_max_comfort_threshold"]:
                    P2 = 0

                # Update consecutive ventilation usage counter
                if V == 0:
                    vent_counter = 0
                else:
                    vent_counter += 1

                # Calculate objective function
                objective_value += price_matrix[day][hour] * (V * data["ventilation_power"] + P1 + P2)

                # Store logs
                day_log["hour"].append(hour)
                day_log["V"].append(V)
                day_log["P1"].append(P1)
                day_log["P2"].append(P2)
                day_log["T1"].append(temperature_room1)
                day_log["T2"].append(temperature_room2)
                day_log["H"].append(humidity)

            objective_value_record.append(objective_value)
            all_logs.append({"rep": rep, "day": day, "log": day_log})

            if plot:
                temp_thresholds = (
                    data["temp_min_comfort_threshold"],
                    data["temp_OK_threshold"]
                )
                humidity_threshold = data.get("humidity_threshold", None)
                _plot_experiment(
                    rep,
                    day,
                    day_log,
                    price_series=price_matrix[day][:NUM_TIMESLOTS],
                    temp_thresholds=temp_thresholds,
                    humidity_threshold=humidity_threshold
                )

        all_objectives.append(objective_value_record)

    return {
        "objectives": all_objectives,
        "logs": all_logs
    }


if __name__ == "__main__": # executes the following block only if this file is run directly (not imported as a module)
    data = v2_SystemCharacteristics.get_fixed_data()
    price_data = np.genfromtxt("Data/v2_PriceData.csv", delimiter=",", skip_header=1)
    price_matrix = price_data[:, 1:] 
    results = run_environment(
        SP_policy_30, # policy currently under evaluation
        n_experiments=2, # number of days to simulate
        n_repetitions=2, # number of repetitions of the whole set of experiments
        plot=True # include plots for each experiment (day)
    )
    # Print hourly costs for day 1, rep 1
    log = results["logs"][0]["log"]
    print("Day 1 - Hourly costs:")
    total = 0
    for h in range(len(log["hour"])):
        cost = price_matrix[0][h] * (log["V"][h] * data["ventilation_power"] + log["P1"][h] + log["P2"][h])
        total += cost
        print(f"  Hour {h}: {cost:.2f}")
    print(f"  Total: {total:.2f}")
    all_objectives  = np.array(results["objectives"]) 
    mean_objectives = np.mean(all_objectives, axis=0)
    std_objectives  = np.std(all_objectives, axis=0)

    data = get_fixed_data()
    T_out = data['outdoor_temperature']  

    t = 5
    print(T_out[t])

    # plt.figure(figsize=(10, 5))
    # plt.errorbar(range(1, len(mean_objectives) + 1), mean_objectives, yerr=std_objectives, fmt='-o', capsize=5)
    # plt.title('Objective Value Across Experiments')
    # plt.xlabel('Experimento')
    # plt.ylabel('Objective Value')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()