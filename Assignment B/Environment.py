from Utils import SystemCharacteristics, Checks
import SP_policy_30
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# Not finished yet


def update_overrule_controler_state(overrule_state, temperature, data):
    if (overrule_state == False) and (temperature < data["temp_min_comfort_threshold"]):
        overrule_state = True

    elif (overrule_state == True) and (temperature > data["temp_OK_threshold"]):
        overrule_state = False

    return overrule_state


def calculate_room_temperature(P, occupancy, prev_temperature, other_room_prev_temp, data, V, outside_temperature):
    return (
        prev_temperature + 
        data["heat_exchange_coeff"] * (other_room_prev_temp - prev_temperature) -
        data["thermal_loss_coeff"] * (prev_temperature - outside_temperature) +
        data["heating_efficiency_coeff"] * P - 
        data["heat_vent_coeff"] * V + 
        data["heat_occupancy_coeff"] * occupancy
    )


def run_environment(policy, n_experiments, n_hours):
    # Import data
    data = SystemCharacteristics.get_fixed_data()
    price_matrix      = np.genfromtxt("Data/PriceData.csv",      delimiter=",", skip_header=1)
    occupancy1_matrix = np.genfromtxt("Data/OccupancyRoom1.csv", delimiter=",", skip_header=1)
    occupancy2_matrix = np.genfromtxt("Data/OccupancyRoom2.csv", delimiter=",", skip_header=1)

    # Initial variables
    vent_counter = 0
    is_override_room1 = False
    is_override_room2 = False

    INITIAL_PREVIOUS_PRICE = 6
    outside_temperature = data["outdoor_temperature"]

    # Simulation
    for day in range(0, n_experiments):
        objective_value = 0

        for hour in range(0, n_hours):
            print("Hour: ", hour)
            if hour == 0:
                previous_price    = INITIAL_PREVIOUS_PRICE
                temperature_room1 = data["initial_temperature"]
                temperature_room2 = data["initial_temperature"]
                humidity          = data["initial_humidity"]

            else:
                previous_price = price_matrix[day][hour-1]
                temperature_room1 = calculate_room_temperature(P1, occupancy1_matrix[day][hour], temperature_room1, temperature_room2, data, V, outside_temperature[hour-1])
                temperature_room2 = calculate_room_temperature(P2, occupancy2_matrix[day][hour], temperature_room2, temperature_room1, data, V, outside_temperature[hour-1])
                humidity = (
                    humidity + 
                    data["humidity_occupancy_coeff"] * (occupancy1_matrix[day][hour] + occupancy2_matrix[day][hour]) - 
                    data["humidity_vent_coeff"] * V
                )


            # update overrule controlers state
            is_override_room1 = update_overrule_controler_state(is_override_room1, temperature_room1, data)
            is_override_room2 = update_overrule_controler_state(is_override_room2, temperature_room2, data)

            #print("Temperatures: ", temperature_room1, temperature_room2)
            #print("Humidity: ", humidity)
            # print("Overrule controlers: ", is_override_room1, is_override_room2)


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


            # Policy takes a decision based on the state
            #decision = policy.select_action(state)
            #print("decision: ", decision)

            # Evaluate policy's decision
            POWER_MAX = {1 : data["heating_max_power"], 2 : data["heating_max_power"]}
            decision = Checks.check_and_sanitize_action(policy, state, POWER_MAX)
            # print(sanitazied_decision)


            # Update decision variables
            V  = 1 if (humidity > data["humidity_threshold"]) or (0 < vent_counter < 3) else decision["VentilationON"] 
            P1 = decision["HeatPowerRoom1"] if not is_override_room1 else data["heating_max_power"]
            P2 = decision["HeatPowerRoom2"] if not is_override_room2 else data["heating_max_power"]


            # Update consecutive ventilation usage counter
            if V == 0:
                vent_counter = 0
            else:
                vent_counter += 1

            print("HeatPowerRoom1", decision["HeatPowerRoom1"], P1)
            print("HeatPowerRoom2", decision["HeatPowerRoom2"], P2)
            print("VentilationON ", decision["VentilationON"], V)

            # Calculate objective function
            objective_value += price_matrix[day][hour] * (V * data["ventilation_power"] + P1 + P2) 
            #print("\n")

        print("Objective: ", objective_value)







# run_environment(Policy_Restaurant, n_experiments=1, n_hours=10)
run_environment(SP_policy_30, n_experiments=100, n_hours=10)

           


