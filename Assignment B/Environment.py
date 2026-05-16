from Utils import v2_SystemCharacteristics, v2_Checks, plotting
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")

## IMPORTANT: THINGS TO CHANGE BEFORE SUBMISSION:
# -  CHANGE DATA PATH, SO THE DATA IS IN THE SAME DIRECTORY THAN THE ENVIRONMENT
# -  CHANGE PATHS OF ALL IMPORTS (EVERYTHING IN THE SAME FOLDER)
# -  CHANGE OUTPUT OF ENVIRONMENT TO RETURN JUST THE AVG OBJECTIVE VALUE
# - CHANGE INPUT OF ENVIRONMENT TO JUST TAKE THE POLICY AND NUMBER OF EXPERIMENTS, NOT START AND END DAYS 

DATA_DIRECTORY = "Data/"

data = v2_SystemCharacteristics.get_fixed_data()

# Constants
NUM_TIMESLOTS     = data["num_timeslots"]
HEATING_MAX_POWER = data["heating_max_power"]
VENT_MIN_UP_TIME  = data["vent_min_up_time"]
zeta_exch         = data['heat_exchange_coeff'] # heat exchange coefficient between rooms
zeta_conv         = data['heating_efficiency_coeff'] # heating efficiency: increase in room temperature per kW of heating power
zeta_loss         = data['thermal_loss_coeff'] # thermal loss coefficient: fraction of indoor-outdoor temperature difference lost per hour    
zeta_cool         = data['heat_vent_coeff'] # ventilation cooling effect: temperature decrease in the room for each hour that ventilation is ON (°C)
zeta_occ          = data['heat_occupancy_coeff'] # occupancy heat gain: temperature increase per hour per person in the room (°C)
eta_occ           = data['humidity_occupancy_coeff'] # humidity increase per hour per person in the room (%)
eta_vent    = data['humidity_vent_coeff'] # humidity decrease per hour when ventilation is ON (%)


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

    elif (overrule_state == True) and (temperature >= data["temp_OK_threshold"]):
        overrule_state = False
    
    return overrule_state


def calculate_room_temperature(P, occupancy, prev_temperature, other_room_prev_temp, V, outside_temperature):
    """Calculate the new temperature of a room based on the previous temperature, the heating power, occupancy, 
    ventilation, and outdoor temperature.
    Inputs:
    - P: heating power applied to the room (here-and-now decision p1 or p2)
    - occupancy: number of people in the room
    - prev_temperature: previous temperature of the room
    - other_room_prev_temp: previous temperature of the other room
    - V: ventilation system status (here-and-now decision v)
    - outside_temperature: current outdoor temperature
    """
    # print(f"occupancy: {occupancy}, outside temperature: {outside_temperature}")

    return (
        prev_temperature + 
        zeta_exch * (other_room_prev_temp - prev_temperature) -
        zeta_loss * (prev_temperature - outside_temperature) +
        zeta_conv * P - 
        zeta_cool * V + 
        zeta_occ * occupancy
    )


def verify_ventilation_actions(decision_V, data, humidity, vent_counter, VENT_MIN_UP_TIME):
    """
    Check if the ventilation action complies with the humidity threshold and minimum up time requirement.
    If not, terminate the program with a clear message.
    """
    # The ventilation must be turned on if the humidity is above the threshold, or if the minimum up time has not been met.
    V = (
        1
        if (humidity > data["humidity_threshold"])
        or (0 < vent_counter < VENT_MIN_UP_TIME)
        else decision_V
    )

    # If the policy's decision does not comply with the required action, terminate the program with an error message
    if decision_V != V:
        reason = 'high humidity detected' if humidity > data['humidity_threshold'] else 'minimum ventilation uptime not met'
        print(f"\nERROR: ILLEGAL ACTION TERMINATED THE SIMULATION:")
        print("Ventilation was set to OFF, but the safety system has determined that it must be ON due to the following reason:")
        print(reason)
        sys.exit(1)


def verify_heater_actions(decision_P, data, temperature_room, is_override_room, HEATING_MAX_POWER):
    """
    Check if the heating power actions comply with the overrule controller states and maximum comfort temperature.
    If not, terminate the program with a clear message.
    """
    # If the overrule controller is active, the heating power must be set to the maximum
    P = (
        decision_P
        if not is_override_room
        else HEATING_MAX_POWER
    )

    # If the temperature is above the maximum comfort threshold, the heating power must be 0 
    if temperature_room >= data["temp_max_comfort_threshold"]:
        P = 0

    # If the policy's decision does not comply with the required actions, terminate the program with an error message
    if decision_P != P:
        reason = 'emergency override active' if is_override_room else 'max comfort temperature exceeded'
        print(f"\nERROR: ILLEGAL ACTION TERMINATED THE SIMULATION:")
        print(f"Heating power for the room was set to {decision_P} W, but the safety system has determined that it must be adjusted to {P} W due to the following reason:")
        print(reason)
        sys.exit(1)
    


def run_environment(policy, start, end, plot=False):  
    """
    Run the environment simulation for a given policy.
    Prints the running average daily cost during the simulation.
    """
    # Import OIH results for comparison
    oih_daily_costs = np.genfromtxt("results/OIH_daily_costs.csv", delimiter=",")

    # Import data
    data = v2_SystemCharacteristics.get_fixed_data()
    occupancy1_matrix = np.genfromtxt(DATA_DIRECTORY + "OccupancyRoom1.csv", delimiter=",", skip_header=1)
    occupancy2_matrix = np.genfromtxt(DATA_DIRECTORY + "OccupancyRoom2.csv", delimiter=",", skip_header=1)
    price_data        = np.genfromtxt(DATA_DIRECTORY + "v2_PriceData.csv"  , delimiter=",", skip_header=1)

    # Vector with initial values of price data
    initial_previous_prices = price_data[:, 0]

    # Matrix with rest of price data
    price_matrix = price_data[:, 1:]

    # Outside temperature vector
    outside_temperature_vector = data["outdoor_temperature"]

    # Logs and results storage
    daily_objective_values = []
    daily_logs = [] 
         
    # Simulation
    for day in range(start, end): 
        vent_counter = 0
        is_override_room1 = False
        is_override_room2 = False
        objective_value = 0

        day_log = {
            "hour": [],
            "V": [],
            "P1": [],
            "P2": [],
            "T1": [],
            "T2": [],
            "H": [],
            "objective": [],
            "price": [],
            "occ1": [],
            "occ2": []
        }

        for hour in range(NUM_TIMESLOTS):
            if hour == 0:
                previous_price = initial_previous_prices[day]
                temperature_room1 = data["T1"]
                temperature_room2 = data["T2"]
                humidity = data["H"]

            else:
                previous_price = price_matrix[day][hour - 1]

                old_T1 = temperature_room1
                old_T2 = temperature_room2

                temperature_room1 = calculate_room_temperature(
                    P1,
                    occupancy1_matrix[day][hour - 1],
                    old_T1,
                    old_T2,
                    V,
                    outside_temperature_vector[hour - 1]
                )

                temperature_room2 = calculate_room_temperature(
                    P2,
                    occupancy2_matrix[day][hour - 1],
                    old_T2,
                    old_T1,
                    V,
                    outside_temperature_vector[hour - 1]
                )

                humidity = (
                    humidity
                    + eta_occ *
                    (
                        occupancy1_matrix[day][hour - 1]
                        + occupancy2_matrix[day][hour - 1]
                    )
                    - eta_vent * V
                )

                # Update consecutive ventilation counter
                if V == 0:
                    vent_counter = 0
                else:
                    vent_counter += 1

     

            # Update overrule controllers state
            is_override_room1 = update_overrule_controler_state(
                is_override_room1,
                temperature_room1,
                data
            )

            is_override_room2 = update_overrule_controler_state(
                is_override_room2,
                temperature_room2,
                data
            )

            # Update state
            state = {
                "T1"             : temperature_room1,
                "T2"             : temperature_room2,
                "H"              : humidity,
                "Occ1"           : occupancy1_matrix[day][hour],
                "Occ2"           : occupancy2_matrix[day][hour],
                "price_t"        : price_matrix[day][hour],
                "price_previous" : previous_price,
                "vent_counter"   : vent_counter,
                "low_override_r1": is_override_room1,
                "low_override_r2": is_override_room2,
                "current_time"   : hour
            }


            # Evaluate policy's decisions
            POWER_MAX = {1: HEATING_MAX_POWER, 2: HEATING_MAX_POWER}
            decision = v2_Checks.check_and_sanitize_action(policy, state, POWER_MAX)

            # Extract actions
            V  = decision["VentilationON"]
            P1 = decision["HeatPowerRoom1"]
            P2 = decision["HeatPowerRoom2"]

            # Terminates the program if the policy's decisions violate overrule controlers
            verify_ventilation_actions(V, data, humidity, vent_counter, VENT_MIN_UP_TIME)
            verify_heater_actions(P1, data, temperature_room1, is_override_room1, HEATING_MAX_POWER)
            verify_heater_actions(P2, data, temperature_room2, is_override_room2, HEATING_MAX_POWER)

            # Calculate objective function (electricity cost)
            hourly_cost = price_matrix[day][hour] * (
                V * data["ventilation_power"] + P1 + P2
            )
            objective_value += hourly_cost

            # Save hourly logs
            day_log["hour"].append(hour)
            day_log["V"].append(V)
            day_log["P1"].append(P1)
            day_log["P2"].append(P2)
            day_log["T1"].append(temperature_room1)
            day_log["T2"].append(temperature_room2)
            day_log["H"].append(humidity)
            day_log["objective"].append(hourly_cost)
            day_log["price"].append(state["price_t"])
            day_log["occ1"].append(state["Occ1"])
            day_log["occ2"].append(state["Occ2"])


        # Collect daily logs
        daily_objective_values.append(objective_value)
        daily_logs.append({"day": day, "log": day_log})


        # Show daily results and plots
        running_mean = np.mean(daily_objective_values)

        # Deviation from OIH cost in percentage (How much margin of improvement is left compared to OIH, in percentage)
        deviation = ((round(objective_value, 2) - round(oih_daily_costs[day], 2))/round(oih_daily_costs[day], 2))*100


        print(
            f"Day {day + 1:>3}: "
            f"daily cost = {objective_value:>7.2f}"
            f" | Improvement margin = {deviation:>7.2f}%"
            f" | OIH daily cost = {oih_daily_costs[day]:>7.2f}"
            f" | running average = {running_mean:>7.2f}",
            flush=True
        )

        if round(objective_value, 2) < round(oih_daily_costs[day], 2):
            print("\nSomething went wrong, the policy's daily cost is below the OIH cost, which should be impossible")
            sys.exit(1)

        if plot:
            temp_thresholds = (
                data["temp_min_comfort_threshold"],
                data["temp_OK_threshold"]
            )

            humidity_threshold = data.get("humidity_threshold", None)

            plotting.plot_experiment(
                day,
                day_log,
                price_series=price_matrix[day][:NUM_TIMESLOTS],
                temp_thresholds=temp_thresholds,
                humidity_threshold=humidity_threshold
            )

    
    # Calculate average objective value across experiments
    avg_objective_value = np.mean(daily_objective_values)

    return avg_objective_value, {
        "objectives": daily_objective_values,
        "logs": daily_logs
    }


