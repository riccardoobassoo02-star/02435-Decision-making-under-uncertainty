from Utils.v2_SystemCharacteristics import get_fixed_data

# Parameters extraction from system characteristics
data = get_fixed_data()
HEATING_MAX_POWER = data["heating_max_power"]
VENT_MIN_UP_TIME  = data["vent_min_up_time"]


def select_action(state):    
    # Override controllers state
    low_override_r1 = state["low_override_r1"]
    low_override_r2 = state["low_override_r2"]

    # current humidity and consecutive ventilation counter
    humidity = state["H"]
    vent_counter = state["vent_counter"]


    # Never turns on the ventilation nor any heater, unless is forced by the overrule controllers
    p1 = (0 if not low_override_r1
            else HEATING_MAX_POWER
    )

    p2 = (0 if not low_override_r2
            else HEATING_MAX_POWER
    )

    v = (
        1 if (humidity > data["humidity_threshold"]) or (0 < vent_counter < VENT_MIN_UP_TIME)
        else 0
    )


    HereAndNowActions = {
    "HeatPowerRoom1" : p1,
    "HeatPowerRoom2" : p2,
    "VentilationON" : v
    }
    return HereAndNowActions