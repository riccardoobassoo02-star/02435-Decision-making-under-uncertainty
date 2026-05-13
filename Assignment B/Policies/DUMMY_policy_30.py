### DUMMY POLICY
# never turns on the ventilation nor any heater, and leaves everything up to the overrule controllers

def select_action(state):    
    HereAndNowActions = {
    "HeatPowerRoom1" : 0,
    "HeatPowerRoom2" : 0,
    "VentilationON" : 0
    }
    return HereAndNowActions