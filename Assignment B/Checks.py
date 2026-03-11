# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 11:08:56 2025

@author: geots
"""

# ------------------------------------------------------------
# This code checks timing, feasibility, and clipping
# Students should test their policy using this.
# EXAMPLE USE AT THE BOTTOM
# ------------------------------------------------------------

import time
import numpy as np

# ------------------------------------------------------------
# Dummy safe action
# ------------------------------------------------------------
DUMMY_ACTION = {"p1": 0.0, "p2": 0.0, "v": 0}

def check_and_sanitize_action(policy, state, PowerMax):
    """
    Performs the following checks (as required by the assignment):
      1. Times the policy execution.
         If too long → ignore output and return dummy action.
      2. Ensures the policy returns valid numeric values.
      3. Clips the actions to the feasible bounds in PowerMax.
      4. Maps ventilation value to {0,1}.
      5. If anything fails → return dummy action.

    Inputs:
      - policy: object with .select_action(state)
      - state: dictionary
      - PowerMax: dictionary like {1: max_room1, 2: max_room2}

    Returns:
      A sanitized action dictionary {"p1": float, "p2": float, "v": 0 or 1}
    """

    # ---------------------------------------
    # 1. Ask the policy & time it
    # ---------------------------------------
    t0 = time.time()
    try:
        action = policy.select_action(state)
        elapsed = time.time() - t0

        # If policy is too slow → dummy
        if elapsed > 15.0:
            print(f"[WARNING] Policy too slow ({elapsed:.2f}s). Using dummy action.")
            return DUMMY_ACTION.copy()

    except Exception as e:
        print(f"[WARNING] Policy crashed: {e}. Using dummy action.")
        return DUMMY_ACTION.copy()

    

    # ---------------------------------------
    # 3. Clip actions to feasible bounds
    # ---------------------------------------
    # ---------------------------------------
    # 2. Clip to feasible set (or fail → dummy)
    # ---------------------------------------
    try:
        action["p1"] = float(np.clip(action["p1"], 0, PowerMax[1]))
        action["p2"] = float(np.clip(action["p2"], 0, PowerMax[2]))
    
        # ventilation: threshold to {0,1}
        action["v"] = int(float(action["v"]) > 0.5)

    except Exception as e:
        print(f"[WARNING] Action clipping failed: {e}. Using dummy action.")
        return DUMMY_ACTION.copy()

    # ---------------------------------------
    # Return sanitized action
    # ---------------------------------------
    return {"p1": action["p1"], "p2": action["p2"], "v": action["v"]}


### Example use

class MyPolicy:
    def select_action(self, state):
        return {"p1": 50, "p2": -330, "v": 'something_crazy'}

policy = MyPolicy()

state = {"T1": 21}
PowerMax = {1: 3.0, 2: 3.0}

action = check_and_sanitize_action(policy, state, PowerMax)
print(action)
