"# 02435-Decision-making-under-uncertainty" 
Part A 
Task 1 
Division of work:
-Gerard: overleaf + have a look at the constraints
-Kris: overleaf + have a look at the constraints
-Esteban: Python code
-Riccardo: Python code 
-----------------------------------------------------------------------------------------------------------------------------
STATUS OF WORK 
- check the constraints, especially the utility of the initial conditions 
    -Initial conditions: we open the restaurant at t=0. T0 is given as 21°, but imagine we have an extreme T0>>>>T_High. Then, the controllers should have the possibility to activate at t=0. Therefore, the initial conditions for the controllers are incorrect.Plus, the activation and deactivation constraints already handle t=0.
    Logic: we open the restaurant, as we open the doors to the customers, we also want the controllers to be able to activate, independently on what we know about the temperature.


    -Conflict Management: 
    Code:
    # #  TO BE CHECKED 5.11 Conflict resolution between Low and High Temperature Overrule Controllers: both cannot be activated at the same time 
    # model.conflict_res = ConstraintList()
    # for r in model.R:
    #     for t in model.T:
    #         model.conflict_res.add(model.delta_low[r,t] + model.delta_high[r,t] <= 1)  

    it's redundant because the deactivation constraints already make it physically impossible for both to be active simultaneously

-write the mathematical model on overleaf 


- write MDP model for task 2



