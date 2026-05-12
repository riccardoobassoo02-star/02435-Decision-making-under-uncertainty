from pyomo.environ import *

# 1. Initialize Model
model = ConcreteModel()

# 2. Define Variables
model.x = Var(initialize=0)
model.y = Var(initialize=0)

# 3. Define Objective
model.obj = Objective(expr=model.x**2 + model.y**2, sense=minimize)

# 4. Define Constraints
model.con = Constraint(expr=model.x + model.y == 10)

# 5. IMPORTANT: Declare the Suffix to capture duals
model.dual = Suffix(direction=Suffix.IMPORT)

# 6. Solve (Using Ipopt as it handles non-linear duals well)
solver = SolverFactory('gurobi')
results = solver.solve(model)

# 7. Access the Lagrange Multiplier
lagrange_multiplier = model.dual[model.con]

print(f"Optimal x: {value(model.x)}")
print(f"Optimal y: {value(model.y)}")
print(f"Lagrange Multiplier (Dual) of constraint: {lagrange_multiplier}")