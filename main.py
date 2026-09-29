import pypsa
import pandas as pd

# 1. Initialize Power Network & Time Snapshots (24 Hours)
network = pypsa.Network()
hours = pd.date_range("2026-01-01 00:00", "2026-01-01 23:00", freq="h")
network.set_snapshots(hours)

# 2. Add Electrical Bus & System Demand
network.add("Bus", "Regional_Grid")
network.add("Load", "System_Demand", bus="Regional_Grid", p_set=150.0)

# 3. Add Candidate Technologies (Extendable Capacities for Optimization)
network.add(
    "Generator", 
    "Solar_PV", 
    bus="Regional_Grid", 
    p_nom_extendable=True, 
    capital_cost=45000,
    marginal_cost=0
)

network.add(
    "Generator", 
    "Wind_Turbine", 
    bus="Regional_Grid", 
    p_nom_extendable=True, 
    capital_cost=70000, 
    marginal_cost=0
)

network.add(
    "Generator", 
    "Gas_Peaker", 
    bus="Regional_Grid", 
    p_nom_extendable=True, 
    capital_cost=20000, 
    marginal_cost=65
)

# 4. Run Optimization Solver (HiGHS)
print("--- Solving Capacity Expansion Linear Program (HiGHS) ---")
network.optimize(solver_name="highs")

# 5. Extract Results
print("\n=== OPTIMAL INSTALLED CAPACITY (MW) ===")
print(network.generators.p_nom_opt)

print("\n=== TOTAL GENERATION DISPATCH (MWh) ===")
print(network.generators_t.p.sum())
