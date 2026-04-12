from src.solver import value_iteration

print("Running Value Iteration...")

V, policy = value_iteration()

print("\nSample of Optimal Policy:\n")

# print only a few states (not all)
for i, (state, action) in enumerate(policy.items()):
    print(state, "->", action)
    if i > 10:
        break