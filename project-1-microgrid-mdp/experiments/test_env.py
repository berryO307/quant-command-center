from src.environment import MDPEnvironment, State

# Initial state
initial_state = State(
    battery_level=2,
    price_state="low",
    generation_state="low"
)

# Create environment
env = MDPEnvironment(initial_state)

state = env.reset(initial_state)

print("Initial state:", state)

# Run simulation
for step in range(10):
    action = 0  # hold (you can change later)

    next_state, reward = env.step(action)

    print(f"Step {step+1}")
    print("State:", next_state)
    print("Reward:", reward)
    print("-" * 30)