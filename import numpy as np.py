import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. ENVIRONMENT
# ============================================================

START = np.array([1.0, 1.0])
GOAL = np.array([9.0, 9.0])

# Obstacles: (xmin, ymin, xmax, ymax)
OBSTACLES = [
    (3.0, 2.0, 5.0, 7.0),
    (6.0, 5.0, 8.5, 7.0),
    (6.0, 1.0, 7.5, 3.5)
]

WORLD_MIN = 0
WORLD_MAX = 10


# ============================================================
# 2. PSO PARAMETERS
# ============================================================

NUM_PARTICLES = 50
NUM_WAYPOINTS = 8
MAX_ITERATIONS = 200

# PSO coefficients
w = 0.7
c1 = 1.5
c2 = 1.5

DIMENSIONS = NUM_WAYPOINTS * 2


# ============================================================
# 3. CHECK WHETHER A POINT IS INSIDE AN OBSTACLE
# ============================================================

def point_in_obstacle(point):

    x, y = point

    for xmin, ymin, xmax, ymax in OBSTACLES:

        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True

    return False


# ============================================================
# 4. CHECK WHETHER A LINE SEGMENT COLLIDES WITH AN OBSTACLE
# ============================================================

def line_hits_obstacle(p1, p2):

    # Sample points along the line
    for t in np.linspace(0, 1, 50):

        point = p1 + t * (p2 - p1)

        if point_in_obstacle(point):
            return True

    return False


# ============================================================
# 5. FITNESS FUNCTION
# ============================================================

def fitness(position):

    # Convert particle into waypoint coordinates
    waypoints = position.reshape(NUM_WAYPOINTS, 2)

    # Complete path:
    # START → waypoints → GOAL
    path = np.vstack([
        START,
        waypoints,
        GOAL
    ])

    total_distance = 0
    collision_penalty = 0

    # Calculate distance and collisions
    for i in range(len(path) - 1):

        p1 = path[i]
        p2 = path[i + 1]

        # Distance between two points
        distance = np.linalg.norm(p2 - p1)

        total_distance += distance

        # Collision detection
        if line_hits_obstacle(p1, p2):

            collision_penalty += 1000

    # Final fitness
    return total_distance + collision_penalty


# ============================================================
# 6. INITIALIZE PARTICLES
# ============================================================

positions = np.random.uniform(
    WORLD_MIN,
    WORLD_MAX,
    size=(NUM_PARTICLES, DIMENSIONS)
)

velocities = np.random.uniform(
    -1,
    1,
    size=(NUM_PARTICLES, DIMENSIONS)
)


# ============================================================
# 7. INITIAL FITNESS
# ============================================================

fitness_values = np.array([
    fitness(position)
    for position in positions
])


# ============================================================
# 8. INITIALIZE PERSONAL BEST
# ============================================================

personal_best_positions = positions.copy()

personal_best_values = fitness_values.copy()


# ============================================================
# 9. INITIALIZE GLOBAL BEST
# ============================================================

best_index = np.argmin(personal_best_values)

global_best_position = (
    personal_best_positions[best_index].copy()
)

global_best_value = (
    personal_best_values[best_index]
)


# ============================================================
# 10. PSO ITERATION
# ============================================================

convergence = []

for iteration in range(MAX_ITERATIONS):

    for i in range(NUM_PARTICLES):

        # Random values
        r1 = np.random.random(DIMENSIONS)
        r2 = np.random.random(DIMENSIONS)

        # ----------------------------------------------------
        # VELOCITY UPDATE
        # ----------------------------------------------------

        velocities[i] = (

            w * velocities[i]

            + c1 * r1 *
            (personal_best_positions[i] - positions[i])

            + c2 * r2 *
            (global_best_position - positions[i])
        )

        # ----------------------------------------------------
        # POSITION UPDATE
        # ----------------------------------------------------

        positions[i] = (
            positions[i] + velocities[i]
        )

        # Keep particles inside environment
        positions[i] = np.clip(
            positions[i],
            WORLD_MIN,
            WORLD_MAX
        )

        # ----------------------------------------------------
        # CALCULATE FITNESS
        # ----------------------------------------------------

        current_fitness = fitness(
            positions[i]
        )

        # ----------------------------------------------------
        # UPDATE PERSONAL BEST
        # ----------------------------------------------------

        if current_fitness < personal_best_values[i]:

            personal_best_values[i] = current_fitness

            personal_best_positions[i] = (
                positions[i].copy()
            )

        # ----------------------------------------------------
        # UPDATE GLOBAL BEST
        # ----------------------------------------------------

        if current_fitness < global_best_value:

            global_best_value = current_fitness

            global_best_position = (
                positions[i].copy()
            )

    # Store best result
    convergence.append(global_best_value)

    print(
        f"Iteration {iteration + 1:3d} | "
        f"Best Fitness = {global_best_value:.3f}"
    )


# ============================================================
# 11. EXTRACT BEST PATH
# ============================================================

best_waypoints = global_best_position.reshape(
    NUM_WAYPOINTS,
    2
)

best_path = np.vstack([
    START,
    best_waypoints,
    GOAL
])


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\n==============================")
print("OPTIMIZATION COMPLETE")
print("==============================")

print("Best Fitness:", global_best_value)

print("\nBest Waypoints:")

for i, point in enumerate(best_waypoints):

    print(
        f"Waypoint {i + 1}: "
        f"({point[0]:.2f}, {point[1]:.2f})"
    )


# ============================================================
# 13. PLOT FINAL ROBOT PATH
# ============================================================

fig, ax = plt.subplots(figsize=(9, 9))


# Draw obstacles
for xmin, ymin, xmax, ymax in OBSTACLES:

    width = xmax - xmin
    height = ymax - ymin

    rectangle = plt.Rectangle(
        (xmin, ymin),
        width,
        height,
        color="red",
        alpha=0.6
    )

    ax.add_patch(rectangle)


# Draw path
ax.plot(
    best_path[:, 0],
    best_path[:, 1],
    "b-o",
    linewidth=2,
    markersize=5,
    label="PSO Optimized Path"
)


# Start point
ax.scatter(
    START[0],
    START[1],
    color="green",
    s=200,
    marker="o",
    label="START",
    zorder=5
)


# Goal point
ax.scatter(
    GOAL[0],
    GOAL[1],
    color="blue",
    s=200,
    marker="*",
    label="GOAL",
    zorder=5
)


# Waypoint labels
for i, point in enumerate(best_waypoints):

    ax.text(
        point[0] + 0.1,
        point[1] + 0.1,
        f"W{i + 1}",
        fontsize=9
    )


ax.set_xlim(WORLD_MIN, WORLD_MAX)
ax.set_ylim(WORLD_MIN, WORLD_MAX)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

ax.set_title(
    "Mobile Robot Path Planning Using PSO"
)

ax.grid(True)
ax.legend()

plt.show()


# ============================================================
# 14. PLOT CONVERGENCE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    convergence,
    color="blue",
    linewidth=2
)

plt.xlabel("Iteration")
plt.ylabel("Best Fitness")

plt.title(
    "PSO Convergence Curve"
)

plt.grid(True)

plt.show()