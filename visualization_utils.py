import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon

matplotlib.use("TkAgg")

def setup_sensors(ax, sensors):

    sensor_dots = {}
    sensor_text = {}
    sensor_id_text = {}
    color = "gray"  # Use gray for all sensors.
    for i, s in enumerate(sensors):
        _, sensor_id, _, sx, sy, sz = s
        label = "Sensor" if i == 0 else ""  # Only label the first sensor.
        dot, = ax.plot(sx, sy, 'o', color=color, markersize=8, label=label)
        sensor_dots[sensor_id] = dot
        sensor_text[sensor_id] = ax.text(sx, sy + 100, f"z: {sz:.1f}",
                                         fontsize=9, color="black", ha='center')
        sensor_id_text[sensor_id] = ax.text(sx, sy - 200, f"ID: {sensor_id}",
                                            fontsize=9, color="black", ha='center')
    return sensor_dots, sensor_text, sensor_id_text

def setup_mountains(ax, db, exp_id):

    mountains = db.get_mountains_by_experiment(exp_id)
    mountain_markers = []
    for i, m in enumerate(mountains):
        # m: (id, x, y, radius, height)
        mx = m[1]
        my = m[2]
        mradius = m[3]
        mheight = m[4]

        scale_factor = 5
        scaled_radius = mradius * scale_factor

        triangle = RegularPolygon((mx, my), numVertices=3, radius=scaled_radius,
                                  orientation=0,  # You can adjust this to rotate the triangle if desired.
                                  edgecolor='saddlebrown', facecolor='none', lw=2, zorder=1)
        ax.add_patch(triangle)
        mountain_markers.append(triangle)
        text = ax.text(mx, my + scaled_radius + 10, f"z: {mheight:.1f}",
                       fontsize=9, color='saddlebrown', ha='center', zorder=2)
        mountain_markers.append(text)
    return mountain_markers

def setup_drones(ax, drone_trajectories, drone_fixed_colors):

    drone_dots = {}
    drone_trails = {}
    drone_text = {}
    for i, (drone_id, traj) in enumerate(drone_trajectories.items()):
        color = drone_fixed_colors[i % len(drone_fixed_colors)]
        dot, = ax.plot([], [], 'o', color=color, markersize=8, label=f"Drone {drone_id}")
        trail, = ax.plot([], [], '-', color=color, alpha=0.4)
        drone_dots[drone_id] = dot
        drone_trails[drone_id] = trail
        text_obj = ax.text(0, 0, "", fontsize=9, color="black", ha='center')
        drone_text[drone_id] = text_obj
    return drone_dots, drone_trails, drone_text


def visualize_simulation(db, exp_id, drone_trajectories, region=(1000, 1000)):
    plt.ion()  # Enable interactive mode for PyCharm animations
    fig, ax = plt.subplots()
    ax.set_xlim(0, region[0])
    ax.set_ylim(0, region[1])
    ax.set_aspect('equal')

    # Set up sensors.
    sensors = db.get_sensors_by_exp_id(exp_id)
    sensor_dots, sensor_text, sensor_id_text = setup_sensors(ax, sensors)

    # Set up mountains.
    mountain_markers = setup_mountains(ax, db, exp_id)

    # Set up drones.
    drone_fixed_colors = ['red', 'lime']
    drone_dots, drone_trails, drone_text = setup_drones(ax, drone_trajectories, drone_fixed_colors)

    # Store persistent detection markers (triangles) and detection lines
    detection_markers = []  # List of all detection markers
    detection_lines = []  # List of detection lines

    def update(frame):
        all_artists = []

        for drone_id, traj in drone_trajectories.items():
            if frame >= len(traj):
                continue  # skip if frame exceeds drone's trajectory

            _, _, _, timestamp, _, x, y, z, *_ = traj[frame]
            drone_dots[drone_id].set_data([x], [y])

            drone_text[drone_id].set_position((x, y + 100))
            drone_text[drone_id].set_text(f"z: {z:.1f}")

            trail_length = 10
            trail_x = [traj[max(0, frame - i)][5] for i in range(trail_length) if (frame - i) >= 0]
            trail_y = [traj[max(0, frame - i)][6] for i in range(trail_length) if (frame - i) >= 0]
            drone_trails[drone_id].set_data(trail_x, trail_y)

            all_artists.extend([drone_dots[drone_id], drone_trails[drone_id], drone_text[drone_id]])

        # Add sensor visualizations if needed
        all_artists.extend(sensor_dots.values())
        all_artists.extend(sensor_text.values())
        all_artists.extend(mountain_markers)
        all_artists.extend(detection_markers)
        all_artists.extend(detection_lines)

        return all_artists

    # Choose frame count based on longest trajectory
    max_frames = max(len(traj) for traj in drone_trajectories.values())

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=300, blit=False)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Drone Flight Simulation - Exp {exp_id}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show(block=True)  # Keep the animation running in PyCharm