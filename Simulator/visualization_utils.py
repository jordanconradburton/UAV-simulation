import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.patches import Wedge

from datetime import datetime

# matplotlib.use("TkAgg")
# matplotlib.use("Agg")   # Non-interactive backend

def setup_sensors(ax, sensors, region):

    sensor_dots = {}
    sensor_text = {}
    sensor_id_text = {}
    color = "black"
    y_offset_text = min(region) * 0.02 #100
    y_offset_id_text = min(region) * 0.04 #-200
    size = min(4, min(region) * 0.01) #8

    for i, s in enumerate(sensors):
        _, sensor_id, _, sx, sy, sz = s
        label = "Sensor" if i == 0 else ""  # Only label the first sensor.
        dot, = ax.plot(sx, sy, 'o', color=color, markersize=size, label=label)
        sensor_dots[sensor_id] = dot
        sensor_text[sensor_id] = ax.text(sx, sy + y_offset_text, f"z: {sz:.1f}",
                                         fontsize=9, color="black", ha='center')
        sensor_id_text[sensor_id] = ax.text(sx, sy - y_offset_id_text, f"ID: {sensor_id}",
                                            fontsize=9, color="black", ha='center')
    return sensor_dots, sensor_text, sensor_id_text

def setup_sensor_sectors(ax, db, exp_id, region):
    """
    Draws sector wedges around each sensor, shaded according to gain.
    Returns a dict sensor_id -> list of Wedge patches.
    """
    # Load per-sensor patterns: list of (start, end, gain_db)
    patterns = db.load_sensor_patterns(exp_id)
    # Gather global gain min/max for normalization
    all_gains = [g for sectors in patterns.values() for (_, _, g) in sectors]
    if not all_gains:
        return {}
    gmin, gmax = min(all_gains), max(all_gains)
    # Fallback if flat
    delta = gmax - gmin if gmax > gmin else 1.0

    wedges = {}
    # Sensor positions
    sensors = db.get_sensors_by_exp_id(exp_id)
    for s in sensors:
        _, _, sensor_id, sx, sy, _ = s
        sector_patches = []
        # choose display radius relative to region or fixed
        radius = min(region) * 0.05
        for (start, end, gain_db) in patterns.get(sensor_id, []):
            # normalize gain to [0,1]
            norm = (gain_db - gmin) / delta
            # invert for shading: higher gain -> lighter
            shade = 1.0 - norm
            face = str(shade)
            wedge = Wedge(
                center=(sx, sy),
                r=radius,
                theta1=start,
                theta2=end,
                facecolor=face,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.5,
                zorder=0
            )
            ax.add_patch(wedge)
            sector_patches.append(wedge)
        wedges[sensor_id] = sector_patches
    return wedges

def setup_mountains(ax, db, exp_id, region):

    y_offset = min(region) * 0.02
    mountains = db.get_mountains_by_experiment(exp_id)
    mountain_markers = []
    for i, m in enumerate(mountains):
        # m: (id, x, y, side, height)
        mx = m[1]
        my = m[2]
        mside = m[3]
        mheight = m[4]

        half = mside / 2.0
        # draw square footprint
        square = Rectangle(
            (mx - half, my - half),
            mside, mside,
            facecolor='saddlebrown',
            edgecolor='black',
            alpha=0.6,
            linewidth=1,
            zorder=1
        )
        ax.add_patch(square)
        mountain_markers.append(square)
        text = ax.text(mx, my + half + y_offset, f"z: {mheight:.1f}",
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


def build_detection_map(db, exp_id):
    """
    Returns a dict mapping each datetime to a list of (sensor_id, drone_id)
    pairs for every detection at that moment.
    """
    # 1) Map sensor_up_id → (sensor_id, timestamp_datetime)
    sensor_up_rows = db.get_sensor_up_time_by_experiment(exp_id)
    sensor_up_map = {}
    for row in sensor_up_rows:
        # row: (exp_id, sensor_id, sensor_up_id, start_time, ...)
        sup_id    = row[2]
        sensor_id = row[1]
        ts_raw    = row[3]
        # Parse the string into a datetime
        if isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
        else:
            ts = ts_raw
        sensor_up_map[sup_id] = (sensor_id, ts)

    # 2) Build timestamp → [(sensor_id, drone_id), …]
    detection_map = {}
    for det in db.get_sensor_detections():
        # det: (detection_id, sensor_up_id, drone_id, ...)
        _, sup_id, drone_id, _, _, _ = det
        if sup_id in sensor_up_map:
            sensor_id, ts = sensor_up_map[sup_id]
            detection_map.setdefault(ts, []).append((sensor_id, drone_id))

    return detection_map


def visualize_simulation(db, exp_id, drone_trajectories, region=(1000, 1000)):
    plt.ion()  # Enable interactive mode for PyCharm animations
    fig, ax = plt.subplots()
    ax.set_xlim(0, region[0])
    ax.set_ylim(0, region[1])
    ax.set_aspect('equal')

    # Set up sensors.
    sensors = db.get_sensors_by_exp_id(exp_id)
    sensor_dots, sensor_text, sensor_id_text = setup_sensors(ax, sensors, region)
    sensors_dict = { s[2]:(s[3],s[4],s[5]) for s in sensors }
    # sector_patches = setup_sensor_sectors(ax, db, exp_id, region)

    # Set up mountains.
    mountain_markers = setup_mountains(ax, db, exp_id, region)

    # Set up drones.
    cmap = plt.cm.get_cmap('tab10')
    n_drones = len(drone_trajectories)
    drone_fixed_colors = [cmap(i) for i in range(n_drones)]
    drone_dots, drone_trails, drone_text = setup_drones(ax, drone_trajectories, drone_fixed_colors)
    drones_dict = { did: traj for did,traj in drone_trajectories.items() }

    detection_lines = []  # List of detection lines
    detection_map = build_detection_map(db, exp_id)

    def update(frame):

        #remove old detection lines
        for ln in detection_lines:
            ln.remove()
        detection_lines.clear()

        all_artists = []

        for drone_id, traj in drone_trajectories.items():
            if frame >= len(traj):
                continue  # skip if frame exceeds drone's trajectory

            _, _, timestamp, _, x, y, z, *_ = traj[frame]
            drone_dots[drone_id].set_data([x], [y])

            drone_text[drone_id].set_position((x, y + 100))
            drone_text[drone_id].set_text(f"z: {z:.1f}")

            trail_length = 10
            trail_x = [traj[max(0, frame - i)][5] for i in range(trail_length) if (frame - i) >= 0]
            trail_y = [traj[max(0, frame - i)][6] for i in range(trail_length) if (frame - i) >= 0]
            drone_trails[drone_id].set_data(trail_x, trail_y)

            all_artists.extend([drone_dots[drone_id], drone_trails[drone_id], drone_text[drone_id]])

        current_ts = next(traj[frame][3]
                          for traj in drone_trajectories.values()
                          if frame < len(traj))

        for sensor_id, drone_id in detection_map.get(current_ts, []):
            sx, sy, _ = sensors_dict[sensor_id]
            dx, dy, _ = drones_dict[drone_id][frame][5:8]
            ln, = ax.plot([sx, dx], [sy, dy], '--', lw=1,
                          color='gray', alpha=0.7)
            detection_lines.append(ln)
            all_artists.append(ln)

        # Add sensor visualizations if needed
        all_artists.extend(sensor_dots.values())
        all_artists.extend(sensor_text.values())
        all_artists.extend(mountain_markers)
        all_artists.extend(detection_lines)
        # for patches in sector_patches.values():
        #     all_artists += patches

        return all_artists

    # Choose frame count based on longest trajectory
    max_frames = max(len(traj) for traj in drone_trajectories.values())

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=300, blit=False)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Drone Flight Simulation - Exp {exp_id}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show(block=True)  # Keep the animation running in PyCharm
    return ani