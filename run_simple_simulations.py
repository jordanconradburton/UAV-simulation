import DatabaseManager as dbm
from calculations import *
from simulation_utils import *

db = dbm.DatabaseManager(db_name='simple_simulation.db')

# 1. First Simulation: drone trajectory = circle, 3 sensors randomly placed
def first_simulation(num_drones=1, num_sensors=3, region_size=(5000, 5000)):
    exp_id = 1
    num_circular = 1
    num_steps = 60
    drone_freq_mhz = 300

    setup_environment(db, exp_id, num_sensors=num_sensors, region_size=region_size)
    drone_trajectories = generate_mixed_trajectories(
        exp_id=exp_id,
        num_drones=num_drones,
        num_circular=num_circular,
        num_steps=num_steps
    )
    insert_all_drones(db, exp_id, drone_trajectories, freq_mhz=drone_freq_mhz)
    start, end = get_earliest_and_latest_times(drone_trajectories)
    simulate_sensor_up_times(db, exp_id, sensor_ids=[1, 2, 3], start=start, end=end)

    run_detection_simulation(db, exp_id, drone_trajectories)
    compute_metrics(db, exp_id)

    visualize_simulation(db, exp_id, drone_trajectories, region=region_size)


#2. Second Simulation: 2 drones, 1 with circle trajectory, 1 with random trajectory, 3 sensors randomly placed
def second_simulation(num_drones=2, num_sensors=3, region_size=(5000, 5000)):
    exp_id = 2
    num_circular = 1
    num_steps = 60
    drone_freq_mhz = 2400

    setup_environment(db, exp_id, num_sensors=num_sensors, region_size=region_size)
    drone_trajectories = generate_mixed_trajectories(
        exp_id=exp_id,
        num_drones=num_drones,
        num_circular=num_circular,
        num_steps=num_steps
    )
    insert_all_drones(db, exp_id, drone_trajectories, freq_mhz=drone_freq_mhz)

    start, end = get_earliest_and_latest_times(drone_trajectories)
    simulate_sensor_up_times(db, exp_id, sensor_ids=[1, 2, 3], start=start, end=end)

    run_detection_simulation(db, exp_id, drone_trajectories)
    compute_metrics(db, exp_id)

    visualize_simulation(db, exp_id, drone_trajectories, region=region_size)


if __name__ == '__main__':
    # first_simulation()
    second_simulation()
