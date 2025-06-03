import DatabaseManager as dbm
from calculations import *
from simulation_utils import *
from simulate_environment import (setup_experiment, setup_topography, setup_environment_record,
                                  link_experiment_environment, setup_sensors)
from simulate_trajectories import (generate_random_trajectories,
                                   get_earliest_and_latest_times, insert_drone_trajectories_and_frequencies)
from visualization_utils import visualize_simulation


"""
TODO:
1. environmental features (later)
- how do disturbance and visibility affect the calculations

2. what is the maximum z for sensor placement? Is 500 the maximum z for targets?

3. is there a range for transmitter gain or set to 0.0?
- is this a set value per drone or can it change along the trajectory?

4. what are different scanning rates for the sensors?

5. is it correct to have different region sizes as coded below?

6. is it correct to change the drone radio frequency randomly all the time?

7. does a drone's bandwidth stay the same throughout the experiment?
how is it determined? is it the range of the radio frequency band?
"""

def run_simulation(
    db,
    num_drones=None,
    num_sensors=None,
    region_size=None,
    sensor_height=None,
    target_height=None,
    num_mountains=None,
    num_steps=60
):
    """
    Run one full experiment with optional parameters.
    Any argument left as None will be randomized.
    Only random trajectories are supported.
    """
    # ------- assign auto-increment IDs -------
    exp_id = db.get_next_experiment_id()
    topo_id = db.get_next_topography_id()
    env_id  = db.get_next_environment_id()

    # ------- randomize defaults -------
    num_sensors   = num_sensors or random.randint(3, 20)
    num_drones    = num_drones  or random.randint(1, 8)
    region_size   = region_size or (random.randint(100, 5000), random.randint(100, 5000))
    sensor_height = sensor_height  or 100           #TODO: check what is the maximum z for sensor placement
    num_mountains = num_mountains or random.randint(0, 5)
    target_height = target_height or 500

    # ------- environment setup -------
    setup_experiment(db, exp_id, region_size=region_size)
    setup_topography(db, topo_id, region_size=region_size, num_mountains=num_mountains)
    setup_environment_record(db, env_id, disturbance=random.uniform(0,1), visibility=random.uniform(10,100))
    link_experiment_environment(db, exp_id, topo_id, env_id)
    setup_sensors(db, exp_id, num_sensors=num_sensors, region_size=region_size, height=sensor_height)

    # ------- drone creation & trajectories -------
    drone_info = create_drones(db, exp_id, num_drones)
    drone_ids  = list(drone_info.keys())
    drone_freq_ranges = {did: info['freq_range'] for did, info in drone_info.items()}

    trajectories = generate_random_trajectories(
        db, exp_id, drone_ids,
        num_steps=num_steps,
        drone_freq_ranges=drone_freq_ranges,
        height=target_height,
        region=region_size
    )

    insert_drone_trajectories_and_frequencies(db, exp_id, trajectories)

    # ------- run and record detections -------
    start, end = get_earliest_and_latest_times(trajectories)
    simulate_sensor_up_times(db, exp_id, start=start, end=end)
    run_detection_simulation(db, exp_id, trajectories)

    # ------- visualize -------
    visualize_simulation(db, exp_id, trajectories, region=region_size)


if __name__ == '__main__':
    db = dbm.DatabaseManager(db_name='simple_simulation.db')
    run_simulation(db)
    db.conn.close()
