import DatabaseManager as dbm
from calculations import *
from simulation_utils import *
from simulate_environment import (setup_experiment, setup_topography, setup_environment_record,
                                  link_experiment_environment, setup_sensors)
from simulate_trajectories import (generate_mixed_trajectories, generate_random_trajectories,
                                   get_earliest_and_latest_times, insert_drone_trajectories_and_frequencies)
from visualization_utils import visualize_simulation

"""
TODO:
1. what is sensor sensitivity value?
Is this just the range for all sensors where they can "see" the drone (-80 to -100)?
Or does each individual sensor have a sensitivity limit? If so, how do we use this in the equation?

2. what is the sensor_up_time angle value?
Does this impact the detection and how?
"""

def run_simulation(db, exp_id, topography_id, env_id, num_drones, num_sensors, region_size, height, num_mountains,
                   drone_freq_range=(900e6, 5e9), drone_power_range=(10, 30), drone_gain_range=(0.0, 5.0),
                   drone_types=("Quad", "Hex", "Octo"), drone_properties=("Autonomous", "Manually Controlled", "Hybrid"),
                   num_steps=60, simulation_type="mixed"):

    #-------------------------------------------------------------------------------------------------
    # Setup simulation environment
    setup_experiment(db, exp_id, region_size=region_size, height=height)
    setup_topography(db, topography_id, region_size=region_size, num_mountains=num_mountains)
    setup_sensors(db, exp_id, num_sensors=num_sensors, region_size=region_size, height=height)
    setup_environment_record(db, env_id, disturbance=0.2, visibility=80)
    link_experiment_environment(db, exp_id=exp_id, topography_id=topography_id, env_id=env_id)

    #-------------------------------------------------------------------------------------------------
    # Create drones and their trajectories
    drone_info = create_drones(db, exp_id, num_drones, power_range=drone_power_range,
                  gain_range=drone_gain_range, drone_types=drone_types, properties_list=drone_properties)
    drone_ids = list(drone_info.keys())

    if simulation_type == "mixed":
        drone_trajectories = generate_mixed_trajectories(db,
            exp_id=exp_id,
            drone_ids=drone_ids,
            num_circular=1,
            num_steps=num_steps,
            drone_freq_range=drone_freq_range
        )
    else:
        drone_trajectories = generate_random_trajectories(db,
            exp_id=exp_id,
            drone_ids=drone_ids,
            num_steps=num_steps,
            drone_freq_range=drone_freq_range,
            height=height
        )

    insert_drone_trajectories_and_frequencies(db, exp_id, drone_trajectories)

    #-------------------------------------------------------------------------------------------------

    start, end = get_earliest_and_latest_times(drone_trajectories)
    simulate_sensor_up_times(db, exp_id, start=start, end=end)

    run_detection_simulation(db, exp_id, drone_trajectories)
    compute_metrics(db, exp_id)

    visualize_simulation(db, exp_id, drone_trajectories, region=region_size)

# 1. First Simulation: drone trajectory = circle, 3 sensors randomly placed
def first_simulation(db, num_drones=1, num_sensors=3, region_size=(5000, 5000), height=500):
    exp_id = 1
    topography_id = 1
    env_id = 1
    num_steps = 60

    run_simulation(db, exp_id, topography_id, env_id, num_drones, num_sensors, region_size, height, num_mountains=0,
                   num_steps=num_steps, simulation_type="mixed")


#2. Second Simulation: 2 drones, 1 with circle trajectory, 1 with random trajectory, 3 sensors randomly placed
def second_simulation(db, num_drones=2, num_sensors=3, region_size=(5000, 5000), height=500):
    exp_id = 2
    topography_id = 2
    env_id = 2
    num_steps = 60

    run_simulation(db, exp_id, topography_id, env_id, num_drones, num_sensors, region_size, height, num_mountains=0,
                   num_steps=num_steps, simulation_type="mixed")

#3. Single Mountain Simulation: 2 drones with random trajectories, 3 sensors, and 1 mountain
def single_mountain_simulation(db, num_drones=2, num_sensors=3, region_size=(5000, 5000), height=500):
    exp_id = 3
    topography_id = 3
    env_id = 3

    num_steps = 60
    num_mountains = 1

    run_simulation(db, exp_id, topography_id, env_id, num_drones, num_sensors, region_size, height,
                   num_mountains=num_mountains, num_steps=num_steps, simulation_type="mountain")

#4. Mountain Range Simulation: 2 drones, 3 sensors, and 5 mountains
def mountain_range_simulation(db, num_drones=2, num_sensors=3, region_size=(5000, 5000), height=100):
    exp_id = 4
    topography_id = 4
    env_id = 4

    num_steps = 60
    num_mountains = 5

    run_simulation(db, exp_id, topography_id, env_id, num_drones, num_sensors, region_size, height,
                   num_mountains=num_mountains, num_steps=num_steps, simulation_type="mountain")


if __name__ == '__main__':
    db = dbm.DatabaseManager(db_name='simple_simulation.db')
    # first_simulation(db)
    # second_simulation(db)
    # single_mountain_simulation(db)
    mountain_range_simulation(db)

    db.conn.close()
