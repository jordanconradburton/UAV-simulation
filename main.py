import Simulator.DatabaseManager as dbm
from Simulator.graph_preprocessing import *
from Simulator.run_simple_simulations import *
from Graph_Models.train_model import run_pipeline

def run_simulation_experiments(db, num_exp, csv_out_path):

    for i in tqdm(range(num_exp)):
        run_simulation(db, num_drones=1, num_mountains=0, region_size=(5000, 5000))

    export_all_sensor_detection_summary(db, csv_out_path + "_detections.csv")
    export_sensor_distances_to_csv(db, csv_out_path + "_sensor_distances.csv")
    export_sensor_gain_sectors_to_csv(db, csv_out_path + "_sensor_gain_sectors.csv")
    export_sensor_information_to_csv(db, csv_out_path + "_sensor_information.csv")

def produce_graph_content(db, out_path):
    export_all_graphs_to_one_file(db, out_pt=out_path + "_graphs.pt")
    # export_drone_labels(db, csv_path=out_path + "_labels.csv")


if __name__ == '__main__':
    db_name = '1000simple_simulation.db'
    db = dbm.DatabaseManager(db_name=db_name)

    # num_exp = 1000
    # csv_out_path = "Data_Files_and_Databases/1st_Trial_Data_Files/1000_mountains0_drone1"
    # run_simulation_experiments(db, num_exp, csv_out_path)

    # graph_out_path = "Data_Files_and_Databases/1st_Trial_Data_Files_Graphs/1000_simple_simulations"
    # produce_graph_content(db, graph_out_path)
    #
    # db.conn.close()

    graphs_path = "Data_Files_and_Databases/1st_Trial_Data_Files_Graphs/1000_simple_simulations_graphs.pt"
    run_pipeline(graphs_path)