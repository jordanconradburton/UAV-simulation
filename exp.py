import json, os, time, random, numpy as np, torch
from Simulator.model_preprocessing import *


def set_all_seeds(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def start_run(run_name: str, root="runs", config: dict=None):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, f"{run_name}_{stamp}")
    os.makedirs(path, exist_ok=True)
    if config:
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    # pre-create subfolders most pipelines want
    for sub in ["checkpoints", "plots", "metrics", "artifacts"]:
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    return path

def save_json(obj: dict, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_model(model, run_dir: str, name="checkpoints/best_model.pt"):
    torch.save(model.state_dict(), os.path.join(run_dir, name))

def create_outfiles(db, csv_out_path):
    export_all_sensor_detection_summary(db, csv_out_path + "_detections.csv")
    export_sensor_distances_to_csv(db, csv_out_path + "_sensor_distances.csv")
    export_sensor_gain_sectors_to_csv(db, csv_out_path + "_sensor_gain_sectors.csv")
    export_sensor_information_to_csv(db, csv_out_path + "_sensor_information.csv")
    export_drone_trajectories_to_csv(db, csv_out_path + "_drone_trajectories.csv")
