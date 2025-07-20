import ast
import csv
from datetime import datetime

#TODO: do we need the drone ids as well or just number and positions?


def load_labels(csv_path, max_drones=8):
    labels = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = int(row["exp_id"])
            ts = row["timestamp"]

            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")

            num = int(row["num_drones"])

            positions = ast.literal_eval(row["drone_positions"]) #TODO: check
            if len(positions) != max_drones:
                positions += [(0.0, 0.0, 0.0)] * (max_drones - len(positions))

            labels[(exp_id, ts)] = {
                "num_drones": num,
                "positions": positions
            }
    return labels
