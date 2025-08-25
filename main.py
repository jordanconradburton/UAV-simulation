from utils import *
import argparse, os

import Simulator.DatabaseManager as dbm
from Simulator.graph_preprocessing import *
from Simulator.run_simple_simulations import *
from Graph_Models.train_encoder import run_encoder_pipeline
from Phase2_Decoder.train_decoder import run_decoder_pipeline
from exp import set_all_seeds, start_run, save_json, build_feature_weights

#TODO: move these to simulator folder
def run_simulation_experiments(db, num_exp, csv_out_path):

    for i in tqdm(range(num_exp)):
        run_simulation(db, num_drones=1, num_mountains=0, region_size=(5000, 5000))

    export_all_sensor_detection_summary(db, csv_out_path + "_detections.csv")
    export_sensor_distances_to_csv(db, csv_out_path + "_sensor_distances.csv")
    export_sensor_gain_sectors_to_csv(db, csv_out_path + "_sensor_gain_sectors.csv")
    export_sensor_information_to_csv(db, csv_out_path + "_sensor_information.csv")

def produce_graph_content(db, out_path):
    export_all_graphs_to_one_file(db, out_pt=out_path + "_graphs.pt")
    export_drone_labels(db, csv_path=out_path + "_labels.csv")


def cli():
    p = argparse.ArgumentParser(description="UAV Graph Project Runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1) simulator
    psim = sub.add_parser("simulator", help="Run simulations and build graphs")
    psim.add_argument("--db-name", required=True)
    psim.add_argument("--num-exp", type=int, default=1000)
    psim.add_argument("--csv-out", required=True, help="Folder for raw CSV detections")
    psim.add_argument("--out-path", required=True, help="File path for graphs and labels")
    psim.add_argument("--run-name", default="simulator")
    psim.add_argument("--seed", type=int, default=1337)

    # 2) train-encoder
    penc = sub.add_parser("train-encoder", help="Train encoder to produce embeddings")
    penc.add_argument("--graphs", required=True)
    penc.add_argument("--labels", required=True)
    penc.add_argument("--run-name", default="encoder")
    penc.add_argument("--seed", type=int, default=1337)
    penc.add_argument("--power-index", type=int, default=-1)
    penc.add_argument("--power-weight", type=float, default=20.0)
    penc.add_argument("--static-weight", type=float, default=1.0)
    penc.add_argument("--epochs", type=int, default=50)
    penc.add_argument("--batch-size", type=int, default=16)
    penc.add_argument("--device", default="cpu")

    # 3) train-decoder
    pdec = sub.add_parser("train-decoder", help="Train phase-2 position decoder on saved embeddings")
    pdec.add_argument("--train-file", required=True)
    pdec.add_argument("--val-file", required=True)
    pdec.add_argument("--test-file", required=True)
    pdec.add_argument("--run-name", default="decoder")
    pdec.add_argument("--seed", type=int, default=1337)
    pdec.add_argument("--epochs", type=int, default=50)
    pdec.add_argument("--batch-size", type=int, default=16)
    pdec.add_argument("--device", default="cpu")

    # 4) all-in-one
    pall = sub.add_parser("all", help="Run simulator -> encoder -> decoder sequence")
    pall.add_argument("--db-name", required=True)
    pall.add_argument("--num-exp", type=int, default=1000)
    pall.add_argument("--root-out", required=True, help="Root folder where all outputs live")
    pall.add_argument("--run-name", default="full")
    pall.add_argument("--seed", type=int, default=1337)
    pall.add_argument("--device", default="cpu")

    return p.parse_args()

def main():
    args = cli()
    set_all_seeds(args.seed)

    if args.cmd == "simulator":
        run_dir = start_run(args.run_name, config=vars(args))
        # 1) run experiments -> CSVs
        run_simulation_experiments(db=args.db_name, num_exp=args.num_exp, csv_out_path=args.csv_out)
        # 2) build graphs + labels
        produce_graph_content(db=args.db_name,
                               out_path=args.out_path)
        print(f"[simulator] Outputs in: {run_dir}")
        save_json({"status":"ok"}, os.path.join(run_dir, "metrics", "simulator.json"))

    elif args.cmd == "train-encoder":
        run_dir = start_run(args.run_name, config=vars(args))
        feat_weights = build_feature_weights(Fdim=16, dynamic_idx=args.power_index, w_static=args.static_weight,
                                             w_dynamic=args.dynamic_weight)

        metrics = run_encoder_pipeline(
            graphs_path=args.graphs, label_file=args.labels,
            epochs=args.epochs, batch_size=args.batch_size,
            feat_weights=feat_weights)
        save_json(metrics, os.path.join(run_dir, "metrics", "encoder.json"))
        print(f"[encoder] Outputs in: {run_dir}")

    elif args.cmd == "train-decoder":
        run_dir = start_run(args.run_name, config=vars(args))
        metrics = run_decoder_pipeline(
            train_embeddings_file=args.train_file, val_embeddings_file=args.val_file,
            test_embeddings_file=args.test_file, epochs=args.epochs)
        save_json(metrics, os.path.join(run_dir, "metrics", "decoder.json"))
        print(f"[decoder] Outputs in: {run_dir}")

    elif args.cmd == "all":
        run_dir = start_run(args.run_name, config=vars(args))
        # 1) simulator → write into run_dir/artifacts/
        # ...
        # 2) train encoder → save ckpt & embeddings to run_dir/artifacts/
        # ...
        # 3) train decoder → metrics in run_dir/metrics/
        # ...
        print(f"[all] Outputs in: {run_dir}")

# if __name__ == "__main__":
#     main()



if __name__ == '__main__':
    db_name = '1000simple_simulation.db'
    db = dbm.DatabaseManager(db_name=db_name)

    # ------ 1. Run Simulation Experiments ------

    # num_exp = 1000
    # csv_out_path = "Data_Files_and_Databases/1st_Trial_Data_Files/1000_mountains0_drone1"
    # run_simulation_experiments(db, num_exp, csv_out_path)

    # ------ 2. Create Graphs from Simulations ------
    # graph_out_path = "Data_Files_and_Databases/1st_Trial_Data_Files_Graphs/1000_simple_simulations"
    # produce_graph_content(db, graph_out_path)
    #
    # db.conn.close()

    # ------ 3. Run Encoder Training Pipeline ------
    graphs_path = "Data_Files_and_Databases/1st_Trial_Data_Files_Graphs/1000_simple_simulations_graphs.pt"
    label_file = "Data_Files_and_Databases/1st_Trial_Data_Files_Graphs/1000_simple_simulations_labels.csv"

    model_name = "GAT_weighted_dynamic_50" #GATv2_weighted_dynamic_50, BasicGATv2
    embeddings_save_path = "Graph_Models/outputs/embeddings/" + model_name + "_"
    plots_save_path = "Graph_Models/outputs/plots/" + model_name + "/"
    feat_weights = build_feature_weights(15, -1, 1.0, 50.0)

    run_encoder_pipeline(graphs_path, label_file, feat_weights=feat_weights,
                         embeddings_save_path=embeddings_save_path, plots_save_path=plots_save_path)

    train_embeddings_path = embeddings_save_path + "train_embeddings_with_labels.pt"
    val_embeddings_path = embeddings_save_path + "val_embeddings_with_labels.pt"
    test_embeddings_path = embeddings_save_path + "test_embeddings_with_labels.pt"

    inspect_embeddings(train_embeddings_path, save_dir="Graph_Models/outputs/embeddings_metrics/" + model_name + "/train_")
    inspect_embeddings(val_embeddings_path, save_dir="Graph_Models/outputs/embeddings_metrics/" + model_name + "/val_")

    # ------ 4. Run Decoder Training Pipeline ------
    decoder_plots_save_path = "Phase2_Decoder/outputs/plots/" + model_name + "/"
    decoder_metrics_save_path = "Phase2_Decoder/outputs/metrics/" + model_name + "/"

    run_decoder_pipeline(train_embeddings_path, val_embeddings_path, test_embeddings_path,
                         plots_save_path=decoder_plots_save_path, metrics_save_path=decoder_metrics_save_path)

