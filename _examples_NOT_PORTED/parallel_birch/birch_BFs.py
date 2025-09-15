import argparse
import pickle
import time

if __name__ == "__main__":
    import bitbirch.bitbirch as bb

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--threshold", help="The threshold for the BitBirch model", required=True
    )

    # Parse the arguments
    args = parser.parse_args()

    # Read the fps_split indexes from the .txt file
    with open("fps_splits.txt", "r") as f:
        splits = f.read().splitlines()

    start = time.time()
    bb.set_merge("tolerance", 0.00)
    brc = bb.BitBirch(
        threshold=float(args.threshold), branching_factor=50
    )  # threshold could be changed by the user
    for split in splits:
        with open(f"BFs/BFs_{split}.pkl", mode="rb") as fb:
            data = pickle.load(fb)
        brc.fit_BFs(data)

    for split in splits:
        with open(f"BFs/BFs_problematic_{split}.pkl", mode="rb") as fb:
            data = pickle.load(fb)
        brc.fit_BFs(data)

    f.close()
    del data

    # Save centroids and clusters
    with open("final_centroids_parallel.pkl", "wb") as fwb:
        pickle.dump(brc.get_centroids(), fwb)

    # Save the clustered_ids
    with open("clustered_ids_parallel.pkl", "wb") as fwb:
        pickle.dump(brc.get_cluster_mol_ids(), fwb)

    # Save the time in a .txt file
    with open("time_centroid_clustering.txt", "w") as fwt:
        fwt.write(str(time.time() - start))
        fwt.write("\n")
