
import json
import pickle

# Load JSON file
for file in ["data/STAR_train.json", "data/STAR_val.json", "data/STAR_test.json"]:
    with open(file, "r") as json_file:
        data = json.load(json_file)

    # Save as a Pickle (.pkl) file
    with open(file.replace(".json", ".pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)

    print("Conversion complete: data.json → data.pkl")