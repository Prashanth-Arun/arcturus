from dataclasses import dataclass
from typing import TypedDict, Dict
import csv
import json
import os

DEFAULT_EMOTIC_PATH = os.path.join(os.getcwd(), "data", "emotic")
UNNECESSARY_PROPERTIES = [
    "Filename", "Width", "Height", "Age", "Gender", 
    "X_min", "Y_min", "X_max", "Y_max", "Arr_name", "Crop_name"
]

@dataclass
class EmoticData(TypedDict):
    valence : float
    arousal : float
    dominance : float
    labels : list[str]

def preprocess(
    base_path : str,
    split : str,
) -> list[EmoticData]:
    """
    Loads a split for the Emotic dataset and returns a preprocessed version, which contains only relevant properties
    (i.e., Valence, Arousal, Dominance, string emotion labels)
    """
    preprocessed_dataset : list[EmoticData] = []
    split_path = os.path.join(base_path, f"{split}.csv")

    def strip_unnecessary(datapoint : Dict):
        for key in UNNECESSARY_PROPERTIES:
            del datapoint[key]

    with open(split_path, "r") as f:
        contents = csv.DictReader(f)
        for row in contents:
            strip_unnecessary(row)
            valence, arousal, dominance = (row['Valence'], row['Arousal'], row['Dominance'])
            del row['Valence'], row['Arousal'], row['Dominance']
            string_labels : list[str] = []
            for key in row:
                try:
                    val = float(row[key])
                    if val == 1.0:
                        string_labels.append(key)
                except Exception as e:
                    raise Exception(f"{row[key], key}", {e})
            datapoint = EmoticData(
                valence=float(valence),
                arousal=float(arousal),
                dominance=float(dominance),
                labels=string_labels
            )
            preprocessed_dataset.append(datapoint)

    return preprocessed_dataset

def export(
    dataset : list[EmoticData],
    base_path : str,
    split : str
):
    """
    Exports the preprocessed dataset to its own JSON file
    """
    destination_path = os.path.join(base_path, f"{split}.json")
    with open(destination_path, "w") as f:
        json.dump(dataset, fp=f, indent=4)

if __name__ == "__main__":
    splits : list[str] = ["train", "validation", "test", "train_extra"]
    for split in splits:
        dataset = preprocess(base_path=DEFAULT_EMOTIC_PATH, split=split)
        export(dataset, base_path=DEFAULT_EMOTIC_PATH, split=split)