from .dataset import EmoticItem
from typing import Dict
import csv
import json
import os

DEFAULT_EMOTIC_PATH = os.path.join(os.getcwd(), "data", "emotic")
UNNECESSARY_PROPERTIES = [
    "Filename", "Width", "Height", "Age", "Gender", 
    "X_min", "Y_min", "X_max", "Y_max", "Arr_name", "Crop_name"
]

def preprocess(
    base_path : str,
    split : str,
) -> list[EmoticItem]:
    """
    Loads a split for the Emotic dataset and returns a preprocessed version, which contains only relevant properties
    (i.e., Valence, Arousal, Dominance, string emotion labels)
    """
    preprocessed_dataset : list[EmoticItem] = []
    split_path = os.path.join(base_path, f"{split}.csv")

    def strip_unnecessary(datapoint : Dict):
        for key in UNNECESSARY_PROPERTIES:
            del datapoint[key]

    def scale(value : float):
        """
        Scale value down from Emotic's range (1-10) to EmoBank's range (1-5)
        """
        return round((max([(value - 1), 0]) / 2.25) + 1, 3)

    with open(split_path, "r") as f:
        contents = csv.DictReader(f)
        for row in contents:
            strip_unnecessary(row)
            valence, arousal, dominance = (row['Valence'], row['Arousal'], row['Dominance'])
            del row['Valence'], row['Arousal'], row['Dominance']
            emotion_labels = [float(row[k]) for k in row]
            
            datapoint = EmoticItem(
                valence=scale(float(valence)),
                arousal=scale(float(arousal)),
                dominance=scale(float(dominance)),
                labels=emotion_labels
            )
            preprocessed_dataset.append(datapoint)

    return preprocessed_dataset

def export(
    dataset : list[EmoticItem],
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