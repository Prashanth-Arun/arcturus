from .dataset import EmoBankItem
import csv
import json
import os

DEFAULT_EMOBANK_PATH = os.path.join(os.getcwd(), "data", "emobank", "raw.csv")

def preprocess(
    path : str = DEFAULT_EMOBANK_PATH
) -> tuple[list[EmoBankItem], list[EmoBankItem], list[EmoBankItem]]:
    """
    Loads the raw EmoBank CSV and returns a train-eval-test split with only the required attributes.
    """
    train_dataset : list[EmoBankItem] = []
    eval_dataset : list[EmoBankItem] = []
    test_dataset : list[EmoBankItem] = []

    with open(path, "r") as f:
        contents = csv.DictReader(f)
        for row in contents:
            datapoint = EmoBankItem(
                valence=float(row['V']),
                arousal=float(row['A']),
                dominance=float(row['D']),
                text=row['text']
            )
            match row["split"]:
                case "train":
                    train_dataset.append(datapoint)
                case "dev":
                    eval_dataset.append(datapoint)
                case "test":
                    test_dataset.append(datapoint)
                case _:
                    continue
    return train_dataset, eval_dataset, test_dataset


def export(
    train_set : list[EmoBankItem],
    eval_set : list[EmoBankItem],
    test_set : list[EmoBankItem],
    original_path : str = DEFAULT_EMOBANK_PATH
) -> None:
    
    directory = os.path.dirname(original_path)
    train_path = os.path.join(directory, "train.json")
    eval_path = os.path.join(directory, "validation.json")
    test_path = os.path.join(directory, "test.json")

    with open(train_path, "w") as f:
        json.dump(train_set, fp=f, indent=4)

    with open(eval_path, "w") as f:
        json.dump(eval_set, fp=f, indent=4)

    with open(test_path, "w") as f:
        json.dump(test_set, fp=f, indent=4)


if __name__ == "__main__":
    train_set, eval_set, test_set = preprocess()
    export(train_set, eval_set, test_set)