import os
from dataclasses import dataclass
from typing import TypedDict

DEFAULT_EMOTIC_PATH = os.path.join(os.getcwd(), "data", "emotic")

UNNECESSARY_PROPERTIES = [
    "Filename", "Width", "Height", "Age", "Gender", 
    "X_min", "Y_min", "X_max", "Y_max", "Arr_name", "Crop_name"
]

LABEL_TO_STRING_MAP = dict({
    0:  "Peace",
    1:  "Affection",
    2:  "Esteem",
    3:  "Anticipation",
    4:  "Engagement",
    5:  "Confidence",
    6:  "Happiness",
    7:  "Pleasure",
    8:  "Excitement",
    9:  "Surprise",
    10: "Sympathy",
    11: "Doubt/Confusion",
    12: "Disconnection",
    13: "Fatigue",
    14: "Embarrassment",
    15: "Yearning",
    16: "Disapproval",
    17: "Aversion",
    18: "Annoyance",
    19: "Anger",
    20: "Sensitivity",
    21: "Sadness",
    22: "Disquietment",
    23: "Fear",
    24: "Pain",
    25: "Suffering"
})

@dataclass
class EmoticItem(TypedDict):
    valence : float
    arousal : float
    dominance : float
    labels : list[float]

