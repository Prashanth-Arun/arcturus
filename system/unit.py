from dataset.emotic import LABEL_TO_STRING_MAP
from dataclasses import dataclass
from typing import TypedDict, NotRequired, Mapping
from torch import Tensor

@dataclass
class BERTForVADMappingOutput(TypedDict):
    vad_values : Tensor
    loss : NotRequired[Tensor]


@dataclass
class StringLabelClassifierOutput(TypedDict):
    predictions : Tensor
    scaled_logits : Tensor
    loss : NotRequired[Tensor]


@dataclass
class StringLabelClassifierConfig(TypedDict):
    labels : Mapping[int, str]
    dropout : float
    threshold : float
    hidden_size : int

    @staticmethod
    def default(
        labels : Mapping[int, str] = LABEL_TO_STRING_MAP,
        dropout : float = 0.1,
        threshold : float = 0.5,
        hidden_size : int = 1024
    ) -> "StringLabelClassifierConfig":
        
        return StringLabelClassifierConfig(
            labels=labels,
            dropout=dropout,
            threshold=threshold,
            hidden_size=hidden_size
        )