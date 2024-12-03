from collections.abc import Iterable
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
    

@dataclass
class ArcturusConfig(TypedDict):
    from_text : bool
    from_vad : bool
    use_finetuned_chatmusician : bool
    whisper_variant : str
    label_classifier_config : StringLabelClassifierConfig
    vad_checkpoint : NotRequired[str]
    label_checkpoint : NotRequired[str]
    chatmusician_checkpoint : NotRequired[str]
    stream_output : bool

@dataclass
class AudioInput(TypedDict):
    path: NotRequired[str]
    array: Iterable
    sampling_rate: int