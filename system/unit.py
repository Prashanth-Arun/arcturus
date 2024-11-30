
from dataclasses import dataclass
from typing import TypedDict, NotRequired
from torch import Tensor

@dataclass
class BERTForVADMappingOutput(TypedDict):
    vad_values : Tensor
    loss : NotRequired[Tensor]