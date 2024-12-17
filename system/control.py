from math import e, pi, sin, pow
from torch import Tensor, tensor
from typing import Callable, TypeVar

ControlFunction = TypeVar("ControlFunction", Callable[[float], float], Callable[[Tensor], Tensor])

class ControlMechanism:

    def __init__(
        self, 
        alpha: float = 1.0, 
        midpoint: float = 3.0,
        lower_bound: float = 1.0,
        upper_bound: float = 5.0
    ):

        def sigmoid(v : float, offset : float = midpoint):
            return 1 / (1 + pow(e, offset - v))

        self.valence_control : ControlFunction = lambda valence : sin( (4 / pi) * (valence - midpoint) )
        self.arousal_control : ControlFunction = lambda arousal : sigmoid(arousal) * (1 - sigmoid(arousal))
        self.dominance_control : ControlFunction = lambda dominance : midpoint - dominance
        self.control : ControlFunction = lambda val, func : min(upper_bound, max(lower_bound, val + (alpha * func(val))))


    def __call__(self, vad_embeddings : Tensor) -> Tensor:

        assert vad_embeddings.shape == (3,)
        values = vad_embeddings.clone().detach()
        valence = self.control(values[0].item(), self.valence_control)
        arousal = self.control(values[1].item(), self.arousal_control)
        dominance = self.control(values[2].item(), self.dominance_control)

        return tensor([valence, arousal, dominance])