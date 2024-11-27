
from .components import BERTForVADMapping, StringLabelClassifier
from .chatmusician import load_chatmusician_from_checkpoint, load_chatmusician_from_pretrained # type: ignore
from torch import Tensor
from transformers import AutoModelForCausalLM, WhisperForConditionalGeneration, WhisperProcessor
from typing import TypeVar
from string import Template

# ================================================================
"""
PROMPT_[VAD|STR] are prompts that are fed into the model to ask it 
to structure a piece of music based on the VAD/STR emo values.

PROMPT_GEN takes the output of the previous prompt and generates a
music file.
"""
PROMPT_VAD = Template("...") 
PROMPT_STR = Template("...")
PROMPT_GEN = Template("...")
# ================================================================

ArcturusConfig = TypeVar("ArcturusConfig") # placeholder
Input = TypeVar("Input", str) # can either be a string or an audio file
Output = TypeVar("Output", str, Tensor)

class ArcturusModel():

    def __init__(self, config : ArcturusConfig):

        self.processor = WhisperProcessor.from_pretrained(config.whisper_variant)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(config.whisper_variant)
        self.vad_classifier = BERTForVADMapping.from_pretrained(config.vad_classifier_checkpoint) # type: ignore
        self.string_classifier = StringLabelClassifier.from_pretrained(config.string_classifier_checkpoint) # type: ignore
        self.chatmusician = load_chatmusician() # type: ignore


    def __call__(self, input : Input) -> Output:
        pass