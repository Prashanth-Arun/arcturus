from transformers import BertModel, AutoModelForCausalLM, WhisperForConditionalGeneration
from torch import Tensor
from torch.nn import Module, Linear, Dropout
from typing import TypeVar

Data = TypeVar("Data") # placeholder; remove afterwards.
DataConfig = TypeVar("DataConfig") # placeholder

ChatMusicianModel = TypeVar("ChatMusicianModel", AutoModelForCausalLM)
WhisperModel = TypeVar("WhisperModel", WhisperForConditionalGeneration)

class BERTForVADMapping(Module):

    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-cased")
        self.dropout = Dropout(self.model.config['hidden_dropout_prob'])
        self.vad_head = Linear(self.model.config['hidden_size'], 3)

    def forward(self, data : Data) -> Tensor:
        _last_hidden_state, pooler_out = self.model(**data)
        pooler_out = self.dropout(pooler_out)
        vad_embeddings = self.vad_head(pooler_out)
        return vad_embeddings
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path : str
    ) -> "BERTForVADMapping":
        pass
    

class StringLabelClassifier(Module):

    def __init__(self, config : DataConfig):
        super().__init__()
        self.labels = config.labels
        self.input_layer = Linear(3, config.hidden_size)
        self.output_layer = Linear(config.hidden_size, len(self.labels))
        self.dropout = Dropout(config.dropout)

    def forward(self, data : Data) -> Tensor:
        layer_1_out = self.input_layer(data)
        layer_1_drop = self.dropout(layer_1_out)
        prediction = self.output_layer(layer_1_drop)
        return prediction
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path : str
    ) -> "StringLabelClassifier":
        pass