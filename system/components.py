from transformers import (
    BertModel, 
    BertConfig,
    AutoModelForCausalLM, 
    WhisperForConditionalGeneration
)
from dataset.emobank import EmoBankData
from dataset.emotic import EmoticItem
from transformers.modeling_outputs import BaseModelOutput
from torch import Tensor
from torch.nn import Module, Linear, Dropout, MSELoss
from typing import TypeVar
from .unit import BERTForVADMappingOutput

import torch

DataConfig = TypeVar("DataConfig") # placeholder

class BERTForVADMapping(Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = BertModel.from_pretrained("bert-base-cased").to(self.device)
        self.bert_config = self.model.config
        assert isinstance(self.bert_config, BertConfig)

        self.dropout = Dropout(self.bert_config.hidden_dropout_prob)
        self.vad_head = Linear(self.bert_config.hidden_size, 3).to(self.device)
        self.loss_criterion = MSELoss()


    def forward(self, data : EmoBankData) -> BERTForVADMappingOutput:

        if self.training:
            assert 'labels' in data

        B, T = data['input_ids'].shape
        E = self.bert_config.hidden_size

        encoder_output : BaseModelOutput = self.model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            token_type_ids=data['token_type_ids']
        )
        hidden_state : Tensor = encoder_output.last_hidden_state[:, 0, :]
        assert hidden_state.shape == (B, E)

        vad_embeddings : Tensor = self.vad_head(self.dropout(hidden_state))
        assert vad_embeddings.shape == (B, 3)

        output = BERTForVADMappingOutput(
            vad_values=vad_embeddings
        )

        if 'labels' in data:
            labels : Tensor = data['labels']
            assert labels.shape == (B, 3)
            loss : Tensor = self.loss_criterion(vad_embeddings, labels)
            output['loss'] = loss

        return output
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path : str
    ) -> "BERTForVADMapping":
        model = cls()
        device = model.device
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
        return model
    

class StringLabelClassifier(Module):

    def __init__(self, config : DataConfig):
        super().__init__()
        self.labels = config.labels
        self.input_layer = Linear(3, config.hidden_size)
        self.output_layer = Linear(config.hidden_size, len(self.labels))
        self.dropout = Dropout(config.dropout)

    def forward(self, data : EmoticItem) -> Tensor:
        layer_1_out = self.activation(self.input_layer(data))
        layer_1_drop = self.dropout(layer_1_out)
        prediction = self.sigmoid(self.output_layer(layer_1_drop))
        return prediction
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path : str
    ) -> "StringLabelClassifier":
        pass