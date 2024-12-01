from transformers import (
    BertModel, 
    BertConfig,
    AutoModelForCausalLM, 
    WhisperForConditionalGeneration
)
from dataset.emobank import EmoBankData
from dataset.emotic import EmoticData
from transformers.modeling_outputs import BaseModelOutput
from torch import Tensor
from torch.nn import Module, Linear, Dropout, MSELoss, BCELoss, ReLU, Sigmoid
from .unit import BERTForVADMappingOutput, StringLabelClassifierConfig, StringLabelClassifierOutput
import torch

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

    def __init__(self, config : StringLabelClassifierConfig):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.labels = config['labels']

        # Structure
        self.input_layer = Linear(3, config['hidden_size']).to(self.device)
        self.output_layer = Linear(config['hidden_size'], len(self.labels)).to(self.device)

        # Forward() and Trainign Components
        self.activation = ReLU()
        self.dropout = Dropout(config['dropout'])
        self.loss_criterion = BCELoss()
        self.threshold = config['threshold']
        self.sigmoid = Sigmoid()

    def forward(self, data : EmoticData) -> StringLabelClassifierOutput:


        B, _ = data['vad_values'].shape
        L = len(self.labels)

        if self.training:
            assert 'labels' in data
            assert data['labels'].shape == (B,L)

        layer_1_out = self.input_layer(data['vad_values'])
        layer_1_out = self.activation(layer_1_out)
        layer_2_out = self.output_layer(layer_1_out)
        scaled_output = self.sigmoid(layer_2_out)     
        prediction = torch.tensor(scaled_output > self.threshold, dtype=torch.float)

        output = StringLabelClassifierOutput(
            scaled_logits=scaled_output,
            predictions=prediction
        )

        if 'labels' in data:
            assert scaled_output.shape == (B, L)
            loss = self.loss_criterion(scaled_output, data['labels'])
            output['loss'] = loss

        return output
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path : str
    ) -> "StringLabelClassifier":
        model = cls()
        device = model.device
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
        return model