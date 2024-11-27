
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from torch import Tensor, tensor
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, BertTokenizerFast
from typing import TypedDict, NotRequired, Optional
import os
import json

BASE_PATH = os.path.join(os.getcwd(), "data", "emobank")

@dataclass
class EmoBankItem(TypedDict):
    valence : float
    arousal : float
    dominance : float
    text : str

@dataclass
class EmoBankData(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    labels: list[float] 


class EmoBankDataset(Dataset):

    def __init__(
        self,
        split : str,
        base_path : str = BASE_PATH,
        limit : Optional[int] = None
    ):
        super().__init__()
        
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

        assert split in ['train', 'validation', 'test']
        file_path = os.path.join(base_path, f"{split}.json")

        with open(file_path, "r") as f:
            data = json.load(f)

        data = data[:limit] # NoneType does not affect this
        self.dataset : list[EmoBankItem] = []
        for datapoint in data:
            self.dataset.append(EmoBankItem(
                valence=datapoint['valence'],
                arousal=datapoint['arousal'],
                dominance=datapoint['dominance'],
                text=datapoint['text']
            ))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index : int) -> EmoBankData:
        datapoint : EmoBankItem = self.dataset[index]
        tokenized_sentence = self.tokenizer.encode_plus(datapoint['text'])
        labels = [datapoint['valence'], datapoint['arousal'], datapoint['dominance']]
        return EmoBankData(
            input_ids=tokenized_sentence['input_ids'],
            attention_mask=tokenized_sentence['attention_mask'],
            token_type_ids=tokenized_sentence['token_type_ids'],
            labels=labels
        )

    def loader(self, batch_size: int, shuffle: bool) -> DataLoader:
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
    

if __name__ == "__main__":
    train_dataset = EmoBankDataset(split='train', limit=10)
    train_loader = train_dataset.loader(batch_size=6, shuffle=False)
    for batch in train_loader:
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['token_type_ids'].shape)
        print(batch['labels'].shape)
        print()

# NOTE: Interesting point: we shouldn't be passing the labels as a tensor directly. Why is this the case?