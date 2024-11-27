
from .unit import EmoticItem, DEFAULT_EMOTIC_PATH, LABEL_TO_STRING_MAP
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import json
import os

STRING_TO_LABEL_MAP = { v : k for k, v in LABEL_TO_STRING_MAP.items() } 

class EmoticDataset(Dataset):

    def __init__(
        self, 
        split: str, 
        base_path : str = DEFAULT_EMOTIC_PATH,
        limit : Optional[int] = None
    ):
        super().__init__()

        assert split in ['train', 'validation', 'test']
        file_path = os.path.join(base_path, f"{split}.json")

        with open(file_path, "w") as f:
            data = json.load(f)

        data = data[:limit]
        self.dataset : list[EmoticItem] = []
        for item in data:
            self.dataset.append(EmoticItem(
                valence=item['valence'],
                arousal=item['arousal'],
                dominance=item['dominance'],
                labels=item['labels']
            ))

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> EmoticItem:
        return self.dataset[index]