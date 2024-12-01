
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

@dataclass
class TrainingConfig(DataClassJsonMixin):
    experiment_name: str
    num_epochs: int
    do_checkpointing: bool
    learning_rate: float
    batch_size: int