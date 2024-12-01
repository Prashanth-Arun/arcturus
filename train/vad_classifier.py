from system.components import BERTForVADMapping, BERTForVADMappingOutput
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Optional
from .util import to_device

import os
import sys
import torch

VERBOSITY_INTERVAL : int = 20

def train_vad_classifier(
    model: BERTForVADMapping,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs: int,
    do_checkpointing: bool,
    checkpoint_base_path: Optional[str] = None
) -> None:
    
    for epoch in range(1, num_epochs + 1):
        print(f"========== EPOCH {epoch} ==========")

        print("Training:")
        model.train()
        for i, batch in enumerate(train_dataloader):
            batch = to_device(batch, model.device)
            output : BERTForVADMappingOutput = model(batch)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                print(f"Batch {i + 1}; Loss = {loss.item()}")

            del batch, output, loss

        if do_checkpointing:
            assert checkpoint_base_path is not None
            torch.save(model.state_dict(), os.path.join(checkpoint_base_path, f"epoch_{epoch}.pt"))

        print("\nValidation:")
        model.eval()
        for i, batch in enumerate(validation_dataloader):
            batch = to_device(batch, model.device)
            with torch.no_grad():
                output : BERTForVADMappingOutput = model(batch)
                loss = output['loss']

            print(f"Batch {i + 1}; Loss = {loss.item()}")

            del batch, output, loss
