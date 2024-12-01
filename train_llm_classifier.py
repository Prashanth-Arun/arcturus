from system import BERTForVADMapping
from dataset.emobank import EmoBankDataset
from torch.optim import Adam
from train.vad_classifier import train
from train.config import TrainingConfig
import os
import sys

# BASE PATH
BASE_PATH = os.path.join(os.getcwd(), "artifacts")

# EXPERIMENT PARAMETERS
TASK = "train"
EXPERIMENT_NAME = "trial_lr1en4_bs64"
MODEL = "vad_classifier"

# TRAINING PARAMETERS
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
DO_CHECKPOINTING = True

if __name__ == "__main__":
    
    task_path = os.path.join(BASE_PATH, TASK)
    if not os.path.exists(task_path):
        os.mkdir(task_path)

    model_path = os.path.join(task_path, MODEL)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    exp_path = os.path.join(model_path, EXPERIMENT_NAME)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    else:
        print(f"Warning: Experiment {exp_path} already exists. Delete this folder manually and run again.")
        exit(0)

    config = TrainingConfig(
        experiment_name=EXPERIMENT_NAME,
        num_epochs=NUM_EPOCHS,
        do_checkpointing=True,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE
    )

    config_path = os.path.join(exp_path, "config.json")
    with open(config_path, "w") as f:
        f.write(config.to_json())

    # Load the model and optimizer
    model = BERTForVADMapping()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Load the datasets and loaders
    train_dataset = EmoBankDataset(split="train")
    train_loader = train_dataset.loader(batch_size=BATCH_SIZE, shuffle=True)
    validation_dataset = EmoBankDataset(split="validation")
    validation_loader = validation_dataset.loader(batch_size=BATCH_SIZE, shuffle=False)

    print(config)

    # Call the training script
    train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        num_epochs=NUM_EPOCHS,
        do_checkpointing=DO_CHECKPOINTING,
        checkpoint_base_path=exp_path
    )