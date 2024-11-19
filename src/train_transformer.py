from datasets import load_from_disk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, DataCollatorWithPadding
import logging
from transformer_model import TransformerModel
import os

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model():
    # Load the processed dataset from the 'processed_data' folder
    logging.info("Loading the processed dataset...")
    dataset = load_from_disk('processed_data')
    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    # Initialize the tokenizer and model
    logging.info("Initializing tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Assign a pad_token to the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    # Set the model_max_length
    block_size = 128  # Ensure this matches the block_size used in data_preparation.py
    tokenizer.model_max_length = block_size

    vocab_size = len(tokenizer)

    model = TransformerModel(
        vocab_size=vocab_size,
        embed_size=512,
        num_heads=8,
        hidden_dim=2048,
        num_layers=6,
        dropout=0.1
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Check if a saved model exists
    model_path = "models/transformer_model.pth"
    if os.path.exists(model_path):
        logging.info(f"Loading saved model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logging.info("No saved model found. Starting from scratch.")

    # Create data loaders using DataCollatorWithPadding
    logging.info("Creating data loaders...")
    batch_size = 16  # Adjust based on your GPU memory

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
        pad_to_multiple_of=block_size
    )

    def collate_fn(batch):
        # Remove 'labels' from batch examples before padding
        for example in batch:
            example.pop('labels', None)
        # Pad the batch
        batch_padded = data_collator(batch)
        # Set 'labels' equal to 'input_ids' shifted by one position
        input_ids = batch_padded['input_ids']
        labels = input_ids.clone()

        # Shift inputs and labels
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]

        # Update batch_padded with shifted inputs and labels
        batch_padded['input_ids'] = input_ids
        batch_padded['labels'] = labels

        return batch_padded

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    # Define optimizer and loss function
    logging.info("Setting up optimizer and loss function...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training loop
    epochs = 15  # Increased the number of epochs for better training
    logging.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_num, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].transpose(0, 1).to(device)  # Shape: [seq_length, batch_size]
            labels = batch['labels'].transpose(0, 1).to(device)        # Shape: [seq_length, batch_size]

            optimizer.zero_grad()
            outputs = model(input_ids)                                 # Shape: [seq_length, batch_size, vocab_size]

            # Reshape outputs and labels for loss computation
            outputs = outputs.view(-1, vocab_size)                     # Shape: [seq_length * batch_size, vocab_size]
            labels = labels.reshape(-1)                                # Shape: [seq_length * batch_size]

            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_num % 100 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_num}], Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_dataloader)
        logging.info(
            f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}"
        )

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].transpose(0, 1).to(device)
                labels = batch['labels'].transpose(0, 1).to(device)

                outputs = model(input_ids)
                outputs = outputs.view(-1, vocab_size)
                labels = labels.reshape(-1)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        logging.info(
            f"Validation Loss after Epoch [{epoch+1}/{epochs}]: {avg_val_loss:.4f}"
        )

        # Update the scheduler
        scheduler.step(avg_val_loss)

    # Save the trained model
    logging.info("Training complete. Saving the model...")
    torch.save(model.state_dict(), "models/transformer_model.pth")
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    train_model()