from datasets import load_from_disk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, DataCollatorWithPadding
import logging
from transformer_model import TransformerModel

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    vocab_size = len(tokenizer)

    model = TransformerModel(vocab_size=vocab_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set block size to match your model's expected input size
    block_size = 128  # Ensure this matches the block_size used in data_preparation.py

    # Create data loaders using DataCollatorWithPadding
    logging.info("Creating data loaders...")
    batch_size = 16  # Adjust based on your GPU memory

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=block_size
    )

    def collate_fn(batch):
        # Use DataCollatorWithPadding to pad input_ids and attention_mask
        batch = data_collator(batch)
        # Set labels to be the same as input_ids
        batch['labels'] = batch['input_ids'].clone()
        return batch

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Define optimizer and loss function
    logging.info("Setting up optimizer and loss function...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 3  # Adjust the number of epochs as needed
    logging.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_num, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].transpose(0, 1).to(device)  # Shape: [seq_length, batch_size]
            labels = batch['labels'].transpose(0, 1).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            # Use reshape instead of view
            loss = criterion(outputs.reshape(-1, vocab_size), labels.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_num % 100 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_num}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].transpose(0, 1).to(device)
                labels = batch['labels'].transpose(0, 1).to(device)

                outputs = model(input_ids)
                # Use reshape instead of view
                loss = criterion(outputs.reshape(-1, vocab_size), labels.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        logging.info(f"Validation Loss after Epoch [{epoch+1}/{epochs}]: {avg_val_loss:.4f}")

    # Save the trained model
    logging.info("Training complete. Saving the model...")
    torch.save(model.state_dict(), "transformer_model.pth")
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    train_model()