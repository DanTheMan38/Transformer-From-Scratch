import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import load_from_disk
import logging
from transformers import GPT2Tokenizer
from transformer_model import TransformerModel
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate(model, dataloader, device, tokenizer):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)      # Shape: [batch_size, seq_length]
            labels = batch['labels'].to(device)            # Shape: [batch_size, seq_length]

            # Transpose to match model input shape [seq_length, batch_size]
            input_ids = input_ids.transpose(0, 1)          # Shape: [seq_length, batch_size]
            labels = labels.transpose(0, 1)                # Shape: [seq_length, batch_size]

            outputs = model(input_ids)                     # Outputs shape: [seq_length, batch_size, vocab_size]

            # Reshape for loss computation
            outputs = outputs.view(-1, outputs.size(-1))   # Shape: [seq_length * batch_size, vocab_size]
            labels = labels.reshape(-1)                    # Shape: [seq_length * batch_size]

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Exclude padding tokens from total tokens count
            total_tokens += (labels != tokenizer.pad_token_id).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()

def generate_sample_outputs(model, tokenizer, device, prompts, max_length=100):
    model.eval()
    for prompt in prompts:
        logging.info(f"Prompt: {prompt}")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)  # Shape: [1, seq_length]
        input_ids = input_ids.transpose(0, 1)  # Shape: [seq_length, 1]

        generated = input_ids  # Initialize generated sequence with input_ids

        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(generated)  # Outputs shape: [seq_length, batch_size, vocab_size]
                next_token_logits = outputs[-1, 0, :]  # Logits for the last token

                # Apply temperature scaling
                temperature = 1.0  # Adjust this value as needed
                next_token_logits = next_token_logits / temperature

                # Apply top-k and nucleus (top-p) sampling
                top_k = 50  # Adjust as needed
                top_p = 0.9  # Adjust as needed

                # Filter logits using top_k
                if top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Filter logits using nucleus (top-p)
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')

                # Re-normalize probabilities
                probabilities = F.softmax(next_token_logits, dim=-1)

                # Sample from the filtered distribution
                next_token_id = torch.multinomial(probabilities, num_samples=1).unsqueeze(0)  # Shape: [1, 1]

                # Append the predicted token to the generated sequence
                generated = torch.cat((generated, next_token_id), dim=0)  # Shape: [seq_length + 1, 1]

                # Stop if the model predicts the end-of-sequence token
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

            # Convert generated tokens to a list and decode
            generated_ids = generated.squeeze(1).tolist()  # Shape: [seq_length + generated_tokens]
            generated_text = tokenizer.decode(generated_ids)
            logging.info(f"Generated Text: {generated_text}\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("Loading the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Set pad token to eos token to avoid adding new tokens to the vocabulary
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    logging.info("Loading the test dataset...")
    lm_datasets = load_from_disk("processed_data")
    test_dataset = lm_datasets['test']

    # Create DataLoader
    batch_size = 32

    def collate_fn(batch):
        input_ids = [torch.tensor(d['input_ids']) for d in batch]
        labels = [torch.tensor(d['labels']) for d in batch]

        # Pad sequences to the maximum length in the batch
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)  # Use pad_token_id to ignore in loss

        return {'input_ids': input_ids_padded, 'labels': labels_padded}

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    logging.info("Initializing the model...")
    model = TransformerModel(
        vocab_size=vocab_size,
        embed_size=512,
        num_heads=8,
        hidden_dim=2048,
        num_layers=6,
        dropout=0.1
    ).to(device)

    logging.info("Loading the model weights...")
    model.load_state_dict(torch.load('models/transformer_model.pth', map_location=device))

    logging.info("Evaluating the model...")
    avg_loss, perplexity = evaluate(model, test_dataloader, device, tokenizer)

    logging.info(f"Test Loss: {avg_loss:.4f}")
    logging.info(f"Test Perplexity: {perplexity:.4f}")

    # Generate sample outputs
    sample_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a land far away",
        "The quick brown fox jumps over",
        "In conclusion, we can say that",
        "The benefits of exercise include"
    ]

    logging.info("Generating sample outputs...")
    generate_sample_outputs(model, tokenizer, device, sample_prompts, max_length=50)

if __name__ == "__main__":
    main()