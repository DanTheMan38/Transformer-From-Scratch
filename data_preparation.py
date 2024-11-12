from datasets import load_dataset
from transformers import GPT2Tokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data():
    # Load the dataset
    logging.info("Loading the dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Initialize the tokenizer
    logging.info("Initializing the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Tokenize the dataset
    logging.info("Tokenizing the dataset...")
    def tokenize_function(examples):
        texts = [text if text is not None and text.strip() != "" else " " for text in examples["text"]]
        return tokenizer(texts, truncation=False, padding=False)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Group texts into chunks for training
    logging.info("Grouping texts into chunks for training...")
    block_size = 128  # Adjust as needed

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        # Instead of discarding the last chunk, we include it even if it's shorter
        result = {
            k: [concatenated_examples[k][i: i + block_size] for i in range(0, total_length, block_size)]
            for k in concatenated_examples.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True
    )

    # Log the number of examples in each split
    logging.info(f"Number of examples in train: {len(lm_datasets['train'])}")
    logging.info(f"Number of examples in validation: {len(lm_datasets['validation'])}")
    logging.info(f"Number of examples in test: {len(lm_datasets['test'])}")

    logging.info("Data preparation is complete.")
    return lm_datasets

if __name__ == "__main__":
    logging.info("Starting data preparation...")
    lm_datasets = load_and_prepare_data()
    logging.info("Saving processed data to disk...")
    lm_datasets.save_to_disk("processed_data")
    logging.info("Processed data saved successfully.")