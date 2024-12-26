from transformers import GPT2Tokenizer
import logging
import pandas as pd
import os

# Configure logging at the very beginning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data():
    logging.info("Loading the dataset...")

    # Load your CSV file
    csv_file_path = "../processed_data/dictionary_data.csv"
    data = pd.read_csv(csv_file_path)
    
    # Assuming the CSV contains 'Title' and 'Description' columns
    texts = (data['Title'] + ": " + data['Description']).tolist()

    # Initialize the tokenizer
    logging.info("Initializing the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    block_size = 128  # Adjust as needed
    tokenizer.model_max_length = block_size

    logging.info("Tokenizing the dataset...")
    tokenized_texts = []
    for text in texts:
        tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=False)
        tokenized_texts.append(tokenized["input_ids"])

    logging.info("Grouping texts into chunks for training...")
    def group_texts(tokenized_texts):
        concatenated = sum(tokenized_texts, [])
        result = {
            "input_ids": [
                concatenated[i: i + block_size]
                for i in range(0, len(concatenated), block_size)
            ]
        }
        result["labels"] = result["input_ids"]
        return result

    lm_datasets = group_texts(tokenized_texts)

    logging.info("Data preparation is complete.")
    return lm_datasets

if __name__ == "__main__":
    logging.info("Starting data preparation...")
    lm_datasets = load_and_prepare_data()
    logging.info("Saving processed data to disk...")
    # Save the tokenized data
    os.makedirs("processed_data", exist_ok=True)
    with open("processed_data/train.txt", "w", encoding="utf-8") as file:
        for chunk in lm_datasets["input_ids"]:
            file.write(" ".join(map(str, chunk)) + "\n")
    logging.info("Processed data saved successfully.")