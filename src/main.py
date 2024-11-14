import sys
from pathlib import Path

# Add the 'src' directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import GPT2Tokenizer
from transformer_model import TransformerModel
from fastapi.templating import Jinja2Templates

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

# Mount the static directory for serving CSS and other static files
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "static")),
    name="static",
)

# Directories
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
TEMPLATES_DIR = ROOT_DIR / "templates"
MODELS_DIR = ROOT_DIR / "models"

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load model
vocab_size = tokenizer.vocab_size
model = TransformerModel(
    vocab_size=vocab_size,
    embed_size=512,
    num_heads=8,
    hidden_dim=2048,
    num_layers=6,
    dropout=0.1,
).to(device)

model_path = MODELS_DIR / "transformer_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Set up templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate_text(request: Request):
    form = await request.form()
    prompt = form.get("prompt")
    max_length = int(form.get("max_length", 100))

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # Shape: [1, seq_length]
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

    return templates.TemplateResponse(
        "index.html", {"request": request, "result": generated_text, "prompt": prompt}
    )