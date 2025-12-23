import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer
import requests

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V)
        return self.fc_out(out)


class TransformerLayer(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super().__init__()
        self.attn = SimpleSelfAttention(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_layers=4, max_len=512):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_size)
        self.pos = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList([TransformerLayer(embed_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.tok_embed(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits


def prepare_next_token_batches(tokens, seq_len):
    x = tokens[:-1]
    y = tokens[1:]
    num_chunks = len(x) // seq_len
    x = x[: num_chunks * seq_len].reshape(num_chunks, seq_len)
    y = y[: num_chunks * seq_len].reshape(num_chunks, seq_len)
    return x, y


def train_model(model, x_data, y_data, epochs, batch_size, lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, x_data.size(0), batch_size):
            xb = x_data[i:i+batch_size].to(device)
            yb = y_data[i:i+batch_size].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss / (x_data.size(0)//batch_size):.4f}")

def test_saved_model(prompt="To be or not to be", max_new_tokens=100, temperature=1.2, model_path="token_transformer_model.pth"):
    """
    Load saved model and generate text until hitting a period.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load model
    model = TokenTransformer(vocab_size=len(tokenizer), embed_size=256, num_layers=4, max_len=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Tokenize prompt
    tokens = torch.tensor([tokenizer.encode(prompt)]).to(device)
    generated = tokens.clone()
    
    print(f"Prompt: '{prompt}'")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for next token (last position)
            logits = model(generated)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Decode and check for period
            new_token_str = tokenizer.decode(next_token[0])
            
            # Stop if we hit a period (common sentence ender)
            if '.' in new_token_str:
                break
    
    print()  # New line
    full_text = tokenizer.decode(generated[0])
    print(f"Full generated text: {full_text}")
    
    return full_text


def main(use_saved_model=False):
    epochs = 10
    embed_size = 256
    num_layers = 4
    seq_len = 128
    batch_size = 16
    max_len = 512

    print("Downloading Tiny Shakespeare...")
    response = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    text = response.text
    print(f"Dataset size: {len(text)} characters")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    x_data, y_data = prepare_next_token_batches(tokens, seq_len)

    print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")

    model = TokenTransformer(vocab_size=len(tokenizer), embed_size=embed_size, num_layers=num_layers, max_len=max_len)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if use_saved_model:
        model.load_state_dict(torch.load("token_transformer_model.pth"))
        print("Loaded saved model weights.")

    train_model(model, x_data, y_data, epochs, batch_size)
    torch.save(model.state_dict(), "token_transformer_model.pth")
    print("Model training complete and saved!")


if __name__ == "__main__":
    main(use_saved_model=False)
    #test_saved_model()
