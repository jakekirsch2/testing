import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import requests

class PositionalEncoding(nn.Module):
    """Fixed positional encoding using sine/cosine functions."""
    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * 
                            -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleSelfAttention(nn.Module):
    """Single-head self-attention mechanism."""
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
    """Full transformer layer with attention + FFN + residuals."""
    def __init__(self, embed_size):
        super().__init__()
        self.attn = SimpleSelfAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class ByteEmbedding(nn.Module):
    """MLP-based embedding for byte inputs (0-255), fixed vocab size."""
    def __init__(self, embed_size, hidden_size=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )

    def forward(self, x):
        x = x.float().unsqueeze(-1) / 255.0
        return self.mlp(x)

class MultiLayerTransformer(nn.Module):
    """Multi-layer byte-level transformer model."""
    def __init__(self, embed_size=128, num_layers=4, max_len=5000):
        super().__init__()
        self.embed = ByteEmbedding(embed_size)
        self.pos = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_size) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, 256)

    def forward(self, x):
        embed = self.embed(x)
        pos_embed = self.pos(embed)
        x = pos_embed
        
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.fc_out(x)

def prepare_block_data(bytes_data, block_size=8):
    """Train ONLY to predict final block_size bytes"""
    x_data = bytes_data[:, :-block_size].contiguous()
    y_data = bytes_data[:, -block_size:].contiguous()
    return x_data, y_data

def train_model(model, x_data, y_data, epochs, batch_size, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train with block prediction - FIXED indexing."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    
    print(f"Training with x_data: {x_data.shape}, y_data: {y_data.shape}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, x_data.size(0), batch_size):
            batch_x = x_data[i:i+batch_size]
            batch_y = y_data[i:i+batch_size]
            
            optimizer.zero_grad()
            
            logits = model(batch_x)  # [B, seq_len, 256]
            
            # FIXED: Correct indexing for current batch
            block_len = batch_y.size(1)
            block_logits = logits[:, -block_len:, :]  # [B, block_len, 256]
            block_logits_flat = block_logits.reshape(-1, 256)  # [B*block_len, 256]
            targets_flat = batch_y.reshape(-1)  # [B*block_len]
            
            loss = F.cross_entropy(block_logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

def main(use_saved_model=False):
    # CONFIGURABLE VARIABLES
    epochs = 20
    embed_size = 256
    num_layers = 4
    max_len = 100
    batch_size = 32
    block_size = 8
    
    print(f"CONFIG: epochs={epochs}, batch_size={batch_size}, block_size={block_size}, max_len={max_len}")
    
    # DOWNLOAD Tiny Shakespeare
    print("Downloading Tiny Shakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    
    print(f"Downloaded {len(text)} characters")
    
    # Convert to bytes and create chunks
    bytes_data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)
    num_chunks = len(bytes_data) // max_len
    bytes_data = bytes_data[:num_chunks * max_len].reshape(num_chunks, max_len)
    
    print(f"Total chunks available: {bytes_data.shape[0]}")
    
    x_data, y_data = prepare_block_data(bytes_data, block_size)
    
    print(f"x_data shape: {x_data.shape}")
    print(f"y_data shape: {y_data.shape}")
    
    # Train model
    if use_saved_model:
        model = MultiLayerTransformer(embed_size=embed_size, num_layers=num_layers, max_len=max_len)
        model.load_state_dict(torch.load("byte_transformer_model.pth"))
        print("Loaded saved model!")
    else:
        model = MultiLayerTransformer(embed_size=embed_size, num_layers=num_layers, max_len=max_len)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    train_model(model, x_data, y_data, epochs, batch_size, lr=4e-5)
    print("Training complete!")
    
    # Save model
    torch.save(model.state_dict(), "byte_transformer_model.pth")
    print("Model saved!")


if __name__ == "__main__":
    main(use_saved_model=False)
