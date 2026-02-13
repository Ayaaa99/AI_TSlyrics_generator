"""
Taylor Swift Lyrics Generator - Gradio Interface
A GPT-2-like transformer model that generates lyrics in Taylor Swift's style
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import re

# ============================================================================
# MODEL ARCHITECTURE (Copy from your notebook)
# ============================================================================

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# These will be loaded from checkpoint
vocab_size = None
block_size = None
n_embd = None
n_head = None
n_layer = None
dropout = None

class Head(nn.Module):
    """Single head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        scale = C ** -0.5
        wei = q @ k.transpose(-2, -1) * scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """GPT-2 style language model"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, block_special_tokens=True):
        """
        Generate text with temperature and top-k sampling
        Blocks special tokens from appearing in lyrics
        """
        self.eval()
        
        # Get special token IDs to block
        if block_special_tokens:
            special_token_ids = []
            special_tokens_list = ['<ALBUM>', '</ALBUM>', '<YEAR>', '</YEAR>', '<TITLE>', '</TITLE>']
            for token in special_tokens_list:
                try:
                    token_id = tokenizer.encode(token)[0]
                    special_token_ids.append(token_id)
                except:
                    pass
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Block special tokens after initial prompt
            if block_special_tokens and idx.size(1) > 20:
                for token_id in special_token_ids:
                    logits[:, token_id] = -float('Inf')
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path='best_model.pt'):
    """Load the trained model and tokenizer"""
    global vocab_size, block_size, n_embd, n_head, n_layer, dropout, tokenizer, model
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {
        'additional_special_tokens': [
            '<|startoftext|>', '<|endoftext|>',
            '<ALBUM>', '</ALBUM>',
            '<YEAR>', '</YEAR>',
            '<TITLE>', '</TITLE>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get configuration
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', len(tokenizer))
    block_size = config.get('block_size', 256)
    n_embd = config.get('n_embd', 256)
    n_head = config.get('n_head', 4)
    n_layer = config.get('n_layer', 4)
    dropout = config.get('dropout', 0.35)
    
    # Initialize and load model
    model = GPTLanguageModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {total_params/1e6:.2f}M")
    print(f"   Device: {device}")
    if 'val_loss' in checkpoint:
        print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
    if 'perplexity' in checkpoint:
        print(f"   Perplexity: {checkpoint['perplexity']:.2f}")
    
    return model, tokenizer


# ============================================================================
# GENERATION FUNCTION
# ============================================================================
ALBUM_YEARS = {
    "Taylor Swift": "2006",
    "Fearless": "2008",
    "Speak Now": "2010",
    "Red": "2012",
    "1989": "2014",
    "Reputation": "2017",
    "Lover": "2019",
    "Folklore": "2020",
    "Evermore": "2020",
    "Midnights": "2022"
}

def generate_lyrics(album, title, starting_line, max_length, temperature, top_k):
    """Generate Taylor Swift style lyrics"""
    year = ALBUM_YEARS.get(album, "2014")
    try:
        # Build prompt with metadata
        if starting_line.strip():
            prompt = f"<|startoftext|>\n<ALBUM>{album}</ALBUM>\n<YEAR>{year}</YEAR>\n<TITLE>{title}</TITLE>\n\n{starting_line}\n"
        else:
            prompt = f"<|startoftext|>\n<ALBUM>{album}</ALBUM>\n<YEAR>{year}</YEAR>\n<TITLE>{title}</TITLE>\n\n"
        
        # Encode prompt
        context = torch.tensor(
            tokenizer.encode(prompt),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                context,
                max_new_tokens=int(max_length),
                temperature=float(temperature),
                top_k=int(top_k) if top_k else None,
                block_special_tokens=True
            )
        
        # Decode
        output = tokenizer.decode(generated[0].tolist())
        
        # Clean up output
        if '<|endoftext|>' in output:
            output = output[:output.index('<|endoftext|>')]
        
        # Remove special tokens
        output = re.sub(r'<\|startoftext\|>', '', output)
        output = re.sub(r'</?ALBUM>', '', output)
        output = re.sub(r'</?YEAR>', '', output)
        output = re.sub(r'</?TITLE>', '', output)
        
        # Extract just the lyrics (after metadata)
        lines = output.strip().split('\n')
        lyrics_lines = []
        metadata_done = False
        for line in lines:
            if not metadata_done:
                # Skip metadata lines
                if line.strip() and not any(x in line for x in [album, year, title]):
                    metadata_done = True
                    lyrics_lines.append(line)
            else:
                lyrics_lines.append(line)
        
        result = '\n'.join(lyrics_lines).strip()
        
        # If result is too short, return full output
        if len(result) < 50:
            result = output.strip()
        
        return result
        
    except Exception as e:
        return f"Error generating lyrics: {str(e)}\n\nPlease try adjusting the parameters or check that the model is loaded correctly."


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Taylor Swift Lyrics Generator") as demo:
        gr.Markdown("""
        # 🎵 Taylor Swift Lyrics Generator
        
        Generate original lyrics in Taylor Swift's style using a GPT-2-like transformer model.
                    
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Song Details")
                
                album = gr.Dropdown(
                    choices=[
                        "Taylor Swift",
                        "Fearless",
                        "Speak Now",
                        "Red",
                        "1989",
                        "Reputation",
                        "Lover",
                        "Folklore",
                        "Evermore",
                        "Midnights"
                    ],
                    label="Album Style",
                    value="1989",
                    info="Choose which album's style to emulate"
                )
                
                title = gr.Textbox(
                    label="Song Title",
                    placeholder="Enter a song title...",
                    value="Midnight Dreams",
                    info="Give your song a title"
                )
                
                starting_line = gr.Textbox(
                    label="Starting Line (Optional)",
                    placeholder="Leave empty to generate from scratch, or provide an opening line...",
                    lines=3,
                    info="Optional: Start with a specific line"
                )
                
                gr.Markdown("### ⚙️ Generation Settings")
                
                max_length = gr.Slider(
                    minimum=100,
                    maximum=500,
                    value=300,
                    step=50,
                    label="Max Length",
                    info="Number of tokens to generate"
                )
                
                temperature = gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    value=0.8,
                    step=0.05,
                    label="Temperature",
                    info="Higher = more creative, Lower = more conservative"
                )
                
                top_k = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=40,
                    step=5,
                    label="Top-K",
                    info="Higher = more diverse vocabulary"
                )
                
                generate_btn = gr.Button("🎸 Generate Lyrics", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Generated Lyrics")
                
                output = gr.Textbox(
                    label="",
                    lines=25,
                    placeholder="Your generated lyrics will appear here...",
                    show_label=False
                )
        
        # Examples section
        gr.Markdown("### 💡 Try These Examples")
        gr.Examples(
            examples=[
                ["1989", "City Lights", "", 300, 0.8, 40],
                ["Red", "Autumn Rain", "The leaves are falling like pieces of gold", 300, 0.85, 45],
                ["Folklore", "The Last Great Memory", "", 250, 0.75, 35],
                ["Lover", "Starlight Dreams", "Dancing under the midnight sky", 300, 0.9, 50],
                ["Reputation", "Dark Paradise", "", 280, 0.8, 40],
            ],
            inputs=[album, title, starting_line, max_length, temperature, top_k],
            label="Click any example to try it"
        )
        
        # Connect button to generation function
        generate_btn.click(
            fn=generate_lyrics,
            inputs=[album, title, starting_line, max_length, temperature, top_k],
            outputs=output
        )
    
    return demo


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load model
    try:
        load_model('best_model.pt')
    except FileNotFoundError:
        print("⚠️  Model file 'best_model.pt' not found!")
        print("Please ensure your trained model is in the same directory as app.py")
        print("or specify the correct path in the load_model() function.")
        exit(1)
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",
        server_port=7860
    )
