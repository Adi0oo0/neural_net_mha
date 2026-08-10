"""
Inference script for trained AbstractTransformer checkpoints.
Usage:
    python generate.py                          # Interactive mode
    python generate.py -p "TITLE: My Topic\nCONTENT:"  # Single prompt
    python generate.py --ckpt checkpoints/best_model.pt  # Specific checkpoint
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, sys, math, re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (Must match training exactly)
# ──────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=2)
        Q, K, V = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class AbstractTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads, dropout, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.7, top_k=50):
        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx

# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    config = ckpt["config"]
    stoi, itos = ckpt["vocab"]
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join(itos[i] for i in l)
    
    # ── Handle both key styles (original UPPERCASE vs optimized lowercase) ───
    def cfg(key_upper, key_lower):
        return config.get(key_upper, config.get(key_lower))
    
    model = AbstractTransformer(
        vocab_size=config["vocab_size"],
        d_model=cfg("EMBEDDING_DIM", "d_model"),
        n_heads=cfg("N_HEADS", "n_heads"),
        n_layers=cfg("N_LAYERS", "n_layers"),
        dropout=cfg("DROPOUT", "dropout"),
        block_size=cfg("BLOCK_SIZE", "block_size")
    ).to(device)
    
    # ── Handle both weight keys ("model_state" vs "model") ───
    state_dict = ckpt.get("model_state", ckpt.get("model"))
    if state_dict is None:
        raise KeyError("Checkpoint missing model weights (expected 'model_state' or 'model' key)")

    # ── CRITICAL FIX: Strip '_orig_mod.' prefix added by torch.compile ───
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("  → Detected torch.compile checkpoint: stripping '_orig_mod.' prefix")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    
    step = ckpt.get('step', '?')
    loss = ckpt.get('loss', '?')
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params | Step: {step} | Val Loss: {loss:.4f}")
    return model, encode, decode, cfg("BLOCK_SIZE", "block_size")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN / CLI
# ──────────────────────────────────────────────────────────────────────────────
def find_latest_ckpt(ckpt_dir):
    files = sorted(Path(ckpt_dir).glob("*.pt"))
    if not files: raise FileNotFoundError(f"No .pt files in {ckpt_dir}")
    return files[-1]

def main():
    parser = argparse.ArgumentParser(description="Generate academic abstracts")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Prompt string (use \\n for newlines)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to .pt checkpoint (default: latest in checkpoints/)")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Run interactive loop after single prompt")
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(8)

    # Resolve checkpoint
    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_ckpt("checkpoints")
    model, encode, decode, block_size = load_checkpoint(ckpt_path, device)

    # Default prompts if none provided
    default_prompts = [
        "TITLE: Deep Learning for Medical Image Segmentation\nCONTENT:",
        "TITLE: A Survey of Reinforcement Learning Methods\nCONTENT:",
        "TITLE: Quantum Computing and Cryptography\nCONTENT:",
    ]

    def run_generation(prompt_text):
        print(f"\n{'='*60}\nPROMPT:\n{prompt_text}\n{'-'*60}")
        ctx = torch.tensor([encode(prompt_text)], dtype=torch.long, device=device)
        out = model.generate(ctx, args.max_tokens, temperature=args.temp, top_k=args.top_k)[0].tolist()
        print(decode(out))
        print(f"{'='*60}\n")

    # 1. Single prompt mode
    if args.prompt:
        # Allow \n escape sequences in shell args
        prompt = args.prompt.replace("\\n", "\n")
        run_generation(prompt)
        if not args.interactive:
            return

    # 2. Interactive mode
    print("\n🧠 Interactive Generation Mode (Ctrl+C to exit)")
    print("Tip: Use format 'TITLE: Your Title\nCONTENT:' for best results.")
    print("Type 'DEFAULTS' to run the 3 built-in prompts.\n")
    
    try:
        while True:
            user_in = input("Prompt> ").strip()
            if not user_in: continue
            if user_in.upper() == "DEFAULTS":
                for p in default_prompts: run_generation(p)
                continue
            # Allow literal \n in interactive input too
            run_generation(user_in.replace("\\n", "\n"))
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")

if __name__ == "__main__":
    main()
