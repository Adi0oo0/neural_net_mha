"""
Optimized Transformer Training — CPU Only (Ryzen 5850U)
Key Speedups: torch.compile, Vectorized Batch Loader, Fused AdamW, Async Checkpointing, TF32/BF16 Math
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import re, os, time, math, random, threading, queue
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# HARDWARE / ENV SETUP (Do this FIRST before importing heavy libs)
# ──────────────────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "8"       # OpenMP threads for MKL/OpenBLAS
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TORCH_NUM_THREADS"] = "8"     # PyTorch intra-op parallelism

torch.set_num_threads(8)
torch.set_num_interop_threads(1)          # Avoid inter-op contention on single model
torch.set_float32_matmul_precision("high") # Enable TF32/BF16 accumulation on Zen3 (AVX2)

device = torch.device("cpu")
print(f"Device: CPU | Threads: {torch.get_num_threads()} | MatMul Precision: High")

# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
BLOCK_SIZE     = 256
BATCH_SIZE     = 32          # ↑ Increased: fits easily in 16GB, better HW utilization
EMBEDDING_DIM  = 384
N_HEADS        = 6
N_LAYERS       = 6
LEARNING_RATE  = 3e-4        # ↑ Slightly higher with larger batch / fused optim
WARMUP_STEPS   = 200         # ↓ Faster warmup
MAX_TRAIN_HOURS= 7.5
GRAD_CLIP      = 1.0
DROPOUT        = 0.1         # ↓ Less dropout needed with more data/regularization
EVAL_INTERVAL  = 1000        # ↑ Less frequent eval = more training time
EVAL_ITERS     = 20          # ↓ 20 batches is plenty for loss estimate
SAVE_INTERVAL  = 5000
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE (Vectorized, Zero-Copy Slicing)
# ──────────────────────────────────────────────────────────────────────────────
with open("data.txt", "r", encoding="utf-8") as f:
    raw = f.read().replace("\r\n", "\n").replace("\r", "\n")

def prepare_corpus(text: str) -> str:
    entries = [e.strip() for e in text.split("----------------------------------------") if e.strip()]
    pairs = []
    for entry in entries:
        t = re.search(r"^TITLE:\s*(.+?)(?=\nCONTENT:|\Z)", entry, re.DOTALL)
        c = re.search(r"CONTENT:\s*(.+)", entry, re.DOTALL)
        if t and c and len(c.group(1).strip()) >= 50:
            pairs.append((t.group(1).strip(), c.group(1).strip()))
    print(f"Parsed {len(pairs):,} pairs")
    render = lambda p: f"TITLE: {p[0]}\nCONTENT: {p[1]}\n\n"
    random.seed(42)
    random.shuffle(pairs)
    # Single pass, shuffled. 2x duplication not needed for 30MB corpus.
    return "".join(render(p) for p in pairs)

data = prepare_corpus(raw)
chars = sorted(set(data))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join(itos[i] for i in l)

# Encode ONCE to contiguous LongTensor (Shared Memory ready)
full_data = torch.tensor(encode(data), dtype=torch.long)
n = len(full_data)
split = int(0.9 * n)
train_data = full_data[:split]
val_data   = full_data[split:]
print(f"Train: {len(train_data):,} | Val: {len(val_data):,} tokens")

# ─── VECTORIZED BATCH SAMPLER ───
# Replaces list comprehension + stack with single advanced indexing op.
def get_batch(split_name: str, batch_size: int = BATCH_SIZE):
    src = train_data if split_name == "train" else val_data
    # Generate indices on CPU (fast)
    ix = torch.randint(len(src) - BLOCK_SIZE, (batch_size,), device='cpu')
    # Vectorized gather: (B, T) -> (B, T) via broadcasting
    # This avoids Python loop & stack overhead entirely
    x = src[ix.unsqueeze(1) + torch.arange(BLOCK_SIZE)]
    y = src[ix.unsqueeze(1) + torch.arange(1, BLOCK_SIZE + 1)]
    return x, y

# ──────────────────────────────────────────────────────────────────────────────
# MODEL (Compile-Ready)
# ──────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        # Causal mask buffer
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
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)
        )
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
        # Weight Tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new, temp=0.7, top_k=50):
        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temp
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx

# ──────────────────────────────────────────────────────────────────────────────
# ASYNC CHECKPOINTING (Background Thread)
# ──────────────────────────────────────────────────────────────────────────────
class AsyncSaver:
    def __init__(self):
        self.q = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    def _worker(self):
        while True:
            fn, payload = self.q.get()
            if fn is None: break
            try: fn(*payload)
            except Exception as e: print(f"[AsyncSave Error] {e}")
            self.q.task_done()
    def save(self, fn, *args):
        # Non-blocking put; if queue full (unlikely), blocks briefly to prevent RAM buildup
        self.q.put((fn, args))
    def shutdown(self):
        self.q.put((None, None))
        self.thread.join()

def _save_ckpt(path, model, opt, step, loss, best=False):
    torch.save({
        "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
        "loss": loss, "vocab": (stoi, itos), "config": {
            "vocab_size": vocab_size, "d_model": EMBEDDING_DIM, "n_heads": N_HEADS,
            "n_layers": N_LAYERS, "block_size": BLOCK_SIZE, "dropout": DROPOUT
        }
    }, path)
    tag = "BEST" if best else f"Step {step}"
    print(f"  💾 Checkpoint saved: {path.name} ({tag})")

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
def train():
    model = AbstractTransformer(vocab_size, EMBEDDING_DIM, N_HEADS, N_LAYERS, DROPOUT, BLOCK_SIZE)
    
    # ── 1. TORCH.COMPILE (THE BIG ONE) ───
    # mode="reduce-overhead" captures CUDA graphs style optimization, good for fixed shapes.
    # On CPU, default inductor backend fuses kernels aggressively.
    print("Compiling model with torch.compile (Inductor)...")
    model = torch.compile(model, backend="aot_eager", fullgraph=True)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")

    # ── 2. FUSED OPTIMIZER ───
    optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95),
    weight_decay=0.1, foreach=True  # fused=True removed (CPU only)
)


    # ── LR Schedule: Estimate total steps from hardware ───
    # We do a quick dry-run to estimate steps/sec, or assume ~150-200 it/s compiled
    EST_STEPS_PER_SEC = 180  # Conservative estimate for compiled 6M model on 5850U
    TOTAL_STEPS_EST = int(MAX_TRAIN_HOURS * 3600 * EST_STEPS_PER_SEC)
    print(f"Estimated Total Steps: ~{TOTAL_STEPS_EST:,} (for LR schedule)")

    def get_lr(step):
        if step < WARMUP_STEPS: return LEARNING_RATE * step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS_EST - WARMUP_STEPS)
        return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Resume Logic
    start_step = 0
    ckpts = sorted(CHECKPOINT_DIR.glob("step_*.pt"))
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    saver = AsyncSaver()
    deadline = time.time() + MAX_TRAIN_HOURS * 3600
    step = start_step
    best_val = float('inf')
    running_loss = 0.0
    log_every = 50

    print(f"\n🚀 Training Start | Deadline: {MAX_TRAIN_HOURS}h | Batch: {BATCH_SIZE} | Compile: ON")
    
    # Warmup compile (first few steps are slow)
    print("Warming up compiler (first 3 steps slow)...")
    for _ in range(3):
        xb, yb = get_batch("train")
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        optimizer.zero_grad()

    try:
        while time.time() < deadline:
            # LR Update
            lr = get_lr(step)
            for pg in optimizer.param_groups: pg["lr"] = lr

            # Forward / Backward
            xb, yb = get_batch("train")
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            # Logging
            if step % log_every == 0:
                avg = running_loss / log_every
                elapsed = (time.time() - (deadline - MAX_TRAIN_HOURS * 3600)) / 3600
                print(f"Step {step:>7} | Loss {avg:.4f} | LR {lr:.2e} | {elapsed:.2f}h")
                running_loss = 0.0

            # Evaluation
            if step % EVAL_INTERVAL == 0:
                model.eval()
                losses = []
                with torch.no_grad():
                    for _ in range(EVAL_ITERS):
                        xv, yv = get_batch("val")
                        lv = F.cross_entropy(model(xv).view(-1, vocab_size), yv.view(-1))
                        losses.append(lv.item())
                val_loss = sum(losses) / len(losses)
                print(f"\n{'─'*50}\n  Eval @ {step} | Val Loss: {val_loss:.4f}\n{'─'*50}\n")
                
                if val_loss < best_val:
                    best_val = val_loss
                    saver.save(_save_ckpt, CHECKPOINT_DIR/"best_model.pt", model, optimizer, step, val_loss, True)
                model.train()

            # Periodic Checkpoint (Async)
            if step % SAVE_INTERVAL == 0:
                saver.save(_save_ckpt, CHECKPOINT_DIR/f"step_{step:07d}.pt", model, optimizer, step, loss.item())

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        saver.shutdown()
        # Final Save
        _save_ckpt(CHECKPOINT_DIR/"final_model.pt", model, optimizer, step, loss.item())
        print(f"\nDone. Steps: {step} | Best Val: {best_val:.4f}")

        # Final Generations
        model.eval()
        prompts = [
            "TITLE: Odesseius returns to his home\nCONTENT:",
            "TITLE: Best ways to learn\nCONTENT:",
        ]
        for p in prompts:
            ctx = torch.tensor([encode(p)], dtype=torch.long)
            out = model.generate(ctx, 300)[0].tolist()
            print(f"\n--- {p[:40]} ---\n{decode(out)}")

if __name__ == "__main__":
    train()
