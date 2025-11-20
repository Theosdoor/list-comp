import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import einops

# Import from the provided files
from model_utils import make_model, configure_runtime
from data import get_dataset

# --- Configuration ---
class SAEConfig:
    d_model = 64        # Matches D_MODEL in train.py
    d_sae = 64 * 32     # Expansion factor (e.g., 32x)
    k = 32              # Top-K active features
    lr = 3e-4
    batch_size = 4096   # Larger batch size for SAE training
    n_steps = 10_000
    
    # Model Config (Matching train.py)
    n_layers = 2
    n_heads = 1
    list_len = 2
    n_digits = 100
    sep_token_index = 2 # [d1, d2, SEP, ...] -> Index 2

# --- BatchTopK SAE Implementation ---
class BatchTopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(cfg.d_model, cfg.d_sae)
        ))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(cfg.d_sae, cfg.d_model)
        ))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))
        
        # Normalize decoder weights immediately
        self.set_decoder_norm_to_unit_norm()

    def set_decoder_norm_to_unit_norm(self):
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8)

    def encode(self, x):
        # Pre-activation: (Batch, d_sae)
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc
        
        # ReLU
        post_relu = F.relu(pre_acts)
        
        # TopK Selection
        topk_values, topk_indices = torch.topk(post_relu, k=self.cfg.k, dim=-1)
        
        # Create sparse vector
        z = torch.zeros_like(post_relu)
        z.scatter_(-1, topk_indices, topk_values)
        
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_reconstruct = self.decode(z)
        return x_reconstruct, z

# --- Helper to get SEP activations ---
def get_sep_activations(model, dataloader, layer_idx=0, max_acts=100_000):
    """
    Extracts residual stream activations specifically at the SEP token position.
    Paper implies Layer 0 writes to SEP, so we usually want hook_resid_post of Layer 0.
    """
    activations = []
    count = 0
    
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    
    print(f"Collecting activations from {hook_name} at SEP token (idx {SAEConfig.sep_token_index})...")
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(model.cfg.device)
            
            # Run model with cache
            _, cache = model.run_with_cache(inputs, stop_at_layer=layer_idx+1, names_filter=hook_name)
            
            # Extract [Batch, Seq, D_Model] -> [Batch, D_Model] at SEP index
            sep_acts = cache[hook_name][:, SAEConfig.sep_token_index, :]
            
            activations.append(sep_acts.cpu())
            count += sep_acts.shape[0]
            if count >= max_acts:
                break
    
    return torch.cat(activations, dim=0)

# --- Main Training Loop ---
def train_sae():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SAEConfig()
    
    # 1. Load or Create the Base Model (HookedTransformer)
    # Note: We assume you have a trained model checkpoint. 
    # If not, uncomment the training code in train.py to generate one.
    # Here we just initialize a fresh one for demonstration if no path provided.
    print("Initializing Base Model...")
    
    # Setup runtime config from model_utils
    configure_runtime(
        list_len=cfg.list_len, 
        seq_len=cfg.list_len*2+1, 
        vocab=cfg.n_digits+2, 
        device=device
    )
    
    model = make_model(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        ln=False, # Based on provided train.py
        use_bias=False,
        freeze_wv=True,
        freeze_wo=True
    )
    
    import os
    if os.path.exists("models/2layer_100dig_64d.pt"):
        model.load_state_dict(torch.load("models/2layer_100dig_64d.pt", map_location=device))
    else:
        print("Warning: No pretrained model found, using random initialization")
    
    # 2. Prepare Data
    train_ds, _ = get_dataset(
        list_len=cfg.list_len,
        n_digits=cfg.n_digits,
        mask_tok=cfg.n_digits,
        sep_tok=cfg.n_digits+1
    )
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
    
    # 3. Collect Buffer of Activations
    # We train the SAE on a fixed buffer of activations for speed, 
    # though you can also stream if memory is tight.
    all_acts = get_sep_activations(model, train_dl, layer_idx=0)
    all_acts = all_acts.to(device)
    
    # Normalize activations (optional but standard for SAE training)
    act_mean = all_acts.mean(0)
    all_acts = all_acts - act_mean
    # You might also normalize variance, but centering is usually sufficient
    
    sae_dl = DataLoader(all_acts, batch_size=cfg.batch_size, shuffle=True)
    
    # 4. Initialize SAE
    sae = BatchTopKSAE(cfg).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    
    # 5. Train
    print("Starting SAE Training...")
    pbar = tqdm(range(cfg.n_steps))
    
    # Create an infinite iterator
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    iter_dl = cycle(sae_dl)
    
    for step in pbar:
        batch_acts = next(iter_dl)
        
        # Forward pass
        x_reconstruct, z = sae(batch_acts)
        
        # Loss: Reconstruction MSE
        # Note: BatchTopK enforces sparsity via the algorithm, so we don't need L1 loss
        loss = F.mse_loss(x_reconstruct, batch_acts)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        # 1.0 is a standard value, but can be tuned
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)

        optimizer.step()
        
        # Normalize decoder weights (prevents shrinkage of features)
        sae.set_decoder_norm_to_unit_norm()
        
        
        if step % 100 == 0:
            pbar.set_postfix({"mse_loss": f"{loss.item():.6f}"})

    # 6. Save
    checkpoint = {
        "state_dict": sae.state_dict(),
        "cfg": vars(cfg),          # Good practice to save config too
        "act_mean": act_mean       # CRITICAL for inference
    }
    
    torch.save(checkpoint, "sep_token_sae_batch_topk.pt")
    print("SAE checkpoint saved with activation mean.")

if __name__ == "__main__":
    train_sae()

    # # Loading for analysis
    # checkpoint = torch.load("sep_token_sae_batch_topk.pt")
    # sae.load_state_dict(checkpoint["state_dict"])
    # act_mean = checkpoint["act_mean"].to(device)

    # During inference/analysis:
    # x = get_model_activations(...)
    # x_centered = x - act_mean
    # z = sae.encode(x_centered)