"""
TrophicTrace — Physics-Informed Neural Network (PINN) for PFAS Bioaccumulation
Stage 2: Predicts fish tissue PFAS concentration from water concentration + species parameters.

The PINN is trained to satisfy the bioaccumulation ODE system (Gobas 1993):
    dC_fish/dt = k_uptake * C_water + k_diet * Σ(diet_j * C_prey_j) - (k_elim + k_growth) * C_fish

The network learns the steady-state solution AND transient dynamics.
Once trained, weights are frozen and inference is a single forward pass.

Scientific basis:
- Gobas (1993) Ecological Modelling
- Sun et al. (2022) Env Sci: Processes & Impacts
- Kelly, Sun, McDougall, Sunderland & Gobas (2024) Env Sci & Tech
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os

# Field-measured BAF values (L/kg) by trophic level — geometric means
# Source: Burkhard (2021) Environ Toxicol Chem, compiled field BAFs
# BAF = C_fish / C_water — already includes bioconcentration + dietary uptake + TMF
BAF_TABLE = {
    'PFOS':  {2.5: 500, 3.0: 1000, 3.5: 1500, 4.0: 2500, 4.5: 3500},
    'PFOA':  {2.5: 10,  3.0: 20,   3.5: 40,   4.0: 70,   4.5: 100},
    'PFNA':  {2.5: 100, 3.0: 200,  3.5: 350,  4.0: 600,  4.5: 900},
    'PFHxS': {2.5: 30,  3.0: 60,   3.5: 100,  4.0: 160,  4.5: 250},
    'PFDA':  {2.5: 150, 3.0: 300,  3.5: 500,  4.0: 850,  4.5: 1200},
    'GenX':  {2.5: 3,   3.0: 5,    3.5: 10,   4.0: 18,   4.5: 25},
}

# Legacy BCF/TMF — kept for PINN ODE physics constraint structure
BCF_BASE = {
    'PFOS': 3100, 'PFOA': 132, 'PFNA': 1200,
    'PFHxS': 316, 'PFDA': 2000, 'GenX': 40,
}

TMF = {
    'PFOS': 3.5, 'PFOA': 1.5, 'PFNA': 3.0,
    'PFHxS': 2.0, 'PFDA': 3.2, 'GenX': 1.2,
}


def get_field_baf(congener: str, trophic_level: float) -> float:
    """Interpolate field-measured BAF (L/kg) from Burkhard 2021 compiled data."""
    import math
    baf_table = BAF_TABLE[congener]
    tls = sorted(baf_table.keys())
    tl = max(tls[0], min(tls[-1], trophic_level))
    for i in range(len(tls) - 1):
        if tls[i] <= tl <= tls[i + 1]:
            frac = (tl - tls[i]) / (tls[i + 1] - tls[i])
            log_baf = math.log(baf_table[tls[i]]) * (1 - frac) + math.log(baf_table[tls[i + 1]]) * frac
            return math.exp(log_baf)
    return baf_table[tls[-1]]

# K_DOC values (L/kg) — DOC partition coefficients
K_DOC = {
    'PFOS': 1100, 'PFOA': 400, 'PFNA': 800,
    'PFHxS': 300, 'PFDA': 900, 'GenX': 200,
}

# Rate constants for ODE (per day) — from Gobas 1993, scaled for PFAS
# k_uptake: respiratory uptake rate (L water / kg fish / day)
# k_diet: dietary uptake efficiency (fraction)
# k_elim: elimination rate constant (per day)
# k_growth: growth dilution rate (per day)
RATE_CONSTANTS = {
    'k_uptake_base': 100.0,    # L/kg/day (gill uptake, allometric scaled)
    'k_diet_efficiency': 0.5,   # 50% dietary assimilation for PFAS
    'k_elim_base': 0.01,        # per day (slow for PFAS — "forever chemicals")
    'k_growth': 0.003,          # per day (~0.1% body mass/day growth)
}

REFERENCE_LIPID = 4.0
REFERENCE_TROPHIC = 3.0
CONGENER_LIST = ['PFOS', 'PFOA', 'PFNA', 'PFHxS', 'PFDA', 'GenX']


class PINNBioaccumulation(nn.Module):
    """
    Physics-Informed Neural Network for PFAS bioaccumulation.

    Inputs (8-dimensional):
        - water_pfas_ng_l: dissolved PFAS in water
        - trophic_level: species trophic level (2.0–4.5)
        - lipid_pct: species lipid content (1–10%)
        - body_mass_g: species body mass (100–5000g)
        - temperature_c: water temperature (5–30°C)
        - doc_mg_l: dissolved organic carbon (1–15 mg/L)
        - congener_idx: PFAS congener index (0–5)
        - time_days: time since exposure (0–365 days)

    Output (1-dimensional):
        - log_tissue_concentration: log(tissue ng/g) — exponentiate for actual value

    MC Dropout: dropout stays active at inference for uncertainty quantification.
    Call enable_mc_dropout() before running predict_with_ci().
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        layers = []
        in_dim = 8

        for i in range(n_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())  # Tanh for smooth gradients (needed for physics loss)
            layers.append(nn.Dropout(dropout))  # MC Dropout for uncertainty quantification
            in_dim = out_dim

        # Output is log(tissue concentration) — unbounded, covers full dynamic range
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        # Learnable physics parameters (initialized near published values)
        self.log_bcf_scale = nn.Parameter(torch.zeros(6))  # per-congener BCF scaling
        self.log_tmf_scale = nn.Parameter(torch.zeros(6))  # per-congener TMF scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Network predicts log(tissue), we return exp(log_tissue) = tissue
        log_tissue = self.network(x)
        return torch.exp(log_tissue)

    def enable_mc_dropout(self):
        """Enable dropout at inference time for MC Dropout uncertainty estimation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_mc_dropout(self):
        """Disable dropout at inference time (standard deterministic prediction)."""
        self.eval()


def generate_ode_training_data(n_samples: int = 50000) -> tuple:
    """
    Generate training data by solving the bioaccumulation ODE for many parameter combos.
    This is the "ground truth" that the PINN learns to approximate.
    """
    np.random.seed(42)

    # Sample parameter space
    water_pfas = np.random.lognormal(3, 1.5, n_samples).clip(0.1, 10000)  # ng/L
    trophic_level = np.random.uniform(2.0, 4.5, n_samples)
    lipid_pct = np.random.uniform(1.0, 10.0, n_samples)
    body_mass_g = np.random.lognormal(6, 1.0, n_samples).clip(50, 10000)
    temperature_c = np.random.uniform(5, 30, n_samples)
    doc_mg_l = np.random.lognormal(1.2, 0.5, n_samples).clip(0.5, 20)
    congener_idx = np.random.randint(0, 6, n_samples)
    time_days = np.random.choice(
        [30, 60, 90, 180, 365, 365, 365, 365],  # Bias toward steady-state (365 days)
        n_samples
    ).astype(float)

    # Solve ODE analytically for each sample using field-measured BAFs
    # BAF already includes bioconcentration + dietary uptake + trophic magnification
    # tissue_ss (ng/g) = C_water (ng/L) × BAF (L/kg) / 1000 (g/kg)
    # Transient: C(t) = C_ss × (1 - exp(-k_total × t))
    tissue = np.zeros(n_samples)

    for i in range(n_samples):
        congener = CONGENER_LIST[congener_idx[i]]

        # Field-measured BAF (already includes trophic magnification)
        baf = get_field_baf(congener, trophic_level[i])

        # Dissolved fraction correction for DOC binding
        c_dissolved = water_pfas[i] / (1 + K_DOC[congener] * doc_mg_l[i] * 1e-6)

        # Steady-state tissue concentration from field BAF
        c_steady = c_dissolved * baf / 1000  # ng/g

        # Temperature effect on elimination rate (Q10 = 2)
        temp_factor = 2.0 ** ((temperature_c[i] - 15.0) / 10.0)
        k_elim = RATE_CONSTANTS['k_elim_base'] * temp_factor

        # Transient solution: C(t) = C_ss * (1 - exp(-k_total * t))
        k_total = k_elim + RATE_CONSTANTS['k_growth']
        t = time_days[i]
        c_transient = c_steady * (1 - np.exp(-k_total * t))

        # Add realistic measurement noise (log-normal, ~20%)
        noise = np.random.lognormal(0, 0.15)
        tissue[i] = max(0.001, c_transient * noise)

    # Assemble input tensor
    inputs = np.column_stack([
        water_pfas, trophic_level, lipid_pct, body_mass_g,
        temperature_c, doc_mg_l, congener_idx.astype(float), time_days
    ])

    return inputs, tissue


def normalize_inputs(inputs: np.ndarray) -> tuple:
    """Normalize inputs to [0, 1] range for stable training."""
    mins = np.array([0.1, 2.0, 1.0, 50, 5, 0.5, 0, 0])
    maxs = np.array([10000, 4.5, 10.0, 10000, 30, 20, 5, 365])

    normalized = (inputs - mins) / (maxs - mins + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    return normalized, mins, maxs


def sample_collocation_points(n_points: int, device: torch.device) -> torch.Tensor:
    """
    Sample random points across the input domain for physics-only supervision.
    No labels needed — the ODE residual is the only loss at these points.
    Returns normalized inputs in [0, 1].
    """
    # Sample in physical space, then normalize
    water_pfas = np.random.lognormal(3, 1.5, n_points).clip(0.1, 10000)
    trophic_level = np.random.uniform(2.0, 4.5, n_points)
    lipid_pct = np.random.uniform(1.0, 10.0, n_points)
    body_mass_g = np.random.lognormal(6, 1.0, n_points).clip(50, 10000)
    temperature_c = np.random.uniform(5, 30, n_points)
    doc_mg_l = np.random.lognormal(1.2, 0.5, n_points).clip(0.5, 20)
    congener_idx = np.random.randint(0, 6, n_points).astype(float)
    time_days = np.random.uniform(1, 365, n_points)  # Uniform over time for ODE

    raw = np.column_stack([water_pfas, trophic_level, lipid_pct, body_mass_g,
                           temperature_c, doc_mg_l, congener_idx, time_days])
    normalized, _, _ = normalize_inputs(raw)
    return torch.FloatTensor(normalized).to(device), torch.FloatTensor(raw).to(device)


def compute_ode_residual(model, x_norm, x_raw, device):
    """
    Compute Gobas ODE residual: R = dC/dt - [k_uptake * C_water - (k_elim + k_growth) * C_fish]

    The network predicts C_fish(t). We use autograd to get dC/dt from the network,
    then check if it matches what the ODE says dC/dt should be.

    Returns mean squared residual.
    """
    x_norm.requires_grad_(True)

    # Forward pass
    C_pred = model(x_norm)  # predicted tissue concentration (ng/g)

    # Get dC/dt via autograd (time is input index 7)
    grad_outputs = torch.ones_like(C_pred)
    grads = torch.autograd.grad(C_pred, x_norm, grad_outputs=grad_outputs,
                                 create_graph=True, retain_graph=True)[0]
    # dC/d(t_normalized) — need to convert to dC/d(t_days) via chain rule
    # t_norm = (t - 0) / (365 - 0), so dt_days = 365 * dt_norm
    dC_dt_norm = grads[:, 7]
    dC_dt = dC_dt_norm / 365.0  # chain rule: dC/dt_days = dC/dt_norm * dt_norm/dt_days

    # Extract raw physical parameters for ODE computation
    water_pfas = x_raw[:, 0]       # ng/L
    body_mass_g = x_raw[:, 3]
    temperature_c = x_raw[:, 4]
    doc_mg_l = x_raw[:, 5]
    congener_idx = x_raw[:, 6].long()

    # Vectorized physical constants per congener
    # Build BAF lookup tensors for each trophic level bin
    baf_tl_bins = torch.tensor([2.5, 3.0, 3.5, 4.0, 4.5], device=device)
    baf_vals_by_tl = {}
    for c_idx, c_name in enumerate(CONGENER_LIST):
        baf_vals_by_tl[c_idx] = torch.tensor(
            [BAF_TABLE[c_name][tl] for tl in [2.5, 3.0, 3.5, 4.0, 4.5]],
            dtype=torch.float32, device=device
        )

    kdoc_vals = torch.tensor([K_DOC[c] for c in CONGENER_LIST], dtype=torch.float32, device=device)
    kdoc = kdoc_vals[congener_idx]

    # Dissolved fraction: C_dissolved = C_water / (1 + K_DOC * DOC * 1e-6)
    C_dissolved = water_pfas / (1 + kdoc * doc_mg_l * 1e-6)

    # Temperature-dependent elimination: Q10 = 2
    k_elim = RATE_CONSTANTS['k_elim_base'] * (2.0 ** ((temperature_c - 15.0) / 10.0))
    k_growth = RATE_CONSTANTS['k_growth']
    k_total = k_elim + k_growth

    # Compute BAF per sample via interpolation on trophic level
    trophic_level = x_raw[:, 1]
    tl_clamped = torch.clamp(trophic_level, 2.5, 4.5)

    # Vectorized log-linear BAF interpolation
    baf_per_sample = torch.zeros(len(x_raw), device=device)
    for c_idx in range(6):
        mask = (congener_idx == c_idx)
        if mask.sum() == 0:
            continue
        tl_masked = tl_clamped[mask]
        baf_tensor = baf_vals_by_tl[c_idx]  # [5] values at tl bins
        log_baf = torch.log(baf_tensor)
        # Find bin index for each sample
        bin_idx = torch.searchsorted(baf_tl_bins, tl_masked.contiguous()) - 1
        bin_idx = torch.clamp(bin_idx, 0, 3)
        frac = (tl_masked - baf_tl_bins[bin_idx]) / (baf_tl_bins[bin_idx + 1] - baf_tl_bins[bin_idx] + 1e-8)
        frac = torch.clamp(frac, 0, 1)
        interp_log_baf = log_baf[bin_idx] * (1 - frac) + log_baf[bin_idx + 1] * frac
        baf_per_sample[mask] = torch.exp(interp_log_baf)

    # Steady-state from field BAF: C_ss = C_dissolved * BAF / 1000
    C_ss = C_dissolved * baf_per_sample / 1000.0  # ng/g

    # ODE: dC/dt = k_total * (C_ss - C_fish)
    C_fish = C_pred.squeeze()
    ode_rhs = k_total * (C_ss - C_fish)

    # Residual: what the network says dC/dt is, minus what physics says it should be
    residual = dC_dt - ode_rhs

    return torch.mean(residual ** 2)


def train_pinn(n_epochs: int = 500, lr: float = 1e-3, batch_size: int = 2048,
               lambda_physics: float = 0.1, lambda_ode: float = 0.05,
               n_collocation: int = 4096, output_dir: str = '.'):
    """
    Train the PINN with three loss components:
      1. Data loss: MSE on log-transformed tissue concentrations
      2. Monotonicity loss: ∂C/∂trophic ≥ 0, ∂C/∂lipid ≥ 0, ∂C/∂time ≥ 0
      3. ODE residual loss: dC/dt must satisfy the Gobas bioaccumulation ODE
    """
    print("=" * 60)
    print("TrophicTrace — PINN Bioaccumulation Model Training")
    print("=" * 60)

    # Device selection: try CUDA (NVIDIA), then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Device: Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print(f"Device: CPU")

    # Generate training data from ODE solutions
    print("\nGenerating ODE training data (50,000 samples)...")
    inputs_raw, tissue_raw = generate_ode_training_data(50000)

    # Normalize
    inputs_norm, input_mins, input_maxs = normalize_inputs(inputs_raw)

    # Log-transform targets — network predicts in log space internally
    tissue_log = np.log(tissue_raw + 1e-6)

    # Split train/val
    n_val = 5000
    idx = np.random.permutation(len(inputs_norm))
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    X_train = torch.FloatTensor(inputs_norm[train_idx]).to(device)
    y_train_log = torch.FloatTensor(tissue_log[train_idx]).unsqueeze(1).to(device)
    y_train_raw = torch.FloatTensor(tissue_raw[train_idx]).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(inputs_norm[val_idx]).to(device)
    y_val_log = torch.FloatTensor(tissue_log[val_idx]).unsqueeze(1).to(device)
    y_val_raw = tissue_raw[val_idx]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Collocation points per batch: {n_collocation}")

    # Initialize model
    model = PINNBioaccumulation(hidden_dim=128, n_layers=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    print(f"Loss = data + {lambda_physics}×monotonicity + {lambda_ode}×ODE_residual")
    start_time = time.time()
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'physics_loss': [], 'data_loss': [],
               'ode_loss': []}

    for epoch in range(n_epochs):
        model.train()
        epoch_data_loss = 0
        epoch_physics_loss = 0
        epoch_ode_loss = 0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_log_shuffled = y_train_log[perm]

        for i in range(0, len(X_train), batch_size):
            xb = X_train_shuffled[i:i+batch_size]
            yb = y_train_log_shuffled[i:i+batch_size]

            # Need gradients for physics loss
            xb.requires_grad_(True)

            # Forward pass — model returns exp(log_pred), so pred is in original scale
            pred = model(xb)
            pred_log = torch.log(pred.clamp(min=1e-6))

            # Data loss: MSE on log-transformed values (handles huge dynamic range)
            data_loss = nn.functional.mse_loss(pred_log, yb)

            # Monotonicity physics loss
            if xb.grad is not None:
                xb.grad.zero_()
            grad_outputs = torch.ones_like(pred)
            grads = torch.autograd.grad(pred, xb, grad_outputs=grad_outputs,
                                         create_graph=True, retain_graph=True)[0]

            # Trophic level is input index 1 (after normalization)
            trophic_grad = grads[:, 1]
            monotonicity_loss = torch.mean(torch.relu(-trophic_grad) ** 2)

            # Lipid gradient should also be positive (index 2)
            lipid_grad = grads[:, 2]
            lipid_loss = torch.mean(torch.relu(-lipid_grad) ** 2)

            # Time gradient should be non-negative (index 7)
            time_grad = grads[:, 7]
            time_loss = torch.mean(torch.relu(-time_grad) ** 2)

            physics_loss = monotonicity_loss + 0.5 * lipid_loss + 0.5 * time_loss

            # ODE residual loss on collocation points (physics-only, no labels needed)
            colloc_norm, colloc_raw = sample_collocation_points(n_collocation, device)
            ode_loss = compute_ode_residual(model, colloc_norm, colloc_raw, device)

            # Total loss
            loss = data_loss + lambda_physics * physics_loss + lambda_ode * ode_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_data_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item()
            epoch_ode_loss += ode_loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_pred_log = torch.log(val_pred.clamp(min=1e-6))
            val_loss = nn.functional.mse_loss(val_pred_log, y_val_log).item()

        avg_data = epoch_data_loss / n_batches
        avg_physics = epoch_physics_loss / n_batches
        avg_ode = epoch_ode_loss / n_batches
        history['train_loss'].append(avg_data + lambda_physics * avg_physics + lambda_ode * avg_ode)
        history['val_loss'].append(val_loss)
        history['data_loss'].append(avg_data)
        history['physics_loss'].append(avg_physics)
        history['ode_loss'].append(avg_ode)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'pinn_best.pt'))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: "
                  f"data={avg_data:.4f} | mono={avg_physics:.4f} | "
                  f"ode={avg_ode:.4f} | val={val_loss:.4f}")

    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.1f}s")

    # Final validation metrics
    model.eval()
    model.load_state_dict(torch.load(os.path.join(output_dir, 'pinn_best.pt'),
                                      weights_only=True))

    with torch.no_grad():
        val_pred = model(X_val).cpu().numpy().flatten()

    # Factor accuracy
    ratio = val_pred / np.maximum(y_val_raw, 0.001)
    within_2 = np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100
    within_3 = np.mean((ratio >= 0.33) & (ratio <= 3.0)) * 100

    r2 = 1 - np.sum((val_pred - y_val_raw)**2) / np.sum((y_val_raw - y_val_raw.mean())**2)

    print(f"\n--- Validation Results ---")
    print(f"  R²: {r2:.4f}")
    print(f"  Within factor of 2: {within_2:.1f}%")
    print(f"  Within factor of 3: {within_3:.1f}%")

    # Save normalization parameters
    norm_params = {
        'input_mins': input_mins.tolist(),
        'input_maxs': input_maxs.tolist(),
    }

    # Save everything
    model_info = {
        'architecture': {
            'hidden_dim': 128,
            'n_layers': 4,
            'n_parameters': n_params,
            'activation': 'Tanh',
            'output_activation': 'Softplus',
        },
        'training': {
            'n_epochs': n_epochs,
            'learning_rate': lr,
            'batch_size': batch_size,
            'lambda_physics': lambda_physics,
            'lambda_ode': lambda_ode,
            'n_collocation_points': n_collocation,
            'dropout': 0.1,
            'train_time_seconds': round(train_time, 2),
            'device': str(device),
        },
        'validation': {
            'r2': round(float(r2), 4),
            'within_factor_2_pct': round(float(within_2), 2),
            'within_factor_3_pct': round(float(within_3), 2),
        },
        'normalization': norm_params,
        'physics_constraints': [
            'Trophic monotonicity: ∂C/∂trophic_level ≥ 0',
            'Lipid monotonicity: ∂C/∂lipid_content ≥ 0',
            'Temporal monotonicity: ∂C/∂time ≥ 0 (accumulation)',
            'ODE residual: dC/dt = k_total × (C_ss - C_fish) (Gobas 1993)',
        ],
        'uncertainty_quantification': {
            'method': 'MC Dropout (Gal & Ghahramani 2016)',
            'dropout_rate': 0.1,
            'n_forward_passes': 50,
            'output': '95% confidence interval via 2.5th/97.5th percentiles',
        },
        'training_history': {
            'final_data_loss': round(history['data_loss'][-1], 6),
            'final_physics_loss': round(history['physics_loss'][-1], 6),
            'final_ode_loss': round(history['ode_loss'][-1], 6),
            'final_val_loss': round(history['val_loss'][-1], 6),
        },
    }

    with open(os.path.join(output_dir, 'pinn_model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\nModel saved to {output_dir}/pinn_best.pt")
    print(f"Model info saved to {output_dir}/pinn_model_info.json")

    return model, model_info


def load_pinn(model_path: str = 'pinn_best.pt', info_path: str = 'pinn_model_info.json'):
    """Load frozen PINN for inference."""
    with open(info_path) as f:
        info = json.load(f)

    model = PINNBioaccumulation(
        hidden_dim=info['architecture']['hidden_dim'],
        n_layers=info['architecture']['n_layers'],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.eval()

    # Freeze all weights
    for param in model.parameters():
        param.requires_grad = False

    return model, info


def predict_tissue(model, info, water_pfas_ng_l, trophic_level, lipid_pct,
                    body_mass_g, temperature_c, doc_mg_l, congener, time_days=365):
    """
    Run frozen PINN inference for a single prediction.
    Returns tissue concentration in ng/g.
    """
    congener_idx = CONGENER_LIST.index(congener)

    inputs = np.array([[water_pfas_ng_l, trophic_level, lipid_pct, body_mass_g,
                        temperature_c, doc_mg_l, congener_idx, time_days]])

    mins = np.array(info['normalization']['input_mins'])
    maxs = np.array(info['normalization']['input_maxs'])
    normalized = (inputs - mins) / (maxs - mins + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    with torch.no_grad():
        x = torch.FloatTensor(normalized)
        pred = model(x).item()

    return pred


def predict_tissue_batch(model, info, inputs_array):
    """
    Batch PINN inference. inputs_array shape: (N, 8)
    Returns tissue concentrations (N,) in ng/g.
    """
    mins = np.array(info['normalization']['input_mins'])
    maxs = np.array(info['normalization']['input_maxs'])
    normalized = (inputs_array - mins) / (maxs - mins + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    with torch.no_grad():
        x = torch.FloatTensor(normalized)
        pred = model(x).numpy().flatten()

    return pred


def predict_with_ci(model, info, water_pfas_ng_l, trophic_level, lipid_pct,
                    body_mass_g, temperature_c, doc_mg_l, congener,
                    time_days=365, n_passes=50):
    """
    MC Dropout prediction with 95% confidence interval.

    Runs n_passes forward passes with dropout enabled, then computes
    mean (point estimate) and 2.5th/97.5th percentiles (95% CI).

    Returns: (mean, lower_95, upper_95)
    """
    congener_idx = CONGENER_LIST.index(congener)

    inputs = np.array([[water_pfas_ng_l, trophic_level, lipid_pct, body_mass_g,
                        temperature_c, doc_mg_l, congener_idx, time_days]])

    mins = np.array(info['normalization']['input_mins'])
    maxs = np.array(info['normalization']['input_maxs'])
    normalized = (inputs - mins) / (maxs - mins + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    x = torch.FloatTensor(normalized)

    # Enable MC Dropout (keep dropout active during inference)
    model.eval()
    model.enable_mc_dropout()

    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(x).item()
            predictions.append(pred)

    # Restore standard eval mode
    model.disable_mc_dropout()

    predictions = np.array(predictions)
    mean = float(np.mean(predictions))
    lower = float(np.percentile(predictions, 2.5))
    upper = float(np.percentile(predictions, 97.5))

    return mean, lower, upper


def predict_batch_with_ci(model, info, inputs_array, n_passes=50):
    """
    Batch MC Dropout prediction with 95% CI.
    inputs_array shape: (N, 8)
    Returns: (means (N,), lowers (N,), uppers (N,))
    """
    mins = np.array(info['normalization']['input_mins'])
    maxs = np.array(info['normalization']['input_maxs'])
    normalized = (inputs_array - mins) / (maxs - mins + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    x = torch.FloatTensor(normalized)

    model.eval()
    model.enable_mc_dropout()

    all_preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(x).numpy().flatten()
            all_preds.append(pred)

    model.disable_mc_dropout()

    all_preds = np.array(all_preds)  # shape: (n_passes, N)
    means = np.mean(all_preds, axis=0)
    lowers = np.percentile(all_preds, 2.5, axis=0)
    uppers = np.percentile(all_preds, 97.5, axis=0)

    return means, lowers, uppers


def accumulation_curve_with_ci(model, info, water_pfas_ng_l, trophic_level, lipid_pct,
                                body_mass_g, temperature_c, doc_mg_l, congener,
                                months=(0, 3, 6, 12, 24, 36), n_passes=50):
    """
    Generate accumulation curve with CI bands over time.
    Returns dict with 'months', 'concentration_ng_g', 'lower_95', 'upper_95'.
    """
    congener_idx = CONGENER_LIST.index(congener)
    time_points = [m * 30.44 for m in months]  # convert months to days

    # Build batch: one row per time point
    inputs = np.array([[water_pfas_ng_l, trophic_level, lipid_pct, body_mass_g,
                        temperature_c, doc_mg_l, congener_idx, t] for t in time_points])

    means, lowers, uppers = predict_batch_with_ci(model, info, inputs, n_passes)

    return {
        'months': list(months),
        'concentration_ng_g': [round(float(v), 2) for v in means],
        'lower_95': [round(float(v), 2) for v in lowers],
        'upper_95': [round(float(v), 2) for v in uppers],
    }


if __name__ == '__main__':
    model, info = train_pinn(n_epochs=500, output_dir='.')
    print("\n=== PINN Training Complete ===")

    # Demo inference with confidence intervals
    print("\n--- Demo Predictions (with 95% CI via MC Dropout) ---")
    for species_name, tl, lipid in [("Largemouth Bass", 4.2, 5.8),
                                      ("Bluegill", 3.1, 3.5),
                                      ("Striped Bass", 4.5, 6.1)]:
        for congener in ['PFOS', 'PFOA']:
            mean, lo, hi = predict_with_ci(model, info, water_pfas_ng_l=100,
                                            trophic_level=tl, lipid_pct=lipid,
                                            body_mass_g=1500, temperature_c=20,
                                            doc_mg_l=5, congener=congener)
            print(f"  {species_name} / {congener}: {mean:.2f} ng/g "
                  f"[95% CI: {lo:.2f} – {hi:.2f}] (water=100 ng/L)")

    # Demo accumulation curve
    print("\n--- Demo Accumulation Curve (Largemouth Bass / PFOS) ---")
    curve = accumulation_curve_with_ci(model, info, water_pfas_ng_l=100,
                                        trophic_level=4.2, lipid_pct=5.8,
                                        body_mass_g=1500, temperature_c=20,
                                        doc_mg_l=5, congener='PFOS')
    for m, c, lo, hi in zip(curve['months'], curve['concentration_ng_g'],
                             curve['lower_95'], curve['upper_95']):
        print(f"  Month {m:2d}: {c:.2f} ng/g [{lo:.2f} – {hi:.2f}]")
