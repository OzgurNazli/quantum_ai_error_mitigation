"""
train_mitigator.py

Production-ready training script for the DiffusionCompiler QEM model.
Supports single-GPU, multi-GPU, and multi-node HPC clusters via
Hugging Face Accelerate — zero code changes required across all scales.

Launch examples
───────────────
Single GPU (development):
    python train_mitigator.py --epochs 100 --batch_size 64 --lr 1e-4

Multi-GPU on one node (4 GPUs):
    accelerate launch --num_processes 4 train_mitigator.py --epochs 100

Multi-node HPC (2 nodes × 8 GPUs each):
    accelerate launch --num_processes 16 --num_machines 2 \
        --machine_rank $RANK --main_process_ip $MASTER_ADDR \
        train_mitigator.py --epochs 100 --batch_size 256

Generate a reusable accelerate config once per cluster:
    accelerate config   →   saves ~/.cache/huggingface/accelerate/default_config.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.utils import set_seed

# ── Local imports ─────────────────────────────────────────────────────────────
# Adjust the import path if your package layout differs.
from diffusion_compiler import CircuitVocab, DiffusionCompiler


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── DATASET
# ══════════════════════════════════════════════════════════════════════════════

class QEMDataset(Dataset):
    """
    Dataset of (circuit_tokens, target_observables, syndromes) triplets for
    training the QEM diffusion model.

    Current implementation
    ──────────────────────
    Returns random tensors so the training loop can be developed and profiled
    independently of the real data pipeline.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  DATA LOADING PLUG-IN POINT                                             │
    │                                                                         │
    │  Replace the three `torch.rand*` calls in __getitem__ with:            │
    │                                                                         │
    │    import numpy as np                                                   │
    │    root = Path("data/raw_noisy_circuits")                               │
    │                                                                         │
    │    self.tokens = np.load(root / "circuit_tokens.npy")      # (N,Q,D)   │
    │    self.obs    = np.load(root / "target_observables.npy")  # (N,O)     │
    │    self.syn    = np.load(root / "syndromes.npy")           # (N,S)     │
    │                                                                         │
    │  Then in __getitem__:                                                   │
    │    return {                                                             │
    │        "circuit_tokens"     : torch.from_numpy(self.tokens[idx]),      │
    │        "target_observables" : torch.from_numpy(self.obs[idx]),         │
    │        "syndromes"          : torch.from_numpy(self.syn[idx]),         │
    │    }                                                                    │
    │                                                                         │
    │  Expected .npy dtypes:                                                  │
    │    circuit_tokens     → np.int64                                        │
    │    target_observables → np.float32                                      │
    │    syndromes          → np.float32                                      │
    └─────────────────────────────────────────────────────────────────────────┘

    Shape contract (per sample)
    ───────────────────────────
        circuit_tokens     : (num_qubits, circuit_depth)  [int64]
        target_observables : (num_observables,)            [float32]
        syndromes          : (num_syndromes,)              [float32]
    """

    def __init__(
        self,
        num_samples: int,
        num_qubits: int,
        circuit_depth: int,
        num_observables: int,
        num_syndromes: int,
        vocab_size: int,
    ) -> None:
        """
        Args:
            num_samples    : Total number of training examples in this dataset.
            num_qubits     : Number of physical qubits per circuit.
            circuit_depth  : Maximum gate time-steps per circuit.
            num_observables: Length of the target observable vector per sample.
            num_syndromes  : Length of the syndrome vector per sample.
            vocab_size     : Number of distinct gate tokens (excluding padding 0).
                             Token values are sampled from [0, vocab_size].
        """
        super().__init__()
        self.num_samples     = num_samples
        self.num_qubits      = num_qubits
        self.circuit_depth   = circuit_depth
        self.num_observables = num_observables
        self.num_syndromes   = num_syndromes
        self.vocab_size      = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # ── MOCK DATA ────────────────────────────────────────────────────────
        # Replace these three lines with real .npy loading (see docstring above).
        circuit_tokens     = torch.randint(
            low=0, high=self.vocab_size + 1,
            size=(self.num_qubits, self.circuit_depth),
            dtype=torch.int64,
        )
        target_observables = torch.randn(self.num_observables, dtype=torch.float32)
        syndromes          = torch.randn(self.num_syndromes,   dtype=torch.float32)
        # ─────────────────────────────────────────────────────────────────────

        return {
            "circuit_tokens"     : circuit_tokens,
            "target_observables" : target_observables,
            "syndromes"          : syndromes,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_model(args: argparse.Namespace) -> DiffusionCompiler:
    """
    Constructs and returns an untrained DiffusionCompiler.

    All hyperparameters are sourced from the parsed argparse namespace so
    that the model config is fully reproducible from the command line.

    Args:
        args: Parsed argparse namespace containing all model hyperparameters.

    Returns:
        Initialised DiffusionCompiler (parameters on CPU; Accelerate moves it).
    """
    vocab = CircuitVocab(gates=args.gates)

    model = DiffusionCompiler(
        num_qubits           = args.num_qubits,
        circuit_depth        = args.circuit_depth,
        vocab                = vocab,
        num_observables      = args.num_observables,
        num_syndromes        = args.num_syndromes,
        embed_dim            = args.embed_dim,
        context_dim          = args.context_dim,
        time_embed_dim       = args.time_embed_dim,
        base_channels        = args.base_channels,
        channel_mults        = tuple(args.channel_mults),
        num_train_timesteps  = args.num_train_timesteps,
        num_inference_steps  = args.num_inference_steps,
        cfg_scale            = args.cfg_scale,
        noise_schedule       = args.noise_schedule,
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    accelerator: Accelerator,
    model: DiffusionCompiler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> None:
    """
    Saves a full training checkpoint to `checkpoint_dir/epoch_{epoch:04d}.pt`.

    Only the main process (rank 0) performs the write so that parallel workers
    do not race each other on a shared filesystem — critical for HPC clusters
    with NFS or Lustre mounts.

    The checkpoint contains everything needed to resume training exactly:
        - model state dict (unwrapped from DDP/FSDP by Accelerate)
        - optimizer state dict
        - scheduler state dict
        - epoch index and last loss
        - the full argparse config for reproducibility

    Args:
        accelerator    : The active Accelerate instance.
        model          : The DiffusionCompiler being trained.
        optimizer      : The AdamW optimizer.
        scheduler      : The CosineAnnealingLR scheduler.
        epoch          : Current (0-indexed) epoch number.
        loss           : Average training loss for this epoch.
        checkpoint_dir : Directory in which to write the .pt file.
        args           : Full argparse namespace (saved for reproducibility).
    """
    # Guard: only rank-0 writes to disk
    if not accelerator.is_main_process:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

    # accelerator.unwrap_model strips DDP/FSDP wrappers → plain nn.Module
    unwrapped = accelerator.unwrap_model(model)

    torch.save(
        {
            "epoch"              : epoch,
            "loss"               : loss,
            "model_state_dict"   : unwrapped.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args"               : vars(args),
        },
        save_path,
    )
    accelerator.print(f"  [checkpoint] saved → {save_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: DiffusionCompiler,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    """
    Restores model (and optionally optimizer / scheduler) from a checkpoint.

    Call this *before* wrapping with Accelerate's `prepare()`.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model          : The DiffusionCompiler to load weights into.
        optimizer      : If provided, restores optimizer state as well.
        scheduler      : If provided, restores scheduler state as well.

    Returns:
        The epoch index stored in the checkpoint (useful to resume the loop).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["epoch"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    """
    Full training procedure for the DiffusionCompiler.

    Workflow
    ────────
    1.  Initialise Accelerator  →  handles device placement and distributed setup.
    2.  Build Dataset + DataLoader.
    3.  Build Model, Optimizer, Scheduler.
    4.  Optionally resume from a checkpoint.
    5.  Hand everything to accelerator.prepare()  →  DDP / FSDP wrapping, etc.
    6.  Run the epoch / batch loop.
    7.  Save checkpoints via rank-0-only write.

    Args:
        args: Parsed argparse namespace with all hyperparameters.
    """
    # ── 1. Accelerator setup ──────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,   # "no" | "fp16" | "bf16"
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with=None,                          # swap in "wandb"/"tensorboard" later
    )
    set_seed(args.seed)

    # accelerator.print only fires on rank-0 — safe to call on every process
    accelerator.print("=" * 60)
    accelerator.print(f"  DiffusionCompiler QEM Training")
    accelerator.print(f"  Processes : {accelerator.num_processes}")
    accelerator.print(f"  Device    : {accelerator.device}")
    accelerator.print(f"  Mixed prec: {args.mixed_precision}")
    accelerator.print("=" * 60)

    # ── 2. Dataset & DataLoader ───────────────────────────────────────────────
    dataset = QEMDataset(
        num_samples     = args.num_samples,
        num_qubits      = args.num_qubits,
        circuit_depth   = args.circuit_depth,
        num_observables = args.num_observables,
        num_syndromes   = args.num_syndromes,
        vocab_size      = len(args.gates),
    )

    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,   # speeds up host→GPU transfer
        drop_last   = True,   # keeps batch size uniform across all steps
    )

    # ── 3. Model, Optimizer, Scheduler ───────────────────────────────────────
    model = build_model(args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
    )

    # Steps per epoch after Accelerate splits data across GPUs
    steps_per_epoch = len(dataloader) // args.grad_accum_steps
    total_steps     = steps_per_epoch * args.epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = total_steps,   # anneal over entire training run
        eta_min = args.lr * 1e-2,
    )

    # ── 4. Optional checkpoint resume ────────────────────────────────────────
    start_epoch = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch += 1   # resume from the *next* epoch
            accelerator.print(f"  Resumed from {resume_path} — starting epoch {start_epoch}")
        else:
            accelerator.print(f"  [warn] --resume_from path not found: {resume_path}")

    # ── 5. Accelerate prepare ─────────────────────────────────────────────────
    # prepare() wraps model in DDP/FSDP, moves tensors, patches dataloader.
    # Order matters: model first, then optimizer, scheduler, dataloader.
    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, optimizer, scheduler, dataloader
    )
    # BUG FIX: Scheduler to GPU
    accelerator.unwrap_model(model).scheduler.to(accelerator.device)
    # ── 6. Epoch / batch loop ─────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train()

        epoch_loss      : float = 0.0
        epoch_start_time: float = time.perf_counter()

        for step, batch in enumerate(dataloader):
            circuit_tokens     : Tensor = batch["circuit_tokens"]      # (B, Q, D)
            target_observables : Tensor = batch["target_observables"]  # (B, O)
            syndromes          : Tensor = batch["syndromes"]           # (B, S)

            # ── Sample random diffusion timesteps uniformly ───────────────────
            # t ~ Uniform{0, …, num_train_timesteps − 1}   shape: (B,)
            t = torch.randint(
                low  = 0,
                high = args.num_train_timesteps,
                size = (circuit_tokens.shape[0],),
                device = accelerator.device,
            )

            # ── Forward pass ──────────────────────────────────────────────────
            # gradient_accumulation_context handles grad accumulation bookkeeping
            with accelerator.accumulate(model):
                eps_pred, eps_true = model(
                    circuit_tokens,
                    target_observables,
                    syndromes,
                    t,
                )

                # DDPM training objective: predict the noise that was added
                loss: Tensor = F.mse_loss(eps_pred, eps_true)

                # ── Backward & optimizer step ─────────────────────────────────
                accelerator.backward(loss)   # replaces loss.backward()

                if accelerator.sync_gradients:
                    # Clip gradients only when Accelerate is about to sync them
                    # across processes — avoids redundant clips mid-accumulation
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

            # ── Step-level logging ────────────────────────────────────────────
            if args.log_every_n_steps > 0 and (step + 1) % args.log_every_n_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"  epoch {epoch + 1:>4d} | "
                    f"step {step + 1:>5d}/{len(dataloader)} | "
                    f"loss {loss.item():.6f} | "
                    f"lr {current_lr:.2e}"
                )

        # ── Epoch-level logging ───────────────────────────────────────────────
        avg_loss   = epoch_loss / len(dataloader)
        elapsed    = time.perf_counter() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]

        accelerator.print(
            f"Epoch {epoch + 1:>4d}/{args.epochs} | "
            f"avg_loss {avg_loss:.6f} | "
            f"lr {current_lr:.2e} | "
            f"time {elapsed:.1f}s"
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        # save_checkpoint internally guards with accelerator.is_main_process
        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_checkpoint(
                accelerator    = accelerator,
                model          = model,
                optimizer      = optimizer,
                scheduler      = scheduler,
                epoch          = epoch,
                loss           = avg_loss,
                checkpoint_dir = checkpoint_dir,
                args           = args,
            )

    # ── Final checkpoint at end of training ──────────────────────────────────
    save_checkpoint(
        accelerator    = accelerator,
        model          = model,
        optimizer      = optimizer,
        scheduler      = scheduler,
        epoch          = args.epochs - 1,
        loss           = avg_loss,
        checkpoint_dir = checkpoint_dir,
        args           = args,
    )

    accelerator.print("Training complete.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    Defines and parses all command-line hyperparameters.

    Groups
    ──────
    Training     : epochs, batch_size, lr, weight_decay, grad_accum_steps, etc.
    Model        : num_qubits, circuit_depth, num_observables, num_syndromes, …
    Data         : num_samples, num_workers
    Checkpointing: checkpoint_dir, save_every_n_epochs, resume_from
    System       : seed, mixed_precision
    """
    p = argparse.ArgumentParser(
        description="Train the DiffusionCompiler QEM model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",              type=int,   default=200)
    p.add_argument("--batch_size",          type=int,   default=64,
                   help="Per-process (per-GPU) batch size.")
    p.add_argument("--lr",                  type=float, default=1e-4,
                   help="Peak learning rate for AdamW.")
    p.add_argument("--weight_decay",        type=float, default=1e-2)
    p.add_argument("--max_grad_norm",       type=float, default=1.0,
                   help="Gradient clipping norm.")
    p.add_argument("--grad_accum_steps",    type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = batch_size × steps).")
    p.add_argument("--mixed_precision",     type=str,   default="no",
                   choices=["no", "fp16", "bf16"],
                   help="Accelerate mixed precision mode. 'bf16' preferred on A100/H100.")

    # ── Model architecture ────────────────────────────────────────────────────
    p.add_argument("--num_qubits",          type=int,   default=40)
    p.add_argument("--circuit_depth",       type=int,   default=12)
    p.add_argument("--num_observables",     type=int,   default=160,
                   help="Length of target observable vector (e.g. 4 × num_qubits).")
    p.add_argument("--num_syndromes",       type=int,   default=40,
                   help="Length of QEC syndrome vector (e.g. num_qubits).")
    p.add_argument("--embed_dim",           type=int,   default=16)
    p.add_argument("--context_dim",         type=int,   default=256)
    p.add_argument("--time_embed_dim",      type=int,   default=128)
    p.add_argument("--base_channels",       type=int,   default=64)
    p.add_argument("--channel_mults",       type=int,   nargs="+", default=[1, 2, 4])
    p.add_argument("--num_train_timesteps", type=int,   default=1000)
    p.add_argument("--num_inference_steps", type=int,   default=40)
    p.add_argument("--cfg_scale",           type=float, default=10.0)
    p.add_argument("--noise_schedule",      type=str,   default="linear",
                   choices=["linear", "cosine"])
    p.add_argument("--gates",              type=str,   nargs="+",
                   default=["h", "cx", "z", "x", "ccx", "swap"],
                   help="Ordered gate vocabulary list.")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--num_samples",         type=int,   default=50_000,
                   help="Total mock dataset size (replace with real .npy row count).")
    p.add_argument("--num_workers",         type=int,   default=4,
                   help="DataLoader worker processes per GPU.")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    p.add_argument("--checkpoint_dir",      type=str,   default="checkpoints/",
                   help="Directory where .pt checkpoint files are written.")
    p.add_argument("--save_every_n_epochs", type=int,   default=10,
                   help="Write a checkpoint every this many epochs.")
    p.add_argument("--resume_from",         type=str,   default=None,
                   help="Path to a .pt checkpoint file to resume training from.")

    # ── Logging ───────────────────────────────────────────────────────────────
    p.add_argument("--log_every_n_steps",   type=int,   default=50,
                   help="Print step-level loss every N steps. 0 = disable.")

    # ── System ────────────────────────────────────────────────────────────────
    p.add_argument("--seed",                type=int,   default=42)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    train(args)
