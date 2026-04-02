"""
train_mitigator.py

Production-ready training script for the DiffusionCompiler QEM model.
Supports single-GPU, multi-GPU, and multi-node HPC clusters.
Includes dynamic scaling for Pauli, Clifford, and Universal gate sets.
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

from diffusion_compiler import CircuitVocab, DiffusionCompiler


# ══════════════════════════════════════════════════════════════════════════════
# GATE SET DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
GATE_SETS = {
    "pauli": {
        "gates": ["i", "x", "y", "z"],
        "recommended_depth": 8,  # Pauli kapıları kendi aralarında etkileşmez, derinliğe gerek yoktur.
    },
    "clifford": {
        "gates": ["i", "x", "y", "z", "h", "s", "cx"],
        "recommended_depth": 8,  # Dolanıklık (CX) ve baz değişimi (H) için orta seviye derinlik.
    },
    "universal": {
        "gates": ["i", "x", "y", "z", "h", "s", "t", "cx", "ccx", "swap"],
        "recommended_depth": 16, # Evrensel kuantum hesaplama için derin ağlar gerekir.
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── DATASET
# ══════════════════════════════════════════════════════════════════════════════

class QEMDataset(Dataset):
    """
    Dataset of (circuit_tokens, target_observables, syndromes) triplets.
    """
    def __init__(self, num_samples: int, num_qubits: int, circuit_depth: int, num_observables: int, num_syndromes: int, vocab_size: int) -> None:
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
        circuit_tokens     = torch.randint(low=0, high=self.vocab_size + 1, size=(self.num_qubits, self.circuit_depth), dtype=torch.int64)
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

def save_checkpoint(accelerator: Accelerator, model: DiffusionCompiler, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, loss: float, checkpoint_dir: Path, args: argparse.Namespace) -> None:
    if not accelerator.is_main_process:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Alt klasörleme mantığı eklendi: gate_set ismine göre kaydeder
    sub_dir = checkpoint_dir / args.gate_set
    sub_dir.mkdir(exist_ok=True)
    save_path = sub_dir / f"epoch_{epoch:04d}.pt"

    unwrapped = accelerator.unwrap_model(model)
    torch.save({
            "epoch"              : epoch,
            "loss"               : loss,
            "model_state_dict"   : unwrapped.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args"               : vars(args),
        }, save_path)
    accelerator.print(f"  [checkpoint] saved → {save_path}")

def load_checkpoint(checkpoint_path: Path, model: DiffusionCompiler, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["epoch"])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.grad_accum_steps)
    set_seed(args.seed)

    accelerator.print("=" * 60)
    accelerator.print(f"  DiffusionCompiler QEM Training")
    accelerator.print(f"  Gate Set  : {args.gate_set.upper()} ({len(args.gates)} gates: {','.join(args.gates)})")
    accelerator.print(f"  Depth     : {args.circuit_depth}")
    accelerator.print(f"  Processes : {accelerator.num_processes}")
    accelerator.print(f"  Device    : {accelerator.device}")
    accelerator.print("=" * 60)

    dataset = QEMDataset(
        num_samples     = args.num_samples,
        num_qubits      = args.num_qubits,
        circuit_depth   = args.circuit_depth,
        num_observables = args.num_observables,
        num_syndromes   = args.num_syndromes,
        vocab_size      = len(args.gates),
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    steps_per_epoch = len(dataloader) // args.grad_accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * args.epochs, eta_min=args.lr * 1e-2)

    start_epoch = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler) + 1
            accelerator.print(f"  Resumed from {resume_path} — starting epoch {start_epoch}")

    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)
    
    # BUG FIX: Scheduler to GPU
    accelerator.unwrap_model(model).scheduler.to(accelerator.device)

    checkpoint_dir = Path(args.checkpoint_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, epoch_start_time = 0.0, time.perf_counter()

        for step, batch in enumerate(dataloader):
            circuit_tokens     = batch["circuit_tokens"]
            target_observables = batch["target_observables"]
            syndromes          = batch["syndromes"]

            t = torch.randint(low=0, high=args.num_train_timesteps, size=(circuit_tokens.shape[0],), device=accelerator.device)

            with accelerator.accumulate(model):
                eps_pred, eps_true = model(circuit_tokens, target_observables, syndromes, t)
                loss = F.mse_loss(eps_pred, eps_true)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

            if args.log_every_n_steps > 0 and (step + 1) % args.log_every_n_steps == 0:
                accelerator.print(f"  epoch {epoch + 1:>4d} | step {step + 1:>5d}/{len(dataloader)} | loss {loss.item():.6f} | lr {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(dataloader)
        accelerator.print(f"Epoch {epoch + 1:>4d}/{args.epochs} | avg_loss {avg_loss:.6f} | lr {scheduler.get_last_lr()[0]:.2e} | time {time.perf_counter() - epoch_start_time:.1f}s")

        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_checkpoint(accelerator, model, optimizer, scheduler, epoch, avg_loss, checkpoint_dir, args)

    save_checkpoint(accelerator, model, optimizer, scheduler, args.epochs - 1, avg_loss, checkpoint_dir, args)
    accelerator.print("Training complete.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Gate Set Configuration (NEW) ──────────────────────────────────────────
    p.add_argument("--gate_set", type=str, default="universal", choices=["pauli", "clifford", "universal", "custom"], help="Select the mathematical gate group for mitigation.")
    p.add_argument("--gates", type=str, nargs="+", default=None, help="Custom gates if gate_set is 'custom'")
    p.add_argument("--circuit_depth", type=int, default=None, help="Override default depth for the chosen gate set.")

    # ── Training & Model ──────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    
    p.add_argument("--num_qubits", type=int, default=40)
    p.add_argument("--num_observables", type=int, default=160)
    p.add_argument("--num_syndromes", type=int, default=40)
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--context_dim", type=int, default=256)
    p.add_argument("--time_embed_dim", type=int, default=128)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--channel_mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--num_inference_steps", type=int, default=40)
    p.add_argument("--cfg_scale", type=float, default=10.0)
    p.add_argument("--noise_schedule", type=str, default="linear")

    # ── Data & Output ─────────────────────────────────────────────────────────
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    p.add_argument("--save_every_n_epochs", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # DYNAMIC CONFIGURATION BASED ON GATE SET
    if args.gate_set != "custom":
        args.gates = GATE_SETS[args.gate_set]["gates"]
        if args.circuit_depth is None:
            args.circuit_depth = GATE_SETS[args.gate_set]["recommended_depth"]
    else:
        if args.gates is None or args.circuit_depth is None:
            p.error("--gates and --circuit_depth must be provided if --gate_set is 'custom'")

    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
