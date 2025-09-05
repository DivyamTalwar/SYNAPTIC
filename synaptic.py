import os
import sys
import math
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class Config:
    vocab_size: int = int(os.getenv('VOCAB_SIZE', 1024))
    hidden_dim: int = int(os.getenv('HIDDEN_DIM', 512))
    num_heads: int = int(os.getenv('NUM_HEADS', 8))
    num_layers: int = int(os.getenv('NUM_LAYERS', 4))
    max_seq_length: int = int(os.getenv('MAX_SEQ_LENGTH', 900))
    dropout: float = float(os.getenv('DROPOUT', 0.1))
    
    N_cycles: int = int(os.getenv('N_CYCLES', 2))
    T_steps: int = int(os.getenv('T_STEPS', 4))
    
    batch_size: int = int(os.getenv('BATCH_SIZE', 32))
    learning_rate: float = float(os.getenv('LEARNING_RATE', 0.001))
    weight_decay: float = float(os.getenv('WEIGHT_DECAY', 0.01))
    gradient_clip: float = float(os.getenv('GRADIENT_CLIP', 1.0))
    warmup_steps: int = int(os.getenv('WARMUP_STEPS', 1000))
    
    use_act: bool = os.getenv('USE_ACT', 'true').lower() == 'true'
    max_segments: int = int(os.getenv('MAX_SEGMENTS', 8))
    epsilon: float = float(os.getenv('EPSILON', 0.1))
    
    use_deep_supervision: bool = os.getenv('USE_DEEP_SUPERVISION', 'true').lower() == 'true'
    use_one_step_grad: bool = os.getenv('USE_ONE_STEP_GRAD', 'true').lower() == 'true'
    
    task_type: str = os.getenv('TASK_TYPE', 'sudoku')
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / norm)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_seq_length: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_length).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                        cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.rope = RotaryPositionalEncoding(self.head_dim, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(attn_output)


class GLU(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        inner_dim = config.hidden_dim * 4
        self.w1 = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.w2 = nn.Linear(inner_dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.dropout(self.w2(gate * up))


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = GLU(config)
        self.norm1 = RMSNorm(config.hidden_dim)
        self.norm2 = RMSNorm(config.hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class RecurrentModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        
    def forward(self, *inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = sum(inputs)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class HierarchicalReasoningModel(nn.Module):    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.input_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        self.L_net = RecurrentModule(config)
        self.H_net = RecurrentModule(config)
        
        self.q_head = nn.Linear(config.hidden_dim, 2, bias=False) if config.use_act else None
        
        self._init_weights()
        
        self.register_buffer('z0_L', torch.zeros(1, 1, config.hidden_dim))
        self.register_buffer('z0_H', torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.z0_L, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.z0_H, std=1.0, a=-2.0, b=2.0)
        
        self.state_history = defaultdict(list)
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
                
    def forward_segment(self, z_L: torch.Tensor, z_H: torch.Tensor, 
                    x_embed: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        N, T = self.config.N_cycles, self.config.T_steps
        
        if self.config.use_one_step_grad:
            with torch.no_grad():
                for i in range(N * T - 1):
                    z_L = self.L_net(z_L, z_H, x_embed, mask=mask)
                    if (i + 1) % T == 0:
                        z_H = self.H_net(z_H, z_L, mask=mask)
                        
            z_L = self.L_net(z_L, z_H, x_embed, mask=mask)
            z_H = self.H_net(z_H, z_L, mask=mask)
        else:
            for i in range(N * T):
                z_L = self.L_net(z_L, z_H, x_embed, mask=mask)
                if (i + 1) % T == 0:
                    z_H = self.H_net(z_H, z_L, mask=mask)
                    
        return z_L, z_H
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        x_embed = self.input_embedding(input_ids)
        z_L = self.z0_L.expand(batch_size, seq_len, -1).contiguous()
        z_H = self.z0_H.expand(batch_size, seq_len, -1).contiguous()
        
        if self.training and self.config.use_deep_supervision and labels is not None:
            return self._forward_with_deep_supervision(z_L, z_H, x_embed, labels)
        else:
            z_L, z_H = self.forward_segment(z_L, z_H, x_embed)
            logits = self.output_head(z_H)
            
            output = {"logits": logits}
            if labels is not None:
                output["loss"] = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            return output
    
    def _forward_with_deep_supervision(self, z_L: torch.Tensor, z_H: torch.Tensor,
                                    x_embed: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        total_loss = 0.0
        num_segments = 0
        
        min_segments = 1
        if self.config.use_act and np.random.rand() < self.config.epsilon:
            min_segments = np.random.randint(2, self.config.max_segments + 1)
            
        for segment in range(self.config.max_segments):
            z_L_new, z_H_new = self.forward_segment(z_L.detach(), z_H.detach(), x_embed)
            logits = self.output_head(z_H_new)
            seg_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
            if self.config.use_act and self.q_head is not None:
                q_values = torch.sigmoid(self.q_head(z_H_new.mean(dim=1)))
                
                with torch.no_grad():
                    correct = (logits.argmax(dim=-1) == labels).float().mean(dim=1)
                    q_target_halt = correct
                    
                    if segment < self.config.max_segments - 1:
                        z_L_next, z_H_next = self.forward_segment(z_L_new.detach(), z_H_new.detach(), x_embed)
                        q_values_next = torch.sigmoid(self.q_head(z_H_next.mean(dim=1)))
                        q_target_continue = q_values_next.max(dim=1)[0] if segment < self.config.max_segments - 2 else q_values_next[:, 0]
                    else:
                        q_target_continue = torch.zeros_like(q_target_halt)
                        
                    q_targets = torch.stack([q_target_halt, q_target_continue], dim=1)
                
                q_loss = F.binary_cross_entropy(q_values, q_targets)
                seg_loss = seg_loss + q_loss
                
                should_halt = (q_values[:, 0] > q_values[:, 1]) & (segment >= min_segments - 1)
                if should_halt.any() or segment >= self.config.max_segments - 1:
                    total_loss += seg_loss
                    num_segments = segment + 1
                    break
            else:
                total_loss += seg_loss
                num_segments += 1
                
            z_L, z_H = z_L_new, z_H_new
            
        return {"loss": total_loss / max(num_segments, 1), "logits": logits, "num_segments": num_segments}
    
    def compute_participation_ratio(self) -> Dict[str, float]:
        if not self.state_history['L']:
            return {"PR_L": 0.0, "PR_H": 0.0, "ratio": 0.0}
            
        def compute_pr(states: List[torch.Tensor]) -> float:
            states_tensor = torch.stack(states)
            states_flat = states_tensor.view(-1, states_tensor.shape[-1])
            cov = torch.cov(states_flat.T)
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
            return pr.item()
            
        pr_L = compute_pr(self.state_history['L']) if self.state_history['L'] else 0
        pr_H = compute_pr(self.state_history['H']) if self.state_history['H'] else 0
        
        return {"PR_L": pr_L, "PR_H": pr_H, "ratio": pr_H / pr_L if pr_L > 0 else 0}


class AdamAtan2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                    
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                denominator = corrected_exp_avg_sq.sqrt().add_(group['eps'])
                angle = torch.atan2(corrected_exp_avg.abs(), denominator)
                step_size = group['lr'] * angle.sign() * corrected_exp_avg.sign()
                
                p.data.add_(step_size)
                
        return loss


class TaskProcessor:    
    @staticmethod
    def encode_sudoku(puzzle: np.ndarray) -> torch.Tensor:
        return torch.tensor(puzzle.flatten(), dtype=torch.long)
    
    @staticmethod
    def decode_sudoku(logits: torch.Tensor) -> np.ndarray:
        return logits.argmax(dim=-1).squeeze().cpu().numpy().reshape(9, 9)
    
    @staticmethod
    def encode_maze(maze: np.ndarray) -> torch.Tensor:
        return torch.tensor(maze.flatten(), dtype=torch.long)
    
    @staticmethod
    def decode_maze(logits: torch.Tensor, shape: Tuple[int, int]) -> np.ndarray:
        path_probs = F.softmax(logits, dim=-1)[:, :, 4].squeeze().cpu().numpy()
        return (path_probs > 0.5).reshape(shape)
    
    @staticmethod
    def encode_arc(grid: np.ndarray) -> torch.Tensor:
        return torch.tensor(grid.flatten(), dtype=torch.long)
    
    @staticmethod
    def decode_arc(logits: torch.Tensor, shape: Tuple[int, int]) -> np.ndarray:
        return logits.argmax(dim=-1).squeeze().cpu().numpy().reshape(shape)


class ReasoningDataset(Dataset):
    def __init__(self, data_path: str, task_type: str, augment: bool = False):
        self.task_type = task_type
        self.augment = augment
        
        if Path(data_path).exists():
            with open(data_path) as f:
                self.data = json.load(f)
        else:
            self.data = self._generate_dummy_data(100)
            
    def _generate_dummy_data(self, n: int):
        data = []
        for _ in range(n):
            if self.task_type == 'sudoku':
                input_data = np.random.randint(0, 10, (9, 9))
                target = np.random.randint(1, 10, (9, 9))
            elif self.task_type == 'maze':
                input_data = np.random.randint(0, 4, (30, 30))
                target = np.random.randint(0, 2, (30, 30))
            else:
                input_data = np.random.randint(0, 10, (30, 30))
                target = np.random.randint(0, 10, (30, 30))
            data.append({'input': input_data.tolist(), 'target': target.tolist()})
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_array = np.array(item['input'])
        target_array = np.array(item['target'])
        
        if self.augment:
            input_array = self._augment(input_array)
            
        if self.task_type == 'sudoku':
            input_tensor = TaskProcessor.encode_sudoku(input_array)
            target_tensor = TaskProcessor.encode_sudoku(target_array)
        elif self.task_type == 'maze':
            input_tensor = TaskProcessor.encode_maze(input_array)
            target_tensor = TaskProcessor.encode_maze(target_array)
        else:
            input_tensor = TaskProcessor.encode_arc(input_array)
            target_tensor = TaskProcessor.encode_arc(target_array)
            
        return {'input_ids': input_tensor, 'labels': target_tensor}
    
    def _augment(self, data: np.ndarray) -> np.ndarray:
        if self.task_type == 'sudoku' and np.random.rand() > 0.5:
            perm = np.random.permutation(3)
            augmented = data.copy()
            for i in range(3):
                augmented[i*3:(i+1)*3] = data[perm[i]*3:(perm[i]+1)*3]
            return augmented
        elif self.task_type == 'arc':
            k = np.random.randint(4)
            return np.rot90(data, k)
        return data

class SYNAPTIC:

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.device = torch.device(self.config.device)
        
        self.model = HierarchicalReasoningModel(self.config).to(self.device)
        self.optimizer = AdamAtan2(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        print(f"SYNAPTIC initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Device: {self.device}")
        
    def train(self, train_path: str, val_path: Optional[str] = None, epochs: int = 100):
        train_dataset = ReasoningDataset(train_path, self.config.task_type, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        val_loader = None
        if val_path:
            val_dataset = ReasoningDataset(val_path, self.config.task_type, augment=False)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in train_bar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, labels)
                loss = outputs['loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                
                train_loss += loss.item()
                train_bar.set_postfix({'loss': loss.item()})
                
            avg_train_loss = train_loss / len(train_loader)
            
            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        outputs = self.model(input_ids, labels)
                        val_loss += outputs['loss'].item()
                        
                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    self.save(f"best_model_epoch_{epoch+1}.pt")
            else:
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
                
    def solve(self, problem: Union[np.ndarray, str], augmentation_rounds: int = 100) -> np.ndarray:
        self.model.eval()
        
        if isinstance(problem, str):
            problem = np.load(problem) if problem.endswith('.npy') else np.array(json.load(open(problem))['problem'])
            
        solutions = []
        
        for _ in range(augmentation_rounds):
            # Augment and encode
            if self.config.task_type == 'sudoku':
                augmented = self._augment_sudoku(problem) if augmentation_rounds > 1 else problem
                input_tensor = TaskProcessor.encode_sudoku(augmented)
            elif self.config.task_type == 'maze':
                input_tensor = TaskProcessor.encode_maze(problem)
            else:
                augmented = np.rot90(problem, np.random.randint(4)) if augmentation_rounds > 1 else problem
                input_tensor = TaskProcessor.encode_arc(augmented)
                
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
            if self.config.task_type == 'sudoku':
                solution = TaskProcessor.decode_sudoku(outputs['logits'])
            elif self.config.task_type == 'maze':
                solution = TaskProcessor.decode_maze(outputs['logits'], problem.shape)
            else:
                solution = TaskProcessor.decode_arc(outputs['logits'], problem.shape)
                
            solutions.append(solution)
            
        if augmentation_rounds > 1:
            solutions_stack = np.stack(solutions)
            final_solution = np.zeros_like(problem)
            for i in range(problem.shape[0]):
                for j in range(problem.shape[1]):
                    values, counts = np.unique(solutions_stack[:, i, j], return_counts=True)
                    final_solution[i, j] = values[np.argmax(counts)]
            return final_solution
        else:
            return solutions[0]
            
    def _augment_sudoku(self, puzzle: np.ndarray) -> np.ndarray:
        augmented = puzzle.copy()
        if np.random.rand() > 0.5:
            # Band permutation
            perm = np.random.permutation(3)
            for i in range(3):
                augmented[i*3:(i+1)*3] = puzzle[perm[i]*3:(perm[i]+1)*3]
        # Digit permutation
        digit_perm = np.random.permutation(9) + 1
        new_puzzle = np.zeros_like(augmented)
        for i in range(1, 10):
            new_puzzle[augmented == i] = digit_perm[i-1]
        return new_puzzle
        
    def analyze(self, test_problem: Optional[np.ndarray] = None):
        if test_problem is None:
            test_problem = np.random.randint(0, 10, (9, 9) if self.config.task_type == 'sudoku' else (30, 30))
            
        self.model.state_history.clear()
        self.model.eval()
        
        with torch.no_grad():
            if self.config.task_type == 'sudoku':
                input_tensor = TaskProcessor.encode_sudoku(test_problem)
            elif self.config.task_type == 'maze':
                input_tensor = TaskProcessor.encode_maze(test_problem)
            else:
                input_tensor = TaskProcessor.encode_arc(test_problem)
                
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            x_embed = self.model.input_embedding(input_tensor)
            z_L = self.model.z0_L.expand(1, input_tensor.shape[1], -1).contiguous()
            z_H = self.model.z0_H.expand(1, input_tensor.shape[1], -1).contiguous()
            
            N, T = self.config.N_cycles, self.config.T_steps
            for i in range(N * T):
                z_L = self.model.L_net(z_L, z_H, x_embed)
                self.model.state_history['L'].append(z_L.cpu())
                
                if (i + 1) % T == 0:
                    z_H = self.model.H_net(z_H, z_L)
                    self.model.state_history['H'].append(z_H.cpu())
                    
        pr_results = self.model.compute_participation_ratio()
        print(f"\nDimensionality Analysis:")
        print(f"L-Module PR: {pr_results['PR_L']:.2f}")
        print(f"H-Module PR: {pr_results['PR_H']:.2f}")
        print(f"H/L Ratio: {pr_results['ratio']:.2f}")
        
        return pr_results
        
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
        
    def benchmark(self, test_path: str) -> Dict[str, float]:
        test_dataset = ReasoningDataset(test_path, self.config.task_type, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Benchmarking"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                predictions = outputs['logits'].argmax(dim=-1)
                
                correct += (predictions == labels).float().sum().item()
                total += labels.numel()
                
        accuracy = correct / total
        print(f"Benchmark Results: Accuracy = {accuracy:.2%}")
        return {'accuracy': accuracy, 'correct': correct, 'total': total}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNAPTIC - Hierarchical Reasoning Model')
    parser.add_argument('command', choices=['train', 'solve', 'analyze', 'benchmark'])
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--val-data', type=str, help='Path to validation data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--problem', type=str, help='Path to problem file')
    parser.add_argument('--output', type=str, help='Path to save output')
    parser.add_argument('--augmentation', type=int, default=100, help='Number of augmentation rounds')
    parser.add_argument('--task', type=str, default='sudoku', choices=['sudoku', 'arc', 'maze'])
    
    args = parser.parse_args()
    
    if args.task:
        os.environ['TASK_TYPE'] = args.task
        
    config = Config()
    synaptic = SYNAPTIC(config)
    
    if args.command == 'train':
        if not args.data:
            print("Error: --data required for training")
            sys.exit(1)
        synaptic.train(args.data, args.val_data, args.epochs)
        
    elif args.command == 'solve':
        if not args.model or not args.problem:
            print("Error: --model and --problem required for solving")
            sys.exit(1)
        synaptic.load(args.model)
        solution = synaptic.solve(args.problem, args.augmentation)
        
        if args.output:
            np.save(args.output, solution)
            print(f"Solution saved to {args.output}")
        else:
            print("Solution:")
            print(solution)
            
    elif args.command == 'analyze':
        if args.model:
            synaptic.load(args.model)
        test_problem = np.load(args.problem) if args.problem else None
        synaptic.analyze(test_problem)
        
    elif args.command == 'benchmark':
        if not args.model or not args.data:
            print("Error: --model and --data required for benchmark")
            sys.exit(1)
        synaptic.load(args.model)
        synaptic.benchmark(args.data)


if __name__ == "__main__":
    main()