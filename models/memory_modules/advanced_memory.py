"""
Advanced neural memory module for the Titans architecture.
Includes momentum, gradient-based surprise, and differentiated QKV mechanisms.
With per-head learning rates, momentum factors, and forget factors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, List, Union

from .memory_mlp import MemoryMLP

class AssocScan(nn.Module):
    """
    Simplified associative scan operator for memory updates.
    """
    def __init__(self, use_accelerated: bool = False):
        """
        Initialize the associative scan module.
        
        Args:
            use_accelerated: Whether to use accelerated implementation
        """
        super().__init__()
        self.use_accelerated = use_accelerated
    
    def forward(
        self, 
        gates: torch.Tensor, 
        inputs: torch.Tensor, 
        prev: Optional[torch.Tensor] = None, 
        remove_prev: bool = False
    ) -> torch.Tensor:
        """
        Simplified implementation of associative scan.
        
        Args:
            gates: Gates tensor
            inputs: Inputs tensor
            prev: Optional previous state
            remove_prev: Whether to remove previous state from output
            
        Returns:
            Scan output tensor
        """
        # Function simplified for clarity, keeping same interface
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        output = torch.zeros_like(inputs)
        
        # Calculate scan with recurrence
        state = prev.clone() if prev is not None else torch.zeros_like(inputs[:, 0:1])
        
        for i in range(seq_len):
            state = gates[:, i:i+1] * state + inputs[:, i:i+1]
            output[:, i:i+1] = state
        
        # Handle removal of prev state from output if needed
        if remove_prev and prev is not None:
            output = output[:, 1:]
            
        return output

class AdvancedNeuralMemory(nn.Module):
    """
    Advanced neural memory module with gradient-based surprise,
    momentum, and differentiated QKV capabilities.
    """
    def __init__(
        self,
        dim: int,                     # Embedding dimension
        chunk_size: int = 8,          # Size of processing chunks
        neural_memory_model: Optional[nn.Module] = None,  # Custom memory model
        memory_depth: int = 2,        # Depth if memory_model not provided
        memory_dim: Optional[int] = None,  # Inner memory dimension
        momentum: bool = True,        # Use momentum for memory updates
        momentum_order: int = 1,      # Order of momentum
        qk_rmsnorm: bool = True,      # Use normalization
        qkv_receives_diff_views: bool = True,  # Allow different views for QKV
        integration_type: str = "mal",  # "mal", "mac", or "mag"
        use_accelerated_scan: bool = False,  # Whether to use accelerated scan
        per_head_learned_parameters: bool = True,
        heads: int = 1,  # Add heads parameter with a default
        max_grad_norm: float = 10.0   # Maximum gradient norm for clipping
    ):
        """
        Initialize the advanced neural memory module.
        
        Args:
            dim: Embedding dimension
            chunk_size: Size of processing chunks
            neural_memory_model: Custom memory model (if None, MemoryMLP is created)
            memory_depth: Depth if memory_model not provided
            memory_dim: Inner memory dimension (if None, same as dim)
            momentum: Whether to use momentum for memory updates
            momentum_order: Order of momentum
            qk_rmsnorm: Whether to use normalization for query and key projections
            qkv_receives_diff_views: Allow different views for QKV
            integration_type: Integration approach ("mal", "mac", or "mag")
            use_accelerated_scan: Whether to use accelerated scan implementation
            per_head_learned_parameters: Whether to use per-head parameters
            heads: Number of attention heads
            max_grad_norm: Maximum gradient norm for clipping
        """
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim if memory_dim is not None else dim//2  # Default to half dim for efficiency
        self.chunk_size = chunk_size
        self.integration_type = integration_type
        self.momentum_enabled = momentum
        self.momentum_order = momentum_order
        self.qk_rmsnorm = qk_rmsnorm
        self.qkv_receives_diff_views = qkv_receives_diff_views
        self.per_head_learned_parameters = per_head_learned_parameters
        self.heads = heads
        self.max_grad_norm = max_grad_norm
        
        # Create dimension mapping layers
        self.down_proj = nn.Linear(dim, self.memory_dim)
        self.up_proj = nn.Linear(self.memory_dim, dim)
        
        # Create memory model if not provided
        if neural_memory_model is None:
            self.memory_model = MemoryMLP(
                dim=self.memory_dim,
                depth=memory_depth
            )
        else:
            self.memory_model = neural_memory_model
        
        # Create parameter dictionary for the memory model
        self.mem_params = nn.ParameterDict()
        for name, param in self.memory_model.named_parameters():
            # Create a flattened version of the parameter
            param_name = name.replace('.', '_')
            
            if self.per_head_learned_parameters:
                # For per-head parameters, expand with a head dimension (heads will be first dimension)
                # Initialize by repeating the same values across all heads
                heads_param = param.clone().unsqueeze(0).expand(self.heads, *param.shape)
                self.mem_params[param_name] = nn.Parameter(heads_param)
            else:
                # Original behavior - single set of parameters
                self.mem_params[param_name] = nn.Parameter(param.clone())
        
        # Projections for query/key/value
        self.to_queries = nn.Linear(self.memory_dim, self.memory_dim)
        self.to_keys = nn.Linear(self.memory_dim, self.memory_dim)
        self.to_values = nn.Linear(self.memory_dim, self.memory_dim)
        
        # Per-head memory dynamics parameters
        self.adaptive_lr = nn.Parameter(torch.ones(heads) * 0.01)
        self.momentum_factor = nn.Parameter(torch.ones(heads) * 0.9)
        self.forget_factor = nn.Parameter(torch.ones(heads) * 0.1)
        
        # Normalization
        if qk_rmsnorm:
            self.q_norm = nn.LayerNorm(self.memory_dim)
            self.k_norm = nn.LayerNorm(self.memory_dim)
        
        # For associative scan
        self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)
        
        # Memory state buffers
        self.register_buffer('prev_momentum_buffer', None, persistent=False)
        
        # Loss function for surprise computation
        self.store_memory_loss_fn = lambda pred, target: (pred - target).pow(2).mean(dim=-1)
        
        # Add block weighting for differentiated views if enabled
        if self.qkv_receives_diff_views:
            # Create parameters for block weights - use parameters directly instead of buffers
            self.q_block_weights = nn.Parameter(torch.zeros(1))  # Placeholder until set_transformer_blocks is called
            self.k_block_weights = nn.Parameter(torch.zeros(1))
            self.v_block_weights = nn.Parameter(torch.zeros(1))
            
            # Create additional projections for views
            self.q_view_proj = nn.Linear(dim, dim)
            self.k_view_proj = nn.Linear(dim, dim)
            self.v_view_proj = nn.Linear(dim, dim)
    
    def set_transformer_blocks(self, num_blocks: int) -> None:
        """
        Initialize block weights for differentiated views.
        Call this after creating the model to set up the blocks.
        
        Args:
            num_blocks: Number of transformer blocks in the model
        """
        if self.qkv_receives_diff_views:
            # Check if already initialized with proper dimensions
            current_size = self.q_block_weights.size(0)
            if current_size == num_blocks:
                # Already the right size, no need to resize
                return
                
            # Only print the resize message once when actually changing
            print(f"Resizing block weights parameters from {current_size} to {num_blocks}")
            
            # Create new parameter tensors
            device = self.q_block_weights.device
            
            q_weights = torch.zeros(num_blocks, device=device)
            k_weights = torch.zeros(num_blocks, device=device)
            v_weights = torch.zeros(num_blocks, device=device)
            
            # Initialize with reasonable priors
            with torch.no_grad():
                q_weights.normal_(0, 0.02)
                k_weights.normal_(0, 0.02)
                v_weights.normal_(0, 0.02)
                
                # Default to attending to early, middle, and late blocks
                if num_blocks >= 3:
                    q_weights[0] = 2.0              # Early block for queries
                    k_weights[num_blocks//2] = 2.0  # Middle block for keys
                    v_weights[-1] = 2.0             # Late block for values
            
            # Safely update parameters
            self.q_block_weights.data = q_weights
            self.k_block_weights.data = k_weights
            self.v_block_weights.data = v_weights
    
    def compute_surprise(
        self, 
        keys: torch.Tensor,  # [batch*heads, seq_len, memory_dim]
        values: torch.Tensor,  # [batch*heads, seq_len, memory_dim]
        head_indices: torch.Tensor  # [batch*heads]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute surprise via gradient computation with safety checks.
        
        Args:
            keys: Input keys
            values: Target values
            head_indices: Indices mapping each batch item to its head
            
        Returns:
            Dictionary of gradients for each parameter
        """
        # Check for NaN or Inf in inputs
        if torch.isnan(keys).any() or torch.isinf(keys).any() or torch.isnan(values).any() or torch.isinf(values).any():
            print("Warning: NaN or Inf detected in keys or values during surprise computation")
            # Create zero gradients as fallback
            grad_dict = {}
            for name, param in self.mem_params.items():
                grad_dict[name] = torch.zeros_like(param)
            return grad_dict
        
        # Create inputs for gradient computation
        keys_for_grad = keys.detach().clone().requires_grad_(True)
        values_detached = values.detach()
        
        # Save model state
        training_state = self.memory_model.training
        self.memory_model.train()
        
        # Save original requires_grad states
        original_requires_grad = {}
        for name, param in self.memory_model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad_(True)

        try:
            # Forward pass through memory model
            preds = self.memory_model(keys_for_grad)
            
            # Compute loss - measure the surprise
            # Add a tiny epsilon to prevent extreme values
            epsilon = 1e-8
            loss = self.store_memory_loss_fn(preds, values_detached).mean()
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("Loss is NaN or Inf in surprise computation")
            
            # Compute gradients with safety checks
            grads = torch.autograd.grad(
                loss, 
                list(self.memory_model.parameters()),
                create_graph=False,
                allow_unused=True
            )
            
            # Create gradient dictionary
            grad_dict = {}
            for (name, param), grad_val in zip(self.memory_model.named_parameters(), grads):
                param_name = name.replace('.', '_')
                if grad_val is not None:
                    # Apply gradient clipping
                    grad_norm = grad_val.norm()
                    if grad_norm > self.max_grad_norm:
                        grad_val = grad_val * (self.max_grad_norm / (grad_norm + epsilon))
                    
                    # Check for NaN gradients
                    if torch.isnan(grad_val).any() or torch.isinf(grad_val).any():
                        print(f"Warning: NaN/Inf gradient for {param_name}, using zeros")
                        grad_val = torch.zeros_like(param)
                    
                    if self.per_head_learned_parameters:
                        # Shape gradients for per-head parameters
                        # We'll need to process per head group separately
                        grad_dict[param_name] = -grad_val  # Negative for surprise
                    else:
                        # Original behavior
                        grad_dict[param_name] = -grad_val
                else:
                    # Zero gradients for params without gradients
                    if self.per_head_learned_parameters:
                        grad_dict[param_name] = torch.zeros_like(param)
                    else:
                        grad_dict[param_name] = torch.zeros_like(param)
        except Exception as e:
            print(f"Error computing surprise: {e}")
            # Fallback: create zero gradients
            grad_dict = {}
            for name, param in self.memory_model.named_parameters():
                param_name = name.replace('.', '_')
                grad_dict[param_name] = torch.zeros_like(param)
        
        finally:
            # Restore original states
            for name, param in self.memory_model.named_parameters():
                param.requires_grad_(original_requires_grad[name])
            
            # Restore training state
            self.memory_model.train(training_state)
        
        # Expand gradients to match per-head parameters if needed
        if self.per_head_learned_parameters:
            final_grad_dict = {}
            for name, grad in grad_dict.items():
                # Create a tensor with the right shape for all heads
                head_grad = torch.zeros(self.heads, *grad.shape, device=grad.device)
                
                # TODO: This is a simplified expansion that applies the same gradient to all heads
                # A more sophisticated approach would link each batch item to its specific head
                for h in range(self.heads):
                    head_grad[h] = grad
                
                final_grad_dict[name] = head_grad
            return final_grad_dict
        else:
            return grad_dict
    
    def functional_update_memory_params(
        self, 
        current_state: Dict[str, torch.Tensor],
        current_momentum: Dict[str, Optional[torch.Tensor]],
        surprise_grads: Dict[str, torch.Tensor],
        head_indices: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute new memory weights and momentum in a functional style with safety checks.
        
        Args:
            current_state: dict of current memory weights.
            current_momentum: dict of current momentum buffers.
            surprise_grads: computed surprise gradients.
            head_indices: Optional indices mapping each batch item to its head
            
        Returns:
            A tuple (new_weights, new_momentum).
        """
        new_weights = {}
        new_momentum = {}
        epsilon = 1e-8  # Small value to prevent division by zero
        
        # Clip per-head parameters to prevent instability
        adaptive_lr = torch.clamp(self.adaptive_lr, 1e-6, 0.1)
        momentum_factor = torch.clamp(self.momentum_factor, 0, 0.99)
        forget_factor = torch.clamp(self.forget_factor, 0, 0.5)
        
        for name in current_state:
            # Get the gradient, use zeros if not available
            grad = surprise_grads.get(name, torch.zeros_like(current_state[name]))
            
            # Check gradient for NaN or Inf
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"Warning: NaN/Inf gradient for {name} detected during update")
                grad = torch.zeros_like(current_state[name])
            
            # For per-head parameters, we need to handle each head separately
            if self.per_head_learned_parameters:
                new_head_weights = []
                new_head_momentum = []
                
                for h in range(self.heads):
                    # Get head-specific parameters
                    head_lr = adaptive_lr[h]
                    head_momentum = momentum_factor[h]
                    head_forget = forget_factor[h]
                    
                    head_state = current_state[name][h]
                    head_grad = grad[h] if grad.dim() > head_state.dim() else grad
                    
                    # Get or initialize momentum buffer for this head
                    if self.momentum_enabled:
                        if current_momentum.get(name) is None:
                            head_mom = head_lr * head_grad
                        else:
                            head_mom_prev = current_momentum[name][h] if current_momentum[name].dim() > head_state.dim() else current_momentum[name]
                            head_mom = head_momentum * head_mom_prev + head_lr * head_grad
                        
                        # Check momentum for NaN/Inf
                        if torch.isnan(head_mom).any() or torch.isinf(head_mom).any():
                            print(f"Warning: NaN/Inf in momentum buffer for {name}, head {h}")
                            head_mom = torch.zeros_like(head_state)
                        
                        new_head_momentum.append(head_mom)
                        update = head_mom
                    else:
                        update = head_lr * head_grad
                        new_head_momentum.append(torch.zeros_like(head_state))  # Placeholder
                    
                    # Apply weight decay/forget factor if enabled
                    head_weight = head_state * (1.0 - head_forget) - update
                    
                    # Check for NaN/Inf in new weights
                    if torch.isnan(head_weight).any() or torch.isinf(head_weight).any():
                        print(f"Warning: NaN/Inf in updated weights for {name}, head {h}")
                        # Fallback to previous weights with small decay
                        head_weight = head_state * 0.99
                    
                    new_head_weights.append(head_weight)
                
                # Stack head results back together
                new_weights[name] = torch.stack(new_head_weights)
                new_momentum[name] = torch.stack(new_head_momentum) if self.momentum_enabled else None
            else:
                # Original behavior for non-per-head parameters
                # Use average values across heads for the hyperparameters
                avg_lr = adaptive_lr.mean()
                avg_momentum = momentum_factor.mean()
                avg_forget = forget_factor.mean()
                
                if self.momentum_enabled:
                    if current_momentum.get(name) is None:
                        mom = avg_lr * grad
                    else:
                        mom = avg_momentum * current_momentum[name] + avg_lr * grad
                    
                    # Check momentum for NaN/Inf
                    if torch.isnan(mom).any() or torch.isinf(mom).any():
                        print(f"Warning: NaN/Inf in momentum buffer for {name}")
                        mom = torch.zeros_like(current_state[name])
                    
                    new_momentum[name] = mom
                    update = mom
                else:
                    update = avg_lr * grad
                    new_momentum[name] = None
                
                # Apply weight decay/forget factor if enabled
                new_weight = current_state[name] * (1.0 - avg_forget) - update
                
                # Check for NaN/Inf in new weights
                if torch.isnan(new_weight).any() or torch.isinf(new_weight).any():
                    print(f"Warning: NaN/Inf in updated weights for {name}")
                    # Fallback to previous weights with small decay
                    new_weight = current_state[name] * 0.99
                
                new_weights[name] = new_weight
        
        return new_weights, new_momentum

    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[Dict[str, Any]] = None,
        block_outputs: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with gradient-based memory updates in a functional, state-passing manner.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            state: Optional previous state; if None, a new state is initialized.
            block_outputs: Optional list of transformer block outputs for differentiated views.
            
        Returns:
            Tuple of (output tensor, new state dict)
        """
        batch_size, seq_len, _ = x.shape
        
        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in input")
            # Replace problematic values with zeros
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)

        # Handle differentiated block views if available
        if self.qkv_receives_diff_views and block_outputs is not None:
            return self.forward_with_diff_views(x, block_outputs, state)

        # Standard forward pass: project input to memory dimension
        x_mem = self.down_proj(x)
        
        # Generate keys/values
        keys = self.to_keys(x_mem)
        values = self.to_values(x_mem)
        
        # Apply normalization if enabled
        if self.qk_rmsnorm:
            keys = self.k_norm(keys)

        # Prepare head indices for multi-head processing
        # This assigns batch items to heads (for example, round-robin)
        head_indices = torch.arange(batch_size, device=keys.device) % self.heads
        
        # Compute surprise gradients
        with torch.set_grad_enabled(True):
            surprise_grads = self.compute_surprise(keys, values, head_indices)

        # Initialize state if not provided
        if state is None:
            current_state = {k: v.clone() for k, v in self.mem_params.items()}
            current_momentum = {k: None for k in self.mem_params.keys()}
            seq_index = seq_len
        else:
            current_state = state.get("weights", {k: v.clone() for k, v in self.mem_params.items()})
            current_momentum = state.get("momentum", {k: None for k in self.mem_params.keys()})
            seq_index = state.get("seq_index", 0) + seq_len

        # Compute new state functionally with per-head parameters
        new_weights, new_momentum = self.functional_update_memory_params(
            current_state, current_momentum, surprise_grads, head_indices
        )

        # Create new state dictionary
        new_state = {
            "seq_index": seq_index,
            "weights": new_weights,
            "momentum": new_momentum
        }

        # Update memory model parameters from the new state for retrieval
        with torch.no_grad():
            for name in self.mem_params:
                if name in new_weights:
                    self.mem_params[name].data.copy_(new_weights[name])
            
            # Update memory model parameters for retrieval
            for model_name, model_param in self.memory_model.named_parameters():
                param_name = model_name.replace(".", "_")
                if param_name in self.mem_params:
                    if self.per_head_learned_parameters:
                        # Average across heads for the model parameters
                        aggregated = self.mem_params[param_name].mean(dim=0)
                        model_param.data.copy_(aggregated)
                    else:
                        model_param.data.copy_(self.mem_params[param_name])

        # Generate queries for memory retrieval
        queries = self.to_queries(x_mem)
        if self.qk_rmsnorm:
            queries = self.q_norm(queries)
        
        # Process through memory model
        try:
            memory_output = self.memory_model(queries)
            
            # Check for NaN/Inf in output
            if torch.isnan(memory_output).any() or torch.isinf(memory_output).any():
                print("Warning: NaN or Inf detected in memory output")
                memory_output = torch.where(
                    torch.isnan(memory_output) | torch.isinf(memory_output), 
                    queries, 
                    memory_output
                )
                
            # Project back to original dimension
            output = self.up_proj(memory_output)
            
        except Exception as e:
            print(f"Error in memory model forward pass: {e}")
            # Fallback: bypass memory and return projected input
            output = self.up_proj(x_mem)

        return output, new_state
        
    def forward_with_diff_views(
        self, 
        x: torch.Tensor, 
        block_outputs: List[torch.Tensor],
        state: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass using differentiated block outputs with per-head parameters.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            block_outputs: List of outputs from transformer blocks
            state: Optional previous state
            
        Returns:
            Tuple of (output tensor, new state dict)
        """
        batch_size, seq_len, _ = x.shape

        # Cache the number of blocks to avoid constant resizing
        num_blocks = len(block_outputs)
        
        # Only resize once during the entire forward pass if needed
        if self.q_block_weights.size(0) != num_blocks:
            self.set_transformer_blocks(num_blocks)

        try:
            # Normalize block view weights - use softmax on the correct dimension
            q_weights = F.softmax(self.q_block_weights, dim=0)
            k_weights = F.softmax(self.k_block_weights, dim=0)
            v_weights = F.softmax(self.v_block_weights, dim=0)

            # Check for NaN in block outputs
            for i, block_out in enumerate(block_outputs):
                if torch.isnan(block_out).any() or torch.isinf(block_out).any():
                    print(f"Warning: NaN/Inf in block output {i}")
                    block_outputs[i] = torch.zeros_like(block_out)

            # Create weighted views for each head
            q_views = []
            k_views = []
            v_views = []
            
            for h in range(self.heads):
                # Create weighted view for this head
                q_view = torch.zeros_like(x)
                k_view = torch.zeros_like(x)
                v_view = torch.zeros_like(x)
                
                for i, block_out in enumerate(block_outputs):
                    q_view += q_weights[i] * block_out
                    k_view += k_weights[i] * block_out
                    v_view += v_weights[i] * block_out
                
                q_views.append(q_view)
                k_views.append(k_view)
                v_views.append(v_view)
            
            # Stack and flatten heads into batch dimension
            q_view = torch.cat(q_views, dim=0)  # [heads*batch, seq_len, dim]
            k_view = torch.cat(k_views, dim=0)
            v_view = torch.cat(v_views, dim=0)
            
            # Project to memory dimension
            q_mem = self.down_proj(q_view)
            k_mem = self.down_proj(k_view)
            v_mem = self.down_proj(v_view)
            
            # Generate projections
            queries = self.to_queries(q_mem)
            keys = self.to_keys(k_mem)
            values = self.to_values(v_mem)
            
            # Apply normalization if enabled
            if self.qk_rmsnorm:
                queries = self.q_norm(queries)
                keys = self.k_norm(keys)
            
            # Prepare head indices
            head_indices = torch.arange(self.heads, device=keys.device).repeat_interleave(batch_size)
            
            # Compute surprise gradients
            with torch.set_grad_enabled(True):
                surprise_grads = self.compute_surprise(keys, values, head_indices)
                
            # Initialize or retrieve the current state
            if state is None:
                current_state = {k: v.clone() for k, v in self.mem_params.items()}
                current_momentum = {k: None for k in self.mem_params.keys()}
                seq_index = seq_len
            else:
                current_state = state.get("weights", {k: v.clone() for k, v in self.mem_params.items()})
                current_momentum = state.get("momentum", {k: None for k in self.mem_params.keys()})
                seq_index = state.get("seq_index", 0) + seq_len
            
            # Compute new state functionally with per-head parameters
            new_weights, new_momentum = self.functional_update_memory_params(
                current_state, current_momentum, surprise_grads, head_indices
            )
            
            new_state = {
                "seq_index": seq_index,
                "weights": new_weights,
                "momentum": new_momentum
            }
            
            # Update memory model parameters
            with torch.no_grad():
                for name in self.mem_params:
                    if name in new_weights:
                        self.mem_params[name].data.copy_(new_weights[name])
                        
                for model_name, model_param in self.memory_model.named_parameters():
                    param_name = model_name.replace(".", "_")
                    if param_name in self.mem_params:
                        if self.per_head_learned_parameters:
                            aggregated = self.mem_params[param_name].mean(dim=0)
                            model_param.data.copy_(aggregated)
                        else:
                            model_param.data.copy_(self.mem_params[param_name])
            
            # Process queries through memory model for each head
            memory_outputs = []
            for h in range(self.heads):
                head_batch = batch_size
                head_queries = queries[h * head_batch:(h + 1) * head_batch]
                try:
                    memory_out = self.memory_model(head_queries)
                    # Check for NaN/Inf
                    if torch.isnan(memory_out).any() or torch.isinf(memory_out).any():
                        memory_out = head_queries  # Use queries as fallback
                    memory_outputs.append(memory_out)
                except Exception as e:
                    print(f"Error in memory model forward pass for head {h}: {e}")
                    memory_outputs.append(head_queries)  # Use queries as fallback
            
            # Reshape the memory outputs to group by head, then average
            all_outputs = torch.cat(memory_outputs, dim=0)
            all_outputs = all_outputs.reshape(self.heads, batch_size, seq_len, -1)
            memory_output = all_outputs.mean(dim=0)  # Average over heads
            
            # Project back to original dimension
            output = self.up_proj(memory_output)
            
        except Exception as e:
            print(f"Error in forward_with_diff_views: {e}")
            # Fallback to standard forward pass
            return self.forward(x, state)
        
        return output, new_state
    
    def get_memory_tokens(
        self, 
        num_tokens: int = 4, 
        block_outputs: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        For MAC integration - get memory tokens to be used as context.
        
        Args:
            num_tokens: Number of tokens to return
            block_outputs: Optional list of transformer block outputs for differentiated views
            
        Returns:
            Memory tokens of shape [batch_size, num_tokens, dim]
        """
        device = next(self.parameters()).device
        
        try:
            if self.qkv_receives_diff_views and block_outputs is not None and len(block_outputs) > 0:
                # Ensure block weights parameters are properly sized - only once
                num_blocks = len(block_outputs)
                if self.q_block_weights.size(0) != num_blocks:
                    self.set_transformer_blocks(num_blocks)
                
                # Create differentiated views for each head
                q_weights = F.softmax(self.q_block_weights, dim=0)
                
                # Create per-head weighted views
                head_views = []
                for h in range(self.heads):
                    # Start with a zero tensor and accumulate weighted block outputs
                    head_view = torch.zeros_like(block_outputs[0][:, :1])
                    
                    for i, block_out in enumerate(block_outputs):
                        head_view += q_weights[i] * block_out[:, :1]
                    
                    head_views.append(head_view)
                
                # Process each head separately
                memory_outputs = []
                batch_size = block_outputs[0].shape[0]
                
                for h in range(self.heads):
                    # Project to memory dimension
                    head_view = head_views[h]
                    q_mem = self.down_proj(head_view)
                    
                    # Generate queries
                    queries = self.to_queries(q_mem)
                    
                    if self.qk_rmsnorm:
                        queries = self.q_norm(queries)
                    
                    # Process with specific head parameters
                    # For memory tokens, we'll temporarily update the memory model params with this head's params
                    with torch.no_grad():
                        # Save original parameters
                        orig_params = {}
                        for model_name, model_param in self.memory_model.named_parameters():
                            param_name = model_name.replace(".", "_")
                            orig_params[param_name] = model_param.data.clone()
                            
                            # Set parameters for this head
                            if self.per_head_learned_parameters and param_name in self.mem_params:
                                model_param.data.copy_(self.mem_params[param_name][h])
                    
                    # Process through memory model
                    try:
                        memory_out = self.memory_model(queries).expand(-1, num_tokens, -1)
                        memory_outputs.append(memory_out)
                    except Exception as e:
                        print(f"Error processing memory tokens for head {h}: {e}")
                        # Use zeros as fallback
                        memory_outputs.append(torch.zeros(batch_size, num_tokens, self.memory_dim, device=device))
                    
                    # Restore original parameters
                    with torch.no_grad():
                        for model_name, model_param in self.memory_model.named_parameters():
                            param_name = model_name.replace(".", "_")
                            if param_name in orig_params:
                                model_param.data.copy_(orig_params[param_name])
                
                # Combine outputs from all heads and project back to original dimension
                combined_output = torch.stack(memory_outputs).mean(dim=0)  # Average over heads
                return self.up_proj(combined_output)
            else:
                # Standard approach for when differentiated views aren't available
                query = torch.zeros(1, 1, self.memory_dim, device=device)
                
                # Process through memory model
                memory_output = self.memory_model(query).expand(1, num_tokens, -1)
                
                # Project to full dimension
                return self.up_proj(memory_output)
                
        except Exception as e:
            print(f"Error in get_memory_tokens: {e}")
            # Return zeros as fallback
            return torch.zeros(1, num_tokens, self.dim, device=device)