"""
Advanced neural memory module for the Titans architecture.
Includes momentum, gradient-based surprise, and differentiated QKV mechanisms.
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
        remove_prev: Optional[bool] = None
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
        heads: int = 1  # Add heads parameter with a default
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
            integration_type: Integration approach ("mal", "mac", "mag")
            use_accelerated_scan: Whether to use accelerated scan implementation
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
        
        # Memory dynamics parameters
        self.adaptive_lr = nn.Parameter(torch.tensor(0.01))
        self.momentum_factor = nn.Parameter(torch.tensor(0.9))
        self.forget_factor = nn.Parameter(torch.tensor(0.1))
        
        # Normalization
        if qk_rmsnorm:
            self.q_norm = nn.LayerNorm(self.memory_dim)
            self.k_norm = nn.LayerNorm(self.memory_dim)
        
        # For associative scan
        self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)
        
        # Memory state buffers
        for name in self.mem_params:
            # Create momentum buffers for each parameter
            if momentum:
                if self.per_head_learned_parameters:
                    # Initialize as None, will be created properly during first use
                    self.register_buffer(f'momentum_{name}', None, persistent=False)
                else:
                    # Original behavior
                    self.register_buffer(f'momentum_{name}', None, persistent=False)
        
        # Loss function for surprise computation
        self.store_memory_loss_fn = lambda pred, target: (pred - target).pow(2).mean(dim=-1)
        
        # Add block weighting for differentiated views if enabled
        if self.qkv_receives_diff_views:
            # These will be initialized in set_transformer_blocks
            self.q_block_weights = None
            self.k_block_weights = None
            self.v_block_weights = None
            
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
            self.q_block_weights = nn.Parameter(torch.zeros(num_blocks))
            self.k_block_weights = nn.Parameter(torch.zeros(num_blocks))
            self.v_block_weights = nn.Parameter(torch.zeros(num_blocks))
            
            # Initialize with reasonable priors
            nn.init.normal_(self.q_block_weights, 0, 0.02)
            nn.init.normal_(self.k_block_weights, 0, 0.02)
            nn.init.normal_(self.v_block_weights, 0, 0.02)
            
            # Default to attending to early, middle, and late blocks
            with torch.no_grad():
                if num_blocks >= 3:
                    self.q_block_weights[0] = 2.0              # Early block for queries
                    self.k_block_weights[num_blocks//2] = 2.0  # Middle block for keys
                    self.v_block_weights[-1] = 2.0             # Late block for values
    
    def compute_surprise(
    self, 
    keys: torch.Tensor,  # [batch*heads, seq_len, memory_dim]
    values: torch.Tensor  # [batch*heads, seq_len, memory_dim]
) -> Dict[str, torch.Tensor]:
        """Compute surprise via gradient computation."""
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
            loss = self.store_memory_loss_fn(preds, values_detached).mean()
            
            # Compute gradients
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
                    if self.per_head_learned_parameters:
                        # Shape gradients for per-head parameters
                        # This needs to match the shape in mem_params
                        batch_head_size = keys.shape[0]
                        # Reshape to add head dimension - this depends on how you've batched your inputs
                        # Let's assume gradients should be reshaped to [heads, ...param_shape]
                        grad_shaped = grad_val.unsqueeze(0)  # Add head dimension
                        grad_dict[param_name] = -grad_shaped  # Negative for surprise
                    else:
                        # Original behavior
                        grad_dict[param_name] = -grad_val
                else:
                    # Zero gradients for params without gradients
                    if self.per_head_learned_parameters:
                        grad_dict[param_name] = torch.zeros_like(self.mem_params[param_name])
                    else:
                        grad_dict[param_name] = torch.zeros_like(param)
        except Exception as e:
            print(f"Error computing surprise: {e}")
            # Fallback: create zero gradients
            grad_dict = {}
            for name in self.mem_params:
                grad_dict[name] = torch.zeros_like(self.mem_params[name])
        
        finally:
            # Restore original states
            for name, param in self.memory_model.named_parameters():
                param.requires_grad_(original_requires_grad[name])
            
            # Restore training state
            self.memory_model.train(training_state)
        
        return grad_dict
    
    def update_memory_params(self, surprise_grads: Dict[str, torch.Tensor]) -> None:
        """Update memory parameters with computed surprise gradients."""
        for name, mem_param in self.mem_params.items():
            if name in surprise_grads:
                # Compute the update using momentum if enabled.
                if self.momentum_enabled:
                    momentum_buffer_name = f"momentum_{name}"
                    momentum_buffer = getattr(self, momentum_buffer_name, None)
                    if momentum_buffer is not None:
                        # Use momentum update.
                        momentum_update = self.momentum_factor * momentum_buffer
                        param_update = momentum_update + self.adaptive_lr * surprise_grads[name]
                        setattr(self, momentum_buffer_name, param_update.detach())
                    else:
                        # Initialize momentum buffer.
                        param_update = self.adaptive_lr * surprise_grads[name]
                        setattr(self, momentum_buffer_name, param_update.detach())
                else:
                    # Simple update without momentum.
                    param_update = self.adaptive_lr * surprise_grads[name]
                
                # Instead of an in-place update, compute the new value and copy it.
                with torch.no_grad():
                    new_value = self.mem_params[name] - param_update
                    self.mem_params[name].copy_(new_value)
        
        # After updating, copy the aggregated parameters into the memory model.
        with torch.no_grad():
            for model_name, model_param in self.memory_model.named_parameters():
                param_name = model_name.replace(".", "_")
                if param_name in self.mem_params:
                    if self.per_head_learned_parameters:
                        # Aggregate per-head parameters by averaging over the head dimension.
                        model_param.copy_(self.mem_params[param_name].mean(dim=0))
                    else:
                        model_param.copy_(self.mem_params[param_name])
    def functional_update_memory_params(self, current_state: Dict[str, torch.Tensor],
                                          current_momentum: Dict[str, Optional[torch.Tensor]],
                                          surprise_grads: Dict[str, torch.Tensor]
                                         ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        """
        Compute new memory weights and momentum in a functional style.
        
        Args:
            current_state: dict of current memory weights.
            current_momentum: dict of current momentum buffers.
            surprise_grads: computed surprise gradients.
        
        Returns:
            A tuple (new_weights, new_momentum).
        """
        new_weights = {}
        new_momentum = {}
        for name in current_state:
            grad = surprise_grads.get(name, torch.zeros_like(current_state[name]))
            if self.momentum_enabled:
                if current_momentum[name] is None:
                    mom = self.adaptive_lr * grad
                else:
                    mom = self.momentum_factor * current_momentum[name] + self.adaptive_lr * grad
                new_momentum[name] = mom
                update = mom
            else:
                update = self.adaptive_lr * grad
                new_momentum[name] = None

            # Use associative scan if toggled on; otherwise, do a simple subtraction.
            if self.assoc_scan.use_accelerated:
                # Here we use assoc_scan as a parallel accumulation operator.
                # We simulate a gate of ones for accumulation.
                ones = torch.ones_like(update)
                accumulated = self.assoc_scan(ones, update, prev=current_state[name])
                new_weight = current_state[name] - accumulated
            else:
                new_weight = current_state[name] - update

            new_weights[name] = new_weight
        return new_weights, new_momentum
    
    def forward(self, x: torch.Tensor, state: Optional[Dict[str, Any]] = None,
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

        if self.qkv_receives_diff_views and block_outputs is not None and self.q_block_weights is not None:
            return self.forward_with_diff_views(x, block_outputs, state)

        # Standard forward pass: project input to memory dimension.
        x_mem = self.down_proj(x)
        keys = self.to_keys(x_mem)
        values = self.to_values(x_mem)
        if self.qk_rmsnorm:
            keys = self.k_norm(keys)

        # Compute surprise gradients.
        with torch.set_grad_enabled(True):
            surprise_grads = self.compute_surprise(keys, values)

        # Initialize state if not provided.
        if state is None:
            current_state = {k: v.clone() for k, v in self.mem_params.items()}
            current_momentum = {k: None for k in self.mem_params.keys()}
            seq_index = seq_len
        else:
            current_state = state["weights"]
            current_momentum = state["momentum"]
            seq_index = state.get("seq_index", 0) + seq_len

        # Compute new state functionally.
        new_weights, new_momentum = self.functional_update_memory_params(current_state, current_momentum, surprise_grads)

        new_state = {
            "seq_index": seq_index,
            "weights": new_weights,
            "momentum": new_momentum
        }

        # Update self.mem_params (and the memory model parameters) from the new state.
        with torch.no_grad():
            for name in self.mem_params:
                self.mem_params[name].data.copy_(new_weights[name])
            for model_name, model_param in self.memory_model.named_parameters():
                param_name = model_name.replace(".", "_")
                if param_name in self.mem_params:
                    if self.per_head_learned_parameters:
                        aggregated = self.mem_params[param_name].mean(dim=0)
                        model_param.data.copy_(aggregated)
                    else:
                        model_param.data.copy_(self.mem_params[param_name].detach())

        # Continue with forward: generate queries and process through memory model.
        queries = self.to_queries(x_mem)
        if self.qk_rmsnorm:
            queries = self.q_norm(queries)
        memory_output = self.memory_model(queries)
        output = self.up_proj(memory_output)

        return output, new_state
        
    def forward_with_diff_views(self, x: torch.Tensor, block_outputs: List[torch.Tensor],
                                state: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass using differentiated block outputs with per-head parameters, using functional updates.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            block_outputs: List of outputs from transformer blocks
            state: Optional previous state
            
        Returns:
            Tuple of (output tensor, new state dict)
        """
        batch_size, seq_len, _ = x.shape

        # Fallback if differentiated views are not available.
        if (not self.qkv_receives_diff_views or self.q_block_weights is None or len(block_outputs) < 2):
            return self.forward(x, state)

        # Normalize block view weights.
        q_weights = F.softmax(self.q_block_weights, dim=0)
        k_weights = F.softmax(self.k_block_weights, dim=0)
        v_weights = F.softmax(self.v_block_weights, dim=0)

        # Create per-head weighted block views.
        q_view = torch.stack([sum(w * block_out for w, block_out in zip(q_weights, block_outputs))
                            for _ in range(self.heads)])
        k_view = torch.stack([sum(w * block_out for w, block_out in zip(k_weights, block_outputs))
                            for _ in range(self.heads)])
        v_view = torch.stack([sum(w * block_out for w, block_out in zip(v_weights, block_outputs))
                            for _ in range(self.heads)])

        # Reshape from [heads, batch, seq_len, dim] to [heads * batch, seq_len, dim].
        q_mem = self.down_proj(q_view.reshape(-1, seq_len, q_view.shape[-1]))
        k_mem = self.down_proj(k_view.reshape(-1, seq_len, k_view.shape[-1]))
        v_mem = self.down_proj(v_view.reshape(-1, seq_len, v_view.shape[-1]))

        # Generate projections.
        queries = self.to_queries(q_mem)
        keys = self.to_keys(k_mem)
        values = self.to_values(v_mem)
        if self.qk_rmsnorm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        # Compute surprise gradients.
        with torch.set_grad_enabled(True):
            surprise_grads = self.compute_surprise(keys, values)

        # Initialize or retrieve the current state.
        if state is None:
            current_state = {k: v.clone() for k, v in self.mem_params.items()}
            current_momentum = {k: None for k in self.mem_params.keys()}
            seq_index = seq_len
        else:
            current_state = state["weights"]
            current_momentum = state["momentum"]
            seq_index = state.get("seq_index", 0) + seq_len

        # Compute new state functionally.
        new_weights, new_momentum = self.functional_update_memory_params(
            current_state, current_momentum, surprise_grads
        )
        new_state = {
            "seq_index": seq_index,
            "weights": new_weights,
            "momentum": new_momentum
        }

        # Update internal memory parameters and the memory model.
        with torch.no_grad():
            for name in self.mem_params:
                self.mem_params[name].data.copy_(new_weights[name])
            for model_name, model_param in self.memory_model.named_parameters():
                param_name = model_name.replace(".", "_")
                if param_name in self.mem_params:
                    if self.per_head_learned_parameters:
                        aggregated = self.mem_params[param_name].mean(dim=0)
                        model_param.data.copy_(aggregated)
                    else:
                        model_param.data.copy_(self.mem_params[param_name].detach())

        # Process queries per head.
        memory_outputs = []
        for i in range(self.heads):
            start = i * batch_size
            end = (i + 1) * batch_size
            memory_out = self.memory_model(queries[start:end])
            memory_outputs.append(memory_out)
        memory_output = torch.stack(memory_outputs, dim=0).mean(dim=0)

        output = self.up_proj(memory_output)
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
        
        if self.qkv_receives_diff_views and block_outputs is not None and self.q_block_weights is not None:
            # Create differentiated views
            q_weights = F.softmax(self.q_block_weights, dim=0)
            
            # Create per-head weighted view
            q_view = torch.stack([
                sum(w * block_out for w, block_out in zip(q_weights, block_outputs))
                for _ in range(self.heads)
            ])
            
            # Project down to memory dimension
            q_mem = self.down_proj(q_view.reshape(-1, 1, q_view.shape[-1]))
            
            # Generate queries
            queries = self.to_queries(q_mem)
            
            if self.qk_rmsnorm:
                queries = self.q_norm(queries)
            
            # Process through memory model with per-head parameters
            memory_output = torch.stack([
                self.memory_model(queries[i:i+1]).expand(-1, num_tokens, -1)
                for i in range(self.heads)
            ])
            
            # Average across heads and project back to full dimension
            return self.up_proj(memory_output.mean(dim=0))
        else:
            # Standard approach
            query = torch.zeros(1, 1, self.memory_dim, device=device)
            
            # Get memory output
            memory_output = self.memory_model(query).expand(1, num_tokens, -1)
            
            # Project to full dimension
            return self.up_proj(memory_output)