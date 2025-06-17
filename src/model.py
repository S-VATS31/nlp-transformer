import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Check Flash Attention 2 availability
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    FLASH_ATTN_AVAILABLE = True
    print("Flash Attention 2 is available")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Flash Attention 2 not available, using PyTorch SDPA")

# CUDA/AMP setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
print(f"Device = {device} | AMP dtype = {dtype}")

@dataclass
class ModelArgs:
    d_model: int = 1440
    num_heads: int = 24
    query_groups: int = 12
    d_ffn: int = 5760
    num_layers: int = 20
    dropout: float = 0.2
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-7
    vocab_size: int = 65536
    max_seq_len: int = 2048
    tie_weights: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 65535 
    gradient_checkpointing: bool = False

@dataclass
class TrainingArgs:
    learning_rate: float = 2e-4
    epochs: int = 300
    batch_size: int = 256
    epsilon: float = 1e-6
    clip_grad_norm: float = 1.0
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_epochs: int = 50
    eta_min: float = 6e-7
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    grad_accum_steps: int = 4
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500

class RoPE(torch.nn.Module):
    """Rotary positional embeddings (RoPE) to be applied to the query and key vectors.

    Rotary Positional Embedding (RoPE) applies a complex rotation to the query and key
    vectors using sinusoidal functions. This is done by element-wise mixing of the even
    and odd dimensions of each vector with precomputed sine and cosine frequencies.

    Args:
        head_dim (int): Dimension of each attention head.
        theta (float): Exponential base of the inverse frequency.
    
    Raises:
        ValueError if `head_dim` is not divisble by 2.
    """
    def __init__(self, head_dim: int, theta: float):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 2 for even splitting.")
        self.head_dim = head_dim
    
        # Compute inverse frequency
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
        self.register_buffer("inv_freq", inv_freq)
        
        # Lazy cache - will be populated as needed to avoid recomputing sin/cos
        self.register_buffer("cos_cache", torch.empty(0))
        self.register_buffer("sin_cache", torch.empty(0))
        self.cached_seq_len = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RoPE layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, num_heads, head_dim].
            
        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            seq_len = x.size(1)

            # Update cache if needed
            if seq_len > self.cached_seq_len:
                self._update_cache(seq_len)

            # Use cached values instead of recomputing
            # Update all values up till sequence length
            cos_freqs = self.cos_cache[:seq_len] # [T, head_dim//2]
            sin_freqs = self.sin_cache[:seq_len] # [T, head_dim//2]

            # Apply rotary embeddings
            return self._apply_rope(x, cos_freqs, sin_freqs)
    
    def _update_cache(self, seq_len: int):
        """Cache sine and cosine to prevent re-computation during forward pass.
        
        Args:
            seq_len (int): Sequence length to cache for.
        """
        # Create position indices
        pos = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies for each position
        freqs = torch.outer(pos, self.inv_freq) # [T, head_dim//2]
        
        # Create rotation matrix components and cache them
        self.cos_cache = torch.cos(freqs) # [T, head_dim//2]
        self.sin_cache = torch.sin(freqs) # [T, head_dim//2]
        self.cached_seq_len = seq_len
    
    def _apply_rope(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the rotation using the RoPE formula.

        This functions applies a rotation to the input via complex multiplication.
        This allows the direction of the vectors to be shifted while preserving the
        magnitude of the vectors

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_heads, head_dim].
            cos_freqs (torch.Tensor): Cosine frequencies of shape [seq_len, head_dim//2].
            sin_freqs (torch.Tensor): Sine frequencies of shape [seq_len, head_dim//2].
        
        Returns:
            rotated_x (torch.Tensor): Rotated output tensor with positional awareness.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            # Split x into even and odd dimensions
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            
            # Expand frequency tensors to match input dimensions
            cos_freqs = cos_freqs.unsqueeze(0).unsqueeze(2) # [1, T, 1, head_dim//2]
            sin_freqs = sin_freqs.unsqueeze(0).unsqueeze(2) # [1, T, 1, head_dim//2]
            
            # Complex rotation via rotation matrix 
            rotated_x1 = x1 * cos_freqs - x2 * sin_freqs
            rotated_x2 = x1 * sin_freqs + x2 * cos_freqs
            
            # Interleave the rotated components back
            rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)
            rotated_x = rotated_x.flatten(-2)
            
            return rotated_x
    
    def get_cos_sin_cache(
        self, 
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache sine and cosine to prevent re-computation during forward pass.

        Args:
            seq_len (int): Sequence length
        
        Returns:
            tuple: Cached values.
                - cos_freqs (Tensor): cached cosine frequencies.
                - sin_freqs (Tensor): cached sine frequencies.
        """
        # Ensure cache is up to date for this sequence length
        if seq_len > self.cached_seq_len:
            self._update_cache(seq_len)
        
        # Return the cached values for the requested length
        cos_freqs = self.cos_cache[:seq_len]
        sin_freqs = self.sin_cache[:seq_len]
        return cos_freqs, sin_freqs

class RMSNorm(torch.nn.Module):
    """Apply RMSNorm to input tensors to normalize their root mean square norm.

    RMSNorm normalizes the tensor's root mean squared norm instead of the mean and
    variance unlike LayerNorm. RMSNorm is also a much simpler alternative.
    
    Formula:
        x_normalized = x / RMS
        Where RMS = sqrt(mean(x**2))

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(self, d_model: int, eps: float):
        super().__init__()

        self.eps = eps
        # Learnable scaling factor (gamma), initialized to ones so that
        # RMSNorm behaves neutrally at the start (no per-dim scaling).
        # During training, the model can learn to scale each dimension
        # if it improves performance.
        self.weight = torch.nn.Parameter(torch.ones(d_model)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the RMSNorm layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Normalized output tensor of same shape.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return self.weight * (x / rms)

class KVCache:
    """Key-Value cache for efficient autoregressive generation.

    Stores key and value tensors for each transformer layer to avoid recomputation
    during sequential token generation. The cache supports dynamic updates and retrieval
    for batched inputs up to a maximum sequence length and batch size.

    Args:
        max_batch_size (int): Maximum batch size supported by the cache.
        max_seq_len (int): Maximum sequence length supported by the cache.
        num_heads (int): Number of attention heads (or query groups for GQA).
        head_dim (int): Dimension of each attention head.
        num_layers (int): Number of transformer layers.
        dtype (torch.dtype, optional): Data type for cache tensors. Defaults to global dtype.
        device (torch.device, optional): Device for cache tensors. Defaults to global device.
    """
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = dtype,
        device: torch.device = device
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        # Initialize cache state to None; they will then be initialized to zeros in the
        # `initialize` method.
        self.cache = None
        self.current_seq_len = None
        self.batch_size = None

    def initialize(self, batch_size: int) -> None:
        """Initialize or reset the cache for a given batch size.

        Creates zero-filled key and value tensors for each layer, sized for the
        specified batch size and maximum sequence length.

        Args:
            batch_size (int): Number of sequences to process.

        Raises:
            ValueError: If batch_size exceeds max_batch_size.
        """
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")

        self.batch_size = batch_size
        self.current_seq_len = 0

        # Initialize KV cache with zeros
        # The loop creates a list of with num_layers amount of elements where each
        # layer of the model will have its own key and value cache. This is because
        # in KV caching, the key and value tensors are cached layer-wise to prevent
        # recomputation of these key and value vectors which have fixed values due
        # to the way embeddings work. However, the query represents the current token
        # so it must be recomputed for all new tokens
        self.cache = [
            {
                'k': torch.zeros((batch_size, self.max_seq_len, self.num_heads, self.head_dim), dtype=self.dtype, device=self.device),
                'v': torch.zeros((batch_size, self.max_seq_len, self.num_heads, self.head_dim), dtype=self.dtype, device=self.device)
            }
            for _ in range(self.num_layers)
        ]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Update the cache with new key and value tensors for a specific layer.

        Args:
            layer_idx (int): Index of the transformer layer to query.
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim].
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim].

        Raises:
            ValueError: If sequence length exceeds max_seq_len.
        """
        # The initialize method will be called if the cache does not have a given value or
        # if there is a mismatch in dimensions.
        if self.cache is None or self.batch_size != k.size(0):
            self.initialize(k.size(0))

        new_seq_len = k.size(1)
        # Ensure current T + new T <= max T
        if self.current_seq_len + new_seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {self.current_seq_len + new_seq_len} exceeds maximum {self.max_seq_len}")

        # Update cache with new key and value tensors
        # Since the cache is a dictionary, the key and value vectors must be accessed as following.
        # First, layer_idx is passed because each layers key and value tensors have different values
        # so we need to update the cache with respect to each layers different KV values. Without
        # adding the layer_idx the same KV slot would be rewritten over for num_layers amount of times.
        #
        # Second, since the cache is a list with nested dictionary, with the keys being either "k" for
        # cached key tensors or "v" for cached value tensors. These are added to make sure that we are
        # updating the cache correctly with respect to either the key or value tensors.
        #
        # Finally, since the key and values caches have shape: [B, T, num_heads, head_dim], we only
        # want to access dimension 1 since it is the sequence length dimension. This is because when
        # new tokens come in we want to take their key and value vectors and append them to our cache.
        # Since we are adding new tensors, the sequence length will be updated while keeping the batch
        # size, number of heads, and the dimension of each head the same. In the code this is done by
        # slicing and updating over the sequence length dimension. This is because T_current is the current
        # sequence length and we are updating the sequence length by taking the T_current and adding the
        # new sequence length based on the new key and value tensors based on new tokens.
        self.cache[layer_idx]['k'][:, self.current_seq_len:self.current_seq_len + new_seq_len] = k
        self.cache[layer_idx]['v'][:, self.current_seq_len:self.current_seq_len + new_seq_len] = v

    def get(self, layer_idx: int, seq_len: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve key and value tensors up to the specified sequence length.

        Args:
            layer_idx (int): Index of the transformer layer to query.
            seq_len (int): Sequence length to retrieve.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: Key and value tensors up to seq_len,
                or (None, None) if cache is uninitialized or seq_len exceeds current_seq_len.
        """
        # If cache is not given or requested sequence length is greater than the actual
        # sequence length of the cache itself return None for the KV cache.
        if self.cache is None or seq_len > self.current_seq_len:
            return None, None

        # Return KV cache up to the requested sequence length
        return (
            self.cache[layer_idx]['k'][:, :seq_len],
            self.cache[layer_idx]['v'][:, :seq_len]
        )

    def increment_seq_len(self, increment: int) -> None:
        """Increment the current sequence length after updating the cache.

        Args:
            increment (int): Amount to increment the current sequence length.
        """
        self.current_seq_len += increment

    def reset(self) -> None:
        """Reset the cache to its initial state.

        Clears the cache, current sequence length, and batch size.
        """
        self.cache = None
        self.current_seq_len = None
        self.batch_size = None

class Attention(torch.nn.Module):
    """Apply Grouped Query Attention (GQA) to QKV vectors as well as causal masking.

    Instead of computing attention for all key/value pairs, GQA splits these key/value
    pairs into query groups. This allows for significantly less computation. This program
    also implements causal masking to allow for autoregressive generation.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads for KV vectors, shared across grouped query heads in GQA.
        query_groups (int): Number of query groups that partition the queries into subsets.
        theta (float): Exponential base of the inverse frequency.
    """
    def __init__(self, d_model: int, num_heads: int, query_groups: int, theta: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.head_dim = d_model // num_heads

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % query_groups != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by query_groups ({query_groups})")

        # QKV projection shapes:
        # - Q shape: [B, T, num_heads * head_dim]
        # - K shape: [B, T, query_groups * head_dim]
        # - V shape: [B, T, query_groups * head_dim]
        # num_heads * head_dim is for the query projection
        # 2 * query_groups * head_dim is because the key and value vectors
        # both get their own set of query_groups and head dimensions
        self.w_qkv = torch.nn.Linear(
            d_model,
            num_heads * self.head_dim + 2 * query_groups * self.head_dim,
            bias=False,
            dtype=dtype
        ).to(device)
    
        # Output projection shape: [B, T, d_model]
        self.w_o = torch.nn.Linear(
            d_model, 
            d_model, 
            bias=False, 
            dtype=dtype
        ).to(device)

        self.rope = RoPE(self.head_dim, theta)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass for GQA layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D].
            causal (bool): To apply causal masking or not.
            padding_mask (torch.Tensor, optional): Padding mask of shape [B, T] where 
                True indicates valid tokens and False indicates padding tokens.
            kv_cache (KVCache, optional): Key-value cache for efficient generation.
            layer_idx (int, optional): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]: 
                - Attention output tensor of shape [B, T, d_model].
                - Cache dictionary with 'k' and 'v' tensors if use_cache is True, else None.
        
        Raises:
            ValueError if `x` does not have 3 dimensions.
            ValueError if sequence length, `T` is 0.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            if x.dim() != 3:
                raise ValueError(f"Expected input of shape [B, T, d_model], got {x.shape}")
            B, T, D = x.shape
            if T == 0:
                # Return empty tensor if sequence length is 0
                return torch.empty(B, 0, D, device=x.device, dtype=x.dtype), None

            # Project QKV
            qkv = self.w_qkv(x) # [B, T, num_heads * head_dim + 2 * query_groups * head_dim]

            # q shape: [B, T, num_heads * head_dim]
            # kv shape: [B, T, 2 * query_groups * head_dim]
            q, kv = torch.split(qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1)

            # k shape: [B, T, query_groups * head_dim]
            # v shape: [B, T, query_groups * head_dim]
            k, v = torch.chunk(kv, 2, dim=-1)

            # Reshape into 4D tensors for GQA
            q = q.view(B, T, self.num_heads, self.head_dim) # [B, T, num_heads, head_dim]
            k = k.view(B, T, self.query_groups, self.head_dim) # [B, T, query_groups, head_dim]
            v = v.view(B, T, self.query_groups, self.head_dim) # [B, T, query_groups, head_dim]

            # Apply RoPE
            q = self.rope(q) # [B, T, num_heads, head_dim]
            k = self.rope(k) # [B, T, query_groups, head_dim]

            # Handle KV cache
            # KV cache can only be used if required values are given
            if use_cache and kv_cache is not None and layer_idx is not None:
                # Get KV cache with the current sequence length
                cached_k, cached_v = kv_cache.get(layer_idx, kv_cache.current_seq_len)
                if cached_k is not None and cached_v is not None:
                    # Concatenate cached and new KV
                    k = torch.cat([cached_k, k], dim=1)
                    v = torch.cat([cached_v, v], dim=1)
                # Update cache using only the T most recent tokens
                # Using k[:, -T:] and v[:, -T:] extracts exactly the new KV tensors from this step.
                # Passing the entire k or v (which may include cached tokens) would corrupt the cache,
                # since the cache should only be updated with new KV tensors, not the full sequence.
                kv_cache.update(layer_idx, k[:, -T:], v[:, -T:])

            # Expand KV for GQA
            if self.query_groups != self.num_heads:
                # heads_per_group will be the number of repeats
                heads_per_group = self.num_heads // self.query_groups
                # Repeating heads_per_group over the query_groups dimension will cause
                # the query_groups to be equal to num_heads. This is because repeating 
                # heads_per_group for query_groups amount of times is simply 
                # heads_per_group * query_groups which is equal to num_heads.
                # This makes computation simpler because attention can be computed as normal.
                # If query_groups is not equal to the number of heads, then the query, key, 
                # and value tensors would all have different shapes and they could not be
                # stacked for Flash Attention.
                # New shapes: [B, T, num_heads, head_dim]
                k = k.repeat_interleave(heads_per_group, dim=2)
                v = v.repeat_interleave(heads_per_group, dim=2)

            # FlashAttention 2 - requires CUDA/flash attn 2 available
            if FLASH_ATTN_AVAILABLE and device.type == "cuda":
                qkv_packed = torch.stack([q, k, v], dim=3) # [B, T, num_heads, 3, head_dim]

                # Call contiguous because stacking tensors returns a view of the original tensor
                # and can return a tensor that is not contiguous. Also, Flash Attention 2 expects
                # contiguous tensors for memory reasons.
                qkv_packed = qkv_packed.contiguous()

                # Handle padding mask for FlashAttention
                if padding_mask is not None:
                    # Ensure padding mask is a boolean tensor
                    padding_mask = padding_mask.bool()
                    # Convert padding mask to cumulative sequence lengths
                    # padding mask -> [B, T]
                    # dim=1 takes sequence length dimension
                    # .int() -> dtype -> torch.int32
                    # .sum() -> takes all ones for each T dimension
                    seq_lens = padding_mask.sum(dim=1).int() # [B]

                    # Compute cumulative sum over the batch dimension to get the
                    # end indices of each sequence in the flattened (packed) tensor.
                    # Since this gives only the end positions, we prepend a 0 to represent
                    # the start of the first sequence. This produces the start indices
                    # for all sequences, as expected by FlashAttention.
                    cu_seqlens = torch.cat([
                        torch.tensor([0], dtype=torch.int32, device=device), # [0]
                        seq_lens.cumsum(0) # [B]
                    ], dim=0) # Shape: [B + 1]
                    
                    # This is an expected parameter for the FlashAttention function
                    max_seqlen = seq_lens.max().item()
                    
                    # Valid tokens = 1, padded tokens = 0
                    # Padding mask shape: [B, T]
                    # Flattened padding mask shape: [B * T]
                    valid_tokens = padding_mask.flatten()

                    # Shape before indexing: [B * T, num_heads, 3, head_dim]
                    # Indexing by valid_tokens where valid_tokens is a one dimensional boolean
                    # tensor. True means the token will not be padded and False means the token
                    # will be padded. This is important because the FlashAttention function
                    # only takes valid tokens so the invalid tokens are filtered out.
                    qkv_packed = qkv_packed.view(-1, self.num_heads, 3, self.head_dim)[valid_tokens]
                else:
                    # Cumulative sequence length is a tensor as following:
                    # Say B = 3, and k.size(1) (or T) = 4 then cu_seqlens becomes:
                    # torch.arange(0, (3 + 1) * 4, 4) or torch.arange(0, 16, 4)
                    # This creates a one dimensional tensor: tensor([0, 4, 8, 12])
                    cu_seqlens = torch.arange(0, (B + 1) * k.size(1), k.size(1), dtype=torch.int32, device=device) # [B + 1]
                    max_seqlen = k.size(1) # [B, T, ...] - takes sequence length dimension

                    # Total tokens if calculated as B * T (batch size * sequence length)
                    # Dimension 1 of the key vector is the sequence length
                    total_tokens = B * max_seqlen

                    # Without the padding mask this is much simpler because since no tokens are
                    # being padded, they are all considered valid. So, we simply just calculated
                    # the total tokens by taking batch size * sequence length.
                    qkv_packed = qkv_packed.view(total_tokens, self.num_heads, 3, self.head_dim)

                # Call FlashAttention 2
                # `causal` is needed for autoregressive generation to ensure the model does not
                # attend to future tokens during training or inference. This mimics the real-world
                # scenario where predictions are made token-by-token.
                #
                # `softmax_scale` controls the magnitude of attention scores (Q @ K^T). Without this scaling,
                # the dot products can get too large, especially when the dimensionality of heads (`d_k`) is high.
                # Large values can push softmax outputs toward 0 or 1, which leads to vanishing gradients —
                # making training unstable and hurting the model's ability to learn meaningful attention patterns.
                #
                # `flash_attn_varlen_qkvpacked_func` expects:
                #   - `qkv_packed`: shape [total_tokens, num_heads, 3, head_dim]
                #   - `cu_seqlens`: cumulative sequence lengths with shape of [B + 1]
                #   - `max_seqlen`: int, maximum sequence length in the batch
                #   - `causal`: bool, whether to apply causal masking
                #   - `softmax_scale`: float, scale factor for softmax (usually 1 / sqrt(d_k))
                out = flash_attn_varlen_qkvpacked_func(
                    qkv_packed,
                    cu_seqlens,
                    max_seqlen,
                    causal=causal,
                    softmax_scale=1.0 / (self.head_dim ** 0.5) # 1.0/sqrt(d_k)
                ) # [total_tokens, num_heads, head_dim]

                # This is done because the FlashAttention 2 function returns only unpadded tokens.
                # We can reconstruct the padded positions by indexing with the valid tokens once again
                # where the valid tokens are True and the padded tokens are False.
                if padding_mask is not None:
                    full_out = torch.zeros(B * T, self.num_heads, self.head_dim,  dtype=out.dtype, device=out.device)

                    # Update valid and padded token positions where valid_tokens is a boolean tensor.
                    full_out[valid_tokens] = out
                    out = full_out.view(B, T, D) # Reshape output to [B, T, d_model]
                else:
                    # If no padding, reshape right away
                    out = out.view(B, T, D)

            else:
                # Fallback to PyTorch SDPA
                # Before Transpose:
                # - Q shape: [B, T, num_heads, head_dim]
                # - K/V shape: [B, T, num_heads, head_dim] `num_heads` ONLY IF `query_groups` ARE REPEATED FOR
                # `heads_per_groups` TIMES, ELSE JUST `query_groups`
                # After Transpose:
                # - Q shape: [B, num_heads, T, head_dim]
                # - KV shape: [B, num_heads, T, head_dim] `num_heads` ONLY IF `query_groups` ARE REPEATED FOR
                # `heads_per_groups` TIMES, ELSE JUST `query_groups`
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                attn_mask = None

                # Apply padding mask for PyTorch SDPA
                if padding_mask is not None:
                    padding_mask = padding_mask.bool()
                    # The input padding mask has shape [B, T], but PyTorch's SDPA expects a
                    # 4 dimensional tensor with shape [B, num_heads, T_k, T_q]. This is done
                    # by adding singleton dimensions to get a shape of [B, 1, T, 1]. Then, 
                    # PyTorch's `expand` method can be used to take the single dimension at
                    # index 3 (or -1) and insert the sequence length of the key tensors.
                    attn_mask = padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, T]
                    attn_mask = attn_mask.expand(B, 1, T, k.size(2)) # [B, 1, T_q, T_k]

                    # Apply causal masking if True
                    if causal:
                        # Causal mask shape: [T_q, T_k] where the upper right diagonal portion is False.
                        # True means the model is allowed to attend to this token, False means the model
                        # is not allowed to attend to the token. This is to ensure the model is actually
                        # predicting the next token rather than looking ahead at future tokens.
                        # The causal mask matrix has shape Q rows x K columns, where query positions are the current token
                        # and key positions are positions the model can look at.
                        causal_mask = torch.tril(torch.ones(T, k.size(2), dtype=torch.bool, device=device))

                        # Since the attention scores tensor has shape [B, num_heads, T_q, T_k], the causal
                        # mask must have same amount of dimensions for correct broadcasting.
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # [1, 1, T_q, T_k] 2 -> 4

                        # This returns a boolean tensor which returns True wherever causal_mask and
                        # attn_mask are True. This ensures that the model respects both causal
                        # and padding masking. This is done by taking wherever both of the boolean
                        # tensors are True. This is because wherever the causal mask is True, it means
                        # the model is allowed to attend to this token, and when the padding mask it 
                        # True, it means the token is valid and it will be used for attention. 
                        # This means that wherever they are both True, the token is one that the model
                        # can attend to and it should not be padded.
                        attn_mask = attn_mask & causal_mask # [B, 1, T_q, T_k]

                    # Expand to [B, num_heads, T_q, T_k] only now to avoid unnecessary memory usage.
                    # The excessive memory using comes from using the bitwise AND operator. This is
                    # because if we expand to num_heads early, the tensors will be larger with many
                    # and the new tensor, attn_mask will have to allocate more memory since the
                    # & operator creates a new tensor instead of a view of both tensors.
                    # NOTE: Memory usage does not come the expand method itself, but rather how the
                    # tensors are used.
                    attn_mask = attn_mask.expand(B, self.num_heads, T, k.size(2))

                # Call PyTorch's scaled dot product attention
                out = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=attn_mask, 
                    is_causal=causal if padding_mask is None else False
                ) # [B, num_heads, T, head_dim]

                # We want the output tensor to be shape [B, T, d_model] but PyTorch's
                # SDPA returns a tensor of shape [B, num_heads, T, head_dim]. We can
                # reshape this tensor since num_heads * head_dim = d_model, but we must
                # first transpose dimensions 1 and 2 because the num_heads and head_dim
                # dimensions are split by the sequence length dimension. This is why 
                # we transpose the dimensions (1, 2). This then gives us a tensor of shape
                # [B, T, num_heads, head_dim]. But, the transpose function changes the order
                # of the default memory pattern. This means the tensor is not contiguous
                # anymore. The view function requires a tensor that is contiguous. After
                # applying the contiguous method, the tensor can then successfully be reshaped
                # using the view() method since the last two dimensions (num_heads and head_dim)
                # multiply to d_model.

                # num_heads * head_dim = d_model
                # [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim]
                out = out.transpose(1, 2).contiguous().view(B, T, D)

            # Get cache output for the Attention layer. We slice over the sequence length
            # dimension and use -T to get the T (sequence length) KV tensors for the T
            # most recent tokens. If cache is not used this will be set to None.
            cache_out = {'k': k[:, -T:], 'v': v[:, -T:]} if use_cache else None

            # Final projection
            return self.w_o(out), cache_out

class FFN(torch.nn.Module):
    """Feed forward neural network (FFN) using Swish-Gated Linear Unit (SwiGLU) activation.

    The FFN features 3 linear transformers. SwiGLU activation allows for the neural
    network to approximate non-linear functions.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the FFN. d_ffn = d_model * 4.
        dropout (float): Probability that model components will be randomly dropped out.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()
        
        self.weight1 = torch.nn.Linear(d_model, d_ffn)
        self.weight2 = torch.nn.Linear(d_ffn, d_model)
        self.weight3 = torch.nn.Linear(d_model, d_ffn)
        self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Gated FFN layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            torch.Tensor: Output tensor passed through the FFN with same shape.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            return self.dropout(self.weight2(F.silu(self.weight1(x)) * self.weight3(x)))

class AttentionBlock(torch.nn.Module):
    """Attention block where RMSNorm, Dropout, and residuals are applied.

    The Attention Block takes the attention module and applies techniques such as
    RMSNorm, Dropout, and residuals for smoother gradient flow.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads for KV vectors, shared across grouped query heads in GQA.
        query_groups (int): Number of query groups that partition the queries into subsets.
        dropout (float): Probability that model components will be randomly dropped out.
        theta (float): Exponential base of the inverse frequency.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        dropout: float,
        theta: float,
        eps: float,
    ):
        super().__init__()

        self.rms_norm = RMSNorm(d_model, eps)
        self.attn = Attention(d_model, num_heads, query_groups, theta)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass of the Attention Block.

        Applies the following sequence of operations:
        1. RMSNorm on the input.
        2. Flash Attention (if available) + GQA on the normalized input.
        3. Dropout on the attention output.
        4. Adds the result to the original input (residual connection).

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
            kv_cache (KVCache, optional): Key-value cache for efficient generation.
            layer_idx (int, optional): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                - Output tensor of shape [B, T, d_model].
                - Cache dictionary with 'k' and 'v' tensors if use_cache is True, else None.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            attn_out, cache_out = self.attn(
                self.rms_norm(x), 
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                use_cache=use_cache
            )
            return x + self.dropout(attn_out), cache_out

class FFNBlock(torch.nn.Module):
    """FFN block where RMSNorm, Dropout, and residuals are applied.

    The FFN Block takes the FFN module and applies techniques such as
    RMSNorm, Dropout, and residuals for smoother gradient flow.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        d_ffn (int): Dimensionality of the FFN. d_ffn = d_model * 4.
        dropout (float): Probability that model components will be randomly dropped out.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(self, d_model: int, d_ffn: int, dropout: float, eps: float):
        super().__init__()

        self.rms_norm = RMSNorm(d_model, eps)
        self.ffn = FFN(d_model, d_ffn, dropout)
        self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the FFN Block.

        Applies the following sequence of operations:
        1. RMSNorm on the input.
        2. Pass through FFN.
        3. Dropout on the FFN output.
        4. Adds the result to the original input (residual connection).
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
                Not used in FFN, but added to prevent parameter errors.

        Returns:
            torch.Tensor: Output tensor passed through the FFN Block with the same shape.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            return x + self.dropout(self.ffn(self.rms_norm(x)))

class TransformerBlock(torch.nn.Module):
    """Transformer block that will be stacked in the final Transformer class.

    Each transformer block consists of a pass through the attention and
    and feed forward network block.

    Note:
        Usually the attention/FFN blocks would be done here, but this program
        already creates the blocks and stores them as separate nn.Modules. Now, the
        blocks can simply be instantiated and stacked in the Transformer.

    Args:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads for KV vectors, shared across grouped query heads in GQA.
        query_groups (int): Number of query groups that partition the queries into subsets.
        d_ffn (int): Dimensionality of the FFN. d_ffn = d_model * 4.
        dropout (float): Probability that model components will be randomly dropped out.
        theta (float): Exponential base of the inverse frequency.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        d_ffn: int,
        dropout: float,
        theta: float,
        eps: float,
    ):
        super().__init__()

        self.attn_block = AttentionBlock(d_model, num_heads, query_groups, dropout, theta, eps)
        self.ffn_block = FFNBlock(d_model, d_ffn, dropout, eps)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
            kv_cache (KVCache, optional): Key-value cache for efficient generation.
            layer_idx (int, optional): Index of the current layer for cache access.
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                - Output tensor of shape [B, T, d_model].
                - Cache dictionary with 'k' and 'v' tensors if use_cache is True, else None.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            x, cache_out = self.attn_block(
                x, 
                padding_mask=padding_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                use_cache=use_cache
            )
            x = self.ffn_block(x, padding_mask=padding_mask)
            return x, cache_out

class Transformer(torch.nn.Module):
    """Complete Transformer class stacking all decoder blocks.

    This class stacks all of the decoder blocks where each decoder block
    consists of the attention and FFN blocks. This class also features a generation
    method, `generate` to autoregressively generate tokens.

    Args:
        model_args (ModelArgs): Dataclass containing all model arguments.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        self.token_embed = torch.nn.Embedding(model_args.vocab_size, model_args.d_model).to(device)
        self.dropout = torch.nn.Dropout(p=model_args.dropout).to(device)

        # Stack transformer blocks
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                model_args.d_model,
                model_args.num_heads,
                model_args.query_groups,
                model_args.d_ffn,
                model_args.dropout,
                model_args.rope_base,
                model_args.rms_norm_eps,
            ).to(device) for _ in range(model_args.num_layers)
        ])

        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps).to(device)
        
        # Language modeling head
        self.lm_head = torch.nn.Linear(model_args.d_model, model_args.vocab_size).to(device)

        # Initialize KV cache
        self.kv_cache = KVCache(
            max_batch_size=32,
            max_seq_len=model_args.max_seq_len,
            num_heads=model_args.query_groups,
            head_dim=model_args.d_model // model_args.num_heads,
            num_layers=model_args.num_layers,
            dtype=dtype,
            device=device
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initializes model weights according to layer type.

    Embeddings:
        Initialized from a normal distribution with mean 0 and std 0.02.
        This helps maintain stable gradients during early training.

    Linear layers:
        Initialization depends on the layer's role:
        - QKV projections, FFN gates and up projections: Xavier uniform,
          optionally scaled by depth if num_layers > 12.
        - Attention outputs and FFN down projections: Normal distribution
          with std scaled by sqrt(2 * num_layers).
        - LM head: Normal with std 0.02.
        - Other linear layers: Xavier uniform.

    Biases:
        Set to zero if present.

    RMSNorm:
        Weight initialized to 1.0.
    """
        num_layers = self.model_args.num_layers
        init_std = 0.02
        
        # Initialize Embeddings
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            
        # Initialize linear layers
        elif isinstance(module, torch.nn.Linear):
            # Identify linear layers
            is_qkv = hasattr(module, '_is_qkv') or any(hasattr(p, 'w_qkv') and p.w_qkv is module for p in self.modules())
            is_attn_out = hasattr(module, '_is_attn_out') or any(hasattr(p, 'w_o') and p.w_o is module for p in self.modules())
            is_ffn_gate = hasattr(module, '_is_ffn_gate') or any(hasattr(p, 'weight1') and p.weight1 is module for p in self.modules())
            is_ffn_up = hasattr(module, '_is_ffn_up') or any(hasattr(p, 'weight3') and p.weight3 is module for p in self.modules())
            is_ffn_down = hasattr(module, '_is_ffn_down') or any(hasattr(p, 'weight2') and p.weight2 is module for p in self.modules())
            is_lm_head = hasattr(self, 'lm_head') and self.lm_head is module
            
            # Initialize input projections
            if is_qkv or is_ffn_gate or is_ffn_up:
                torch.nn.init.xavier_uniform_(module.weight)
                if num_layers > 12:
                    module.weight.data *= (1.0 / math.sqrt(num_layers / 6.0))
                    
            # Initialize output projections
            elif is_attn_out or is_ffn_down:
                scaled_std = init_std / math.sqrt(2 * num_layers)
                torch.nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
                
            # Initialize language modeling head
            elif is_lm_head:
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            
            # Default to Xavier initialization
            else:
                torch.nn.init.xavier_uniform_(module.weight)
            
            # Initialize bias
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
                
        # Initialize RMSNorm weight (scaling factor)
        elif hasattr(module, 'weight') and module.__class__.__name__ == 'RMSNorm':
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """Forward pass of the entire Transformer.
        
        This is the final forward pass of the entire Transformer. It takes in a 2 dimensional
        tensor, `input_ids` and returns the model's unnormalized predictions (logits) as a
        3 dimensional tensor, `logits`. These logits can then be normalized in the generation
        method, `generate` where the model can be used to autoregressively predict tokens. 
        Transformer layers will be stacked using gradient checkpointing if available if the
        gradient_checkpointing flag in the ModelArgs dataclass is set to true. Otherwise, it will
        simply pass through all transformer layers for `num_layers` times.

        Args:
            input_ids (torch.Tensor): Input tensor of shape [B, T].
            padding_mask (torch.Tensor): Padding tensor of shape [B, T].
            use_cache (bool): Whether to use KV caching during forward pass.

        Returns:
            Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
                - Output logits of shape [B, T, vocab_size].
                - List of cache dictionaries for each layer if use_cache is True, else None.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            # Ensure input_ids is a LongTensor (int64)
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()

            # Apply embeddings
            x = self.token_embed(input_ids) # [B, T, d_model]

            # Final dropout
            x = self.dropout(x)

            # Initialize KV cache outputs as list
            cache_outs = [] if use_cache else None

            # Stack transformer layers
            # Using i, layer because i will be used for indexing the layers (layer_idx) which
            # is used for the KV cache. layer is used as the temporary variable for the loop,
            # where self.layers are being looped through. self.layers is a stack of Transformer
            # blocks where the input_ids tensor will be passed through (after embeddings) and
            # dropout.
            for i, layer in enumerate(self.layers):
                if self.model_args.gradient_checkpointing:
                    x, cache_out = checkpoint(
                        layer, 
                        x, 
                        padding_mask, 
                        self.kv_cache,
                        i, 
                        use_cache, 
                        use_reentrant=False
                    )
                else:
                    x, cache_out = layer(
                        x, 
                        padding_mask=padding_mask, 
                        kv_cache=self.kv_cache, 
                        layer_idx=i, 
                        use_cache=use_cache
                    )
                
                # Get the KV tensors from each layer of the transformer and append them
                # to the cache only if the KV cache is being used in the first place.
                if use_cache:
                    cache_outs.append(cache_out)
            
            # Final RMSNorm
            x = self.rms_norm(x)

            # Final projection to logits
            logits = self.lm_head(x)

            return logits, cache_outs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with proper padding handling.
        
        Args:
            input_ids (torch.Tensor): Input token ids of shape [B, T].
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature (higher = more random).
            top_k (int): Keep only top k tokens for sampling.
            top_p (float): Nucleus sampling - keep tokens with cumulative prob <= top_p.
            do_sample (bool): Whether to use sampling or greedy decoding.
            pad_token_id (int, optional): Token ID used for padding.
            eos_token_id (int, optional): Token ID that signals end of sequence.
            attention_mask (torch.Tensor, optional): Mask to ignore padded tokens [B, T] where True=valid, False=pad.
            use_cache (bool): Whether to use KV caching.
            
        Returns:
            torch.Tensor: Generated token sequences of shape [B, T + max_new_tokens]
        """
        # Set default values
        if pad_token_id is None:
            pad_token_id = self.model_args.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.model_args.eos_token_id

        B, T = input_ids.shape
        device = input_ids.device

        # Create attention mask: True for valid tokens, False for padding
        if attention_mask is None:
            # Create padding mask where True values mean the token is valid and
            # the model should attend to it and False means the token will be padded.
            attention_mask = (input_ids != pad_token_id)

        # Clone input to avoid modifying original with shape of [B, T]
        generated_ids = input_ids.clone()

        # Track which sequences are unfinished
        # Creates a boolean tensor of shape [B] where True means the sequence is unfinished
        # and False means the sequence is finished. The tensor is initialized with ones meaning
        # it is filled with True values (since ones are True and zeros are False) because at the
        # start of generation, all sequences must be unfinished. 
        # NOTE: True means unfinished sequence, False means finished sequence.
        unfinished_sequences = torch.ones(B, dtype=torch.bool, device=device)

        self.eval()
        with torch.no_grad():
            # Initial forward pass to fill up cache
            if use_cache:
                self.kv_cache.reset() # None everything out
                self.kv_cache.initialize(B) # Initialize KV cache to zeros
                logits, _ = self.forward(generated_ids, attention_mask, use_cache=True) # Fill KV cache
                self.kv_cache.increment_seq_len(T) # Increment total amount of tokens stored by T
            else:
                # If no KV caching just return logits of shape [B, T, vocab_size]
                logits, _ = self.forward(generated_ids, attention_mask, use_cache=False)

            # Generating tokens for max_new_tokens to generate maximum sequence length outputs
            for _ in range(max_new_tokens):
                # generated_ids shape: [B, T], generated_ids dim 1 = T (sequence length)
                # Current number of generated tokens
                current_T = generated_ids.shape[1]

                # If maximum sequence length has been reached or succeeded then end generation
                if current_T >= self.model_args.max_seq_len:
                    break

                # Forward pass with attention mask if tokens are not done generating
                if unfinished_sequences.any():
                    with torch.amp.autocast(device_type=device.type, dtype=dtype):
                        # Create current attention mask with shape [B, current_T]
                        current_attention_mask = torch.ones(B, current_T, dtype=torch.bool, device=device)
                        # Ensure the mask length covers the amount of generated tokens
                        # attention_mask.shape[1] = original_T = input sequence length
                        # current_T = original_T + generated tokens
                        # This means: original_T <= current_T. This means the if condition will only
                        # occur when the original_T = current_T
                        if attention_mask.shape[1] < current_T:
                            # Fill the first original_T values with the original_T values from
                            # the original attention mask. This is because the original_T values
                            # from the original attention mask contains the padded and non padded,
                            # whereas the current attention mask only contains the valid tokens,
                            # since it only contains True values. This means it does not contain
                            # the padded tokens. Newly generated tokens do not need any padding.
                            current_attention_mask[:, :attention_mask.shape[1]] = attention_mask
                            current_attention_mask[:, attention_mask.shape[1]:] = unfinished_sequences.unsqueeze(1)
                        else:
                            # Copies the first current_T columns and applies to current_attention_mask.
                            # This will only work if original_T is >= current_T, otherwise, the tensor
                            # does not have enough space to store them and an error will be raised. 
                            #
                            # Simply put, the original tensor have enough space to account for these
                            # new elements.
                            current_attention_mask = attention_mask[:, :current_T]
    
                        # For caching, only process last token because the whole purpose of
                        # KV caching is to compute and key and value tensors once and then
                        # cache them. The if statement ensure that the cache is being used
                        # in the first place and also that the model has generated at least
                        # a single token. This is because T is the input sequence length and
                        # current_T is the input + generated tokens. In order for current_T
                        # to be greater than T, the model has had to have generated at least
                        # a single token.
                        if use_cache and current_T > T:
                            # Only process the last token for caching.
                            # contiguous is needed because tensor slicing changes the order
                            # of data in the tensor, but the memory is still the same. We 
                            # call contiguous to ensure no errors since in the attention module
                            # the view method was used which requires a contiguous tensor.
                            input_ids = generated_ids[:, -1:].contiguous()
                            # Same thing from above is done here.
                            current_attention_mask = current_attention_mask[:, -1:].contiguous()
                            # Generate new tokens
                            logits, _ = self.forward(input_ids, current_attention_mask, use_cache=True)
                            # Update by 1 because only one token is being processed.
                            self.kv_cache.increment_seq_len(1)
                        else:
                            # No caching, return logits of shape [B, T, vocab_size]
                            logits, _ = self.forward(generated_ids, current_attention_mask, use_cache=False)

                    # Get logits for the last position
                    # -1 in the sequence length dimension removes the dimension.
                    # The sequence length dimension is removed as when generating
                    # a new token as the sequence length would turned from a dimension
                    # to a single scalar value.
                    next_token_logits = logits[:, -1, :] # [B, vocab_size]

                    # Only process logits for unfinished sequences
                    # next_token_logits (float tensor) * unfinished sequences (boolean tensor) makes
                    # sure that unfinished sequences (True) keep their logits and finished sequences
                    # (False) get zeroed out. Unsqueezing is to make sure that element wise multiplication
                    # is compatible.
                    next_token_logits = next_token_logits * unfinished_sequences.unsqueeze(1).float() # [B, T] * [B, 1]

                    # Apply temperature scaling
                    if temperature < 0:
                        # Can't have negative temperature
                        raise ValueError(f"Temperature must be equal to or greater than 0, got {temperature}")
                    if temperature == 0:
                        # When temperature is 0, softmax approaches argmax behavior
                        do_sample = False
                    else:
                        # Sample for temperature > 0
                        next_token_logits = next_token_logits / temperature

                    # Apply top-k filtering
                    if top_k <= 0:
                        # We cannot have topk as 0 or negative values since we cannot sample 0 or a 
                        # negative amount of logits.
                        raise ValueError(f"top_k value must be greater than 0, got {top_k}")
                    if top_k == 1:
                        # We also ensure that top_k is greater than 1 because a topk value value of 1
                        # would mean we are sampling the largest logit and masking the rest of the
                        # logits out using negative infinity. This would mean that only the largest logit
                        # would have a probability score of 1, and the rest would be 0. This would be the
                        # same as doing argmax where we choose the single largest logit.
                        do_sample = False
                    if top_k > 1 and top_k < self.model_args.vocab_size:
                        # PyTorch's topk method returns a tuple of tensors of values and indices.
                        # Since we only want the logit values for our threshold, we will drop the
                        # indices. We can then pass in our next_token_logits tensor and the top_k
                        # value where it chooses the top k largest logits. The [0] is used to make
                        # sure that we get the logit values instead of the logit indices. The [:, -1]
                        # takes the smallest logit from tensor (since top_k uses descending order)
                        # where this smallest logit from the top k largest logits is used as the
                        # threshold to mask out other logits. Unsqueeze is to make sure that we can
                        # do a comparison to the next_tokens_logit tensor. This will only work if they
                        # both have the same number of dimensions. This is done so any values below
                        # this threshold will be masked out with negative infinity and when softmax is
                        # applied to negative infinity the probability will become 0. This is because
                        # exp(-inf) = 0. We also ensure that top_k is less than the vocabulary size since
                        # it would remove the purpose of sampling the top k logits.
                        threshold = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(1) # [B, 1]

                        # [B, vocab_size] < [B, 1]
                        indices_to_remove = next_token_logits < threshold
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Apply top-p filtering
                    if top_p < 1.0:
                        # Sort the logits in descending order. This helps rank the logits from highest to lowest.
                        # sorted_logits contains the logit values in sorted order, and sorted_indices contains their
                        # corresponding indices from the original tensor.
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)

                        # Convert the sorted logits to probability scores by applying softmax.
                        # The resulting tensor will contain probabilities in descending order.
                        probability_scores = F.softmax(sorted_logits, dim=-1)

                        # Compute the cumulative sum of the probability scores across the vocabulary dimension.
                        # This lets us know how many of the top tokens contribute to the cumulative probability.
                        cumulative_probs = torch.cumsum(probability_scores, dim=-1)

                        # Create a boolean mask where cumulative probabilities exceed top_p.
                        # These are the tokens we want to mask out (give zero probability).
                        sorted_indices_to_remove = cumulative_probs > top_p

                        # Shift the mask to the right by one so that we always retain the first token that crosses
                        # the threshold (instead of discarding it). This ensures at least one token is kept.
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0

                        # Scatter the mask back to the original logits' index space using the sorted indices.
                        # This remaps the boolean mask from sorted logits back to the original positions.
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

                        # Apply the mask by setting logits below the top_p threshold to negative infinity.
                        # These logits will become zero when softmax is applied during sampling.
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Sample or select next tokens
                    if do_sample:
                        probs = F.softmax(next_token_logits, dim=-1)
                        # Shape [B, num_samples] -> squeeze(1) -> [B]
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        # Choose highest probability token each time. This is done by argmaxing
                        # to get the largest logit. This is because the largest logit will yield
                        # the largest probability score and it will be selected anyway.
                        next_tokens = torch.argmax(next_token_logits, dim=-1)
                    
                    # Only update tokens for unfinished sequences, pad others
                    next_tokens = torch.where(unfinished_sequences, next_tokens, pad_token_id)

                    # Append the new tokens
                    generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1) # [B, T+1]

                    # Update attention mask for new tokens
                    # Adds a new column to determine if the token is finished or still generating.
                    attention_mask = torch.cat([attention_mask, unfinished_sequences.unsqueeze(1)], dim=1) # [B, T+1]

                    # Check for EOS tokens and update unfinished sequences
                    if eos_token_id is not None:
                        unfinished_sequences = unfinished_sequences & (next_tokens != eos_token_id)

                    # Break if all sequences are finished
                    if not unfinished_sequences.any():
                        break
        
        # Reset cache after generation
        if use_cache:
            self.kv_cache.reset()

        return generated_ids
