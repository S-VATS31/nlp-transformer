import unittest
import torch

from src.model import Attention, device, dtype
from configs.training_args import TrainingArgs
from configs.model_args.model_args_medium import ModelArgs

model_args = ModelArgs()
train_args = TrainingArgs()

class TestAttention(unittest.TestCase):
    """Test the attention module containing GQA and Flash Attention."""
    def setUp(self):
        self.attention = Attention(model_args.d_model, model_args.num_heads, model_args.query_groups, model_args.rope_base).to(device)
        self.T = 16
        self.x = torch.randn(train_args.batch_size, self.T, model_args.d_model, dtype=dtype).to(device)
    
    def test_output_shape(self):
        """Ensure output has the same shape as input."""
        x_attn, _ = self.attention(self.x)
        self.assertEqual(self.x.shape, x_attn.shape)

    def test_d_model_not_divisble(self):
        """Ensure a ValueError is raised when `d_model` is not divisble by `num_heads`."""
        d_model = 500
        with self.assertRaises(ValueError):
            Attention(d_model, model_args.num_heads, model_args.num_heads, model_args.rope_base).to(device)

    def test_head_dim_not_divisble(self):
        """Ensure a ValueError is raised when `num_heads` is not divisible by `query_groups`."""
        query_groups = 5
        with self.assertRaises(ValueError):
            Attention(model_args.d_model, model_args.num_heads, query_groups, model_args.rope_base).to(device)

    def test_numerical_stability(self):
        """Ensure output tensor is numerically stable."""
        x_attn, _ = self.attention(self.x)
        self.assertTrue(torch.isfinite(x_attn).all())

    def test_deterministic_output(self):
        """Ensure two forward passes give the same output."""
        x1, _ = self.attention(self.x)
        x2, _ = self.attention(self.x)
        self.assertTrue(torch.allclose(x1, x2))
    
    def test_input_tensor_two_dim(self):
        """Ensure value error is raised when input tensor has less than 3 dimensions."""
        x_wrong_dim = torch.randn(train_args.batch_size, self.T, dtype=dtype).to(device)
        with self.assertRaises(ValueError):
            self.attention(x_wrong_dim)

    def test_input_tensor_four_dim(self):
        """Ensure value error is raised when input tensor has more than 3 dimensions."""
        x_wrong_dim = torch.randn(train_args.batch_size, self.T, model_args.d_model, 4, dtype=dtype).to(device)
        with self.assertRaises(ValueError):
            self.attention(x_wrong_dim)
    
    def test_zero_seqlen(self):
        """Ensure empty tensor is returned when sequence length is 0."""
        x = torch.randn(train_args.batch_size, 0, model_args.d_model, dtype=dtype).to(device)
        x_attn, _ = self.attention(x)
        self.assertIsInstance(x_attn, torch.Tensor)
        self.assertEqual(x_attn.shape[1], 0)

    def test_forward_no_causal_masking(self):
        """Ensure forward works when causal masking is disabled."""
        out, cache = self.attention(self.x, causal=False)
        self.assertEqual(out.shape, self.x.shape)
        self.assertIsNone(cache)

    def test_forward_with_padding_mask(self):
        """Ensure forward works correctly with a padding mask."""
        padding_mask = torch.ones(train_args.batch_size, self.T, dtype=torch.bool, device=device)
        # Randomly zero out some positions to simulate padding
        padding_mask[0, -2:] = 0
        out, cache = self.attention(self.x, padding_mask=padding_mask)
        self.assertEqual(out.shape, self.x.shape)
        self.assertIsNone(cache)

    def test_zero_length_sequence_raises_empty_output(self):
        """Ensure zero sequence length returns empty tensor with correct shape."""
        x_zero = torch.randn(train_args.batch_size, 0, model_args.d_model, dtype=dtype).to(device)
        out, cache = self.attention(x_zero)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[1], 0)
        self.assertIsNone(cache)

    def test_output_dtype_matches_input(self):
        """Ensure the output tensor has the same dtype as input."""
        out, _ = self.attention(self.x)
        self.assertEqual(out.dtype, self.x.dtype)

    def test_rope_applied_shapes(self):
        """Check that RoPE returns expected shapes for q and k."""
        with torch.no_grad():
            qkv = self.attention.w_qkv(self.x)
            q, kv = torch.split(qkv, [self.attention.num_heads * self.attention.head_dim, 2 * self.attention.query_groups * self.attention.head_dim], dim=-1)
            q = q.view(train_args.batch_size, self.T, self.attention.num_heads, self.attention.head_dim)
            k, _ = torch.chunk(kv, 2, dim=-1)
            k = k.view(train_args.batch_size, self.T, self.attention.query_groups, self.attention.head_dim)

            q_rope = self.attention.rope(q)
            k_rope = self.attention.rope(k)

            self.assertEqual(q_rope.shape, (train_args.batch_size, self.T, self.attention.num_heads, self.attention.head_dim))
            self.assertEqual(k_rope.shape, (train_args.batch_size, self.T, self.attention.query_groups, self.attention.head_dim))
