import unittest
import torch

from src.model import AttentionBlock, device, dtype
from configs.training_args import TrainingArgs
from configs.model_args.model_args_small import ModelArgs

model_args = ModelArgs()
train_args = TrainingArgs()

class TestAttentionBlock(unittest.TestCase):
    """Test attention block module."""

    def setUp(self):
        self.attention_block = AttentionBlock(
            model_args.d_model,
            model_args.num_heads,
            model_args.query_groups,
            model_args.dropout,
            model_args.rope_base,
            model_args.rms_norm_eps
        ).to(device)

        self.T = 16
        self.x = torch.randn(train_args.batch_size, self.T, model_args.d_model, dtype=dtype).to(device)

    def test_output_shape(self):
        """Ensure output tensor has same shape as input tensor."""
        x_attn_block, _ = self.attention_block(self.x)
        self.assertEqual(self.x.shape, x_attn_block.shape)

    def test_no_dropout(self):
        """Ensure two forward pass tensors are the same with no dropout."""
        self.attention_block.eval()  # Turn off dropout
        x1, _ = self.attention_block(self.x)
        x2, _ = self.attention_block(self.x)
        self.assertTrue(torch.allclose(x1, x2))

    def test_dropout(self):
        """Ensure two forward pass tensors are not the same with dropout."""
        x1, _ = self.attention_block(self.x)
        x2, _ = self.attention_block(self.x)
        self.assertFalse(torch.allclose(x1, x2))

    def test_gradients_propagate(self):
        """Ensure gradients flow through the block."""
        self.x.requires_grad_()
        out, _ = self.attention_block(self.x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(self.x.grad)
        self.assertEqual(self.x.grad.shape, self.x.shape)

    def test_kv_cache_keys_shape(self):
        """Check KV cache 'k' tensor has correct shape."""
        _, cache = self.attention_block(self.x, use_cache=True, layer_idx=0)
        k = cache["k"]
        B, H, T, D = k.shape
        self.assertEqual(B, train_args.batch_size)
        self.assertEqual(H, model_args.num_heads)
        self.assertEqual(T, self.T)
        self.assertEqual(D * H, model_args.d_model)

    def test_kv_cache_values_shape(self):
        """Check KV cache 'v' tensor has correct shape."""
        _, cache = self.attention_block(self.x, use_cache=True, layer_idx=0)
        v = cache["v"]
        B, H, T, D = v.shape
        self.assertEqual(B, train_args.batch_size)
        self.assertEqual(H, model_args.num_heads)
        self.assertEqual(T, self.T)
        self.assertEqual(D * H, model_args.d_model)

    def test_forward_with_none_padding_mask(self):
        """Ensure attention block works when padding_mask is None."""
        out, _ = self.attention_block(self.x, padding_mask=None)
        self.assertEqual(out.shape, self.x.shape)

    def test_forward_with_all_zero_padding_mask(self):
        """Ensure attention block runs with all positions masked out."""
        padding_mask = torch.zeros(train_args.batch_size, self.T, dtype=torch.bool, device=device)
        out, _ = self.attention_block(self.x, padding_mask=padding_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_forward_with_all_one_padding_mask(self):
        """Ensure attention block runs with no positions masked."""
        padding_mask = torch.ones(train_args.batch_size, self.T, dtype=torch.bool, device=device)
        out, _ = self.attention_block(self.x, padding_mask=padding_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_nan_input_fails_gracefully(self):
        """Check that NaN inputs do not crash the block."""
        x_nan = torch.full_like(self.x, float("nan"))
        out, _ = self.attention_block(x_nan)
        self.assertTrue(torch.isnan(out).any())

    def test_dropout_mean_change(self):
        """Check that dropout statistically changes output mean."""
        self.attention_block.train()
        x1, _ = self.attention_block(self.x)
        x2, _ = self.attention_block(self.x)
        mean_diff = (x1 - x2).abs().mean().item()
        self.assertGreater(mean_diff, 0.001)

if __name__ == "__main__":
    unittest.main()
