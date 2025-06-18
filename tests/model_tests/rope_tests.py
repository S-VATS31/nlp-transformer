import unittest
import torch

from src.model import RoPE, device, dtype
from configs.training_args import TrainingArgs
from configs.model_args.model_args_medium import ModelArgs

model_args = ModelArgs()
train_args = TrainingArgs()

class TestRoPE(unittest.TestCase):
    """Test rotary positional embeddings module."""
    def setUp(self):
        """Set up parameters which will be used in tests cases."""
        self.head_dim = model_args.d_model // model_args.num_heads
        self.B, self.T = train_args.batch_size, 16
        self.rope = RoPE(head_dim=self.head_dim, theta=model_args.rope_base).to(device)
        self.x = torch.randn(self.B, self.T, model_args.num_heads, self.head_dim, dtype=dtype).to(device)

    def test_output_shape(self):
        """Ensure output tensor's shape is equal to input tensor's shape."""
        x_rope = self.rope(self.x)
        self.assertEqual(x_rope.shape, self.x.shape)

    def test_deterministic_output(self):
        """Ensure two forward passes have the same output."""
        x1 = self.rope(self.x)
        x2 = self.rope(self.x)
        self.assertTrue(torch.allclose(x1, x2))

    def test_value_error(self):
        """Ensure a ValueError is raised when head_dim is not divisible by 2."""
        head_dim = 3
        with self.assertRaises(ValueError):
            RoPE(head_dim, model_args.rope_base).to(device)

    def test_numerical_stability(self):
        """Ensure the output tensor only contains finite values."""
        x_rope = self.rope(self.x)
        self.assertTrue(torch.isfinite(x_rope).all())

    def test_buffers(self):
        """Ensure buffers are found in the RoPE module."""
        self.assertIn("inv_freq", dict(self.rope.named_buffers()))
        self.assertIn("cos_cache", dict(self.rope.named_buffers()))
        self.assertIn("sin_cache", dict(self.rope.named_buffers()))

    def test_device_consistency(self):
        """Ensure same output on different devices."""
        if torch.cuda.is_available():
            rope_cpu = RoPE(self.head_dim, model_args.rope_base).cpu()
            rope_cuda = RoPE(self.head_dim, model_args.rope_base).cuda()
            x_cpu = self.x.cpu()
            x_cuda = self.x.cuda()
            out_cpu = rope_cpu(x_cpu)
            out_cuda = rope_cuda(x_cuda).cpu()
            self.assertTrue(torch.allclose(out_cpu, out_cuda))

    def test_different_seqlen(self):
        """Test forward pass with different sequence lengths."""
        T_vals = [1, 2, 4, 8, 16, 32, 64, 128]
        for T in T_vals:
            x = torch.randn(self.B, T, model_args.num_heads, self.head_dim, dtype=dtype).to(device)
            x_rope = self.rope(x)
            self.assertEqual(x.shape, x_rope.shape)
            self.assertTrue(torch.isfinite(x_rope).all())
          
    def test_backward_pass(self):
    """Ensure backward pass computes gradients."""
    self.x.requires_grad_()
    out = self.rope(self.x)
    loss = out.sum()
    loss.backward()
    self.assertIsNotNone(self.x.grad)
    self.assertTrue(torch.isfinite(self.x.grad).all())

if __name__ == "__main__":
    unittest.main()
