import unittest
import torch

from src.model import RMSNorm, device, dtype
from configs.training_args import TrainingArgs
from configs.model_args.model_args_medium import ModelArgs

model_args = ModelArgs()
train_args = TrainingArgs()

class TestRMSNorm(unittest.TestCase):
    """Test root mean squared initialization module."""
    def setUp(self):
        self.rms_norm = RMSNorm(model_args.d_model, model_args.rms_norm_eps).to(device)
        self.T = 16 
        self.x = torch.randn(train_args.batch_size, self.T, model_args.d_model, dtype=dtype).to(device)

    def test_output_shape(self):
        """Ensure the output shape is the same as the input shape."""
        x_norm = self.rms_norm(self.x)
        self.assertEqual(self.x.shape, x_norm.shape)

    def test_numerical_stability(self):
        """Ensure the output tensor is numerically stable."""
        x_norm = self.rms_norm(self.x)
        self.assertTrue(torch.isfinite(x_norm).all())

    def test_deterministic_output(self):
        """Test if two forward passes return the same tensor."""
        x1 = self.rms_norm(self.x)
        x2 = self.rms_norm(self.x)
        self.assertTrue(torch.allclose(x1, x2))

    def test_zero_seqlen(self):
        """Test RMSNorm with sequence length of 0."""
        x_zero = torch.randn(train_args.batch_size, 0, model_args.d_model, dtype=dtype).to(device)
        x_norm = self.rms_norm(x_zero)
        self.assertEqual(x_zero.shape, x_norm.shape)
        self.assertTrue(torch.isfinite(x_norm).all())

    def test_zero_epsilon(self):
        """Test epsilon value of 0 to check numerical stability."""
        rms_norm_zero = RMSNorm(model_args.d_model, 0).to(device)
        x_norm = rms_norm_zero(self.x)
        self.assertEqual(self.x.shape, x_norm.shape)
        self.assertTrue(torch.isfinite(x_norm).all())

    def test_device_consistency(self):
        """Ensure output is consistent across devices."""
        if torch.cuda.is_available():
            rms_norm_cpu = RMSNorm(model_args.d_model, model_args.rms_norm_eps).cpu()
            rms_norm_cuda = RMSNorm(model_args.d_model, model_args.rms_norm_eps).cuda()
            x_cpu = torch.randn(train_args.batch_size, self.T, model_args.d_model, dtype=dtype).cpu()
            x_cuda = torch.randn(train_args.batch_size, self.T, model_args.d_model, dtype=dtype).cuda()
            x_norm_cpu = rms_norm_cpu(x_cpu)
            x_norm_cuda = rms_norm_cuda(x_cuda)
            self.assertTrue(torch.allclose(x_norm_cpu, x_norm_cuda))

    def test_backward_pass(self):
        """Ensure backward pass computes gradients."""
        self.x.requires_grad_()
        x_norm = self.rms_norm(self.x)
        loss = x_norm.sum()
        loss.backward()
        self.assertIsNotNone(self.x.grad)
        self.assertTrue(torch.isfinite(self.x.grad).all())

if __name__ == "__main__":
    unittest.main()
