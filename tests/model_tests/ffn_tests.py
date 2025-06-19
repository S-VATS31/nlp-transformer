import unittest
import torch

from src.model import FFN, device, dtype
from configs.training_args import TrainingArgs
from configs.model_args.model_args_small import ModelArgs

model_args = ModelArgs()
train_args = TrainingArgs()

class TestFFN(unittest.TestCase):
    """Test feed forward network module."""
    def setUp(self):
        self.ffn = FFN(model_args.d_model, model_args.d_ffn, model_args.dropout).to(device)
        self.T = 16
        self.x = torch.randn(train_args.batch_size, self.T, model_args.d_model, dtype=dtype).to(device)
    
    def test_output_shape(self):
        """Ensure output tensor's shape matches input tensor shape."""
        x_ffn = self.ffn(self.x)
        self.assertEqual(self.x.shape, x_ffn.shape)

    def test_determinstic_output(self):
        """Ensure two forward passes return the same tensor."""
        self.ffn.eval() # Turn off dropout
        x1 = self.ffn(self.x)
        x2 = self.ffn(self.x)
        self.assertTrue(torch.allclose(x1, x2))
    
    def test_dropout(self):
        """Ensure two forward passes don't return the same tensor with dropout enabled."""
        x1 = self.ffn(self.x)
        x2 = self.ffn(self.x)
        self.assertFalse(torch.allclose(x1, x2))

    def test_zero_seqlen(self):
        """Test input tensor with sequence length of 0."""
        x_zero = torch.randn(train_args.batch_size, 0, model_args.d_model, dtype=dtype).to(device)
        x_ffn = self.ffn(x_zero)
        self.assertTrue(x_zero.shape, x_ffn.shape)
        self.assertTrue(torch.isfinite(x_ffn).all())
    
    def test_numerical_stability(self):
        """Ensure the output tensor is numerically stable."""
        x_ffn = self.ffn(self.x)
        self.assertTrue(torch.isfinite(x_ffn).all())
              
    def test_backward_pass(self):
        """Ensure backward pass computes gradients."""
        self.x.requires_grad_()
        x_ffn = self.ffn(self.x)
        loss = x_ffn.sum()
        loss.backward()
        self.assertIsNotNone(self.x.grad)
        self.assertTrue(torch.isfinite(self.x.grad).all())

    def test_weight_shapes(self):
        """Ensure all 3 weight matrices have the correct shape."""
        assert self.ffn.weight1.weight.shape == (model_args.d_ffn, model_args.d_model)
        assert self.ffn.weight2.weight.shape == (model_args.d_model, model_args.d_ffn)
        assert self.ffn.weight3.weight.shape == (model_args.d_ffn, model_args.d_model)

    def test_weight_are_params(self):
        """Ensure all 3 weights matrices are learnable parameters."""
        params = dict(self.ffn.named_parameters())
        assert "weight1.weight" in params
        assert "weight2.weight" in params
        assert "weight3.weight" in params

if __name__ == "__main__":
    unittest.main()
