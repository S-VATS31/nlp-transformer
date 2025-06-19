import unittest
import torch

from src.model import FFNBlock, device, dtype
from configs.training_args import TrainingArgs
from configs.model_args.model_args_medium import ModelArgs

model_args = ModelArgs()
train_args = TrainingArgs()

class TestFFNBlock(unittest.TestCase):
    """Test cases for the feed forward network block module."""
    def setUp(self):
        self.ffn_block = FFNBlock(
            model_args.d_model, 
            model_args.d_ffn,
            model_args.dropout,
            model_args.rms_norm_eps
        ).to(device)
        self.T = 16
        self.x = torch.randn(train_args.batch_size, self.T, model_args.d_model, dtype=dtype).to(device)

    def test_output_shape(self):
        """Ensure output tensor has same shape as input tensor."""
        x_ffn_block = self.ffn_block(self.x)
        self.assertEqual(self.x.shape, x_ffn_block.shape)

    def test_no_dropout(self):
        """Ensure deterministic output with no dropout."""
        self.ffn_block.eval() # Turn off dropout
        x1 = self.ffn_block(self.x)
        x2 = self.ffn_block(self.x)
        self.assertTrue(torch.allclose(x1, x2))

    def test_dropout(self):
        """Ensure non-deterministic output with dropout."""
        x1 = self.ffn_block(self.x)
        x2 = self.ffn_block(self.x)
        self.assertFalse(torch.allclose(x1, x2))

    def test_numerical_stability(self):
        """Ensure output tensor is numerically stable."""
        x_ffn_block = self.ffn_block(self.x)
        self.assertTrue(torch.isfinite(x_ffn_block).all())

    def test_zero_seqlen(self):
        """Test FFN Block with sequence length of 0."""
        x = torch.randn(train_args.batch_size, 0, model_args.d_model, dtype=dtype).to(device)
        x_ffn_block = self.ffn_block(x)
        self.assertEqual(x.shape, x_ffn_block.shape)
        self.assertTrue(torch.isfinite(x_ffn_block).all())
    
    def test_empty_tensor(self):
        """Test FFN Block with empty tensor."""
        x = torch.empty(train_args.batch_size, self.T, model_args.d_model, dtype=dtype).to(device)
        x_ffn_block = self.ffn_block(x)
        self.assertEqual(x.shape, x_ffn_block.shape)
        self.assertTrue(torch.isfinite(x_ffn_block).all())

if __name__ == "__main__":
    unittest.main()
