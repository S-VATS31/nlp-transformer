import torch

def setup():
    """Set up device, dtype (AMP), and Flash Attention 2 package."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Check Flash Attention 2 availability
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
        flash_attn_varlen_qkvpacked_func = None

    return device, dtype, FLASH_ATTN_AVAILABLE, flash_attn_varlen_qkvpacked_func
