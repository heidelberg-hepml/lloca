"""PyTorch's modern and flexible flex_attention backend."""

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ModuleNotFoundError as err:
    raise ImportError(
        "torch>=2.5 is not installed. Run 'pip install lloca[flex-attention]'."
    ) from err

create_block_mask = create_block_mask

attention = flex_attention
