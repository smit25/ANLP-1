from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device

    half = head_dim//2
    dims = torch.arange(head_dim // 2).float()
    inv_freq = 1.0 / (theta ** (2*dims / (head_dim))).to(device)
    t = torch.arange(seqlen).float()
    freq_cis = torch.outer(t, inv_freq).to(device)

    freq_cis_expanded = freq_cis.repeat_interleave(2, dim=-1)  # expanded to 
    freq_cis_expanded = reshape_for_broadcast(freq_cis_expanded, query)

    # Please refer to Lecture 5 slides in https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # print(query_real.shape)
    # # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # # First, compute the trigonometric values in the second and fourth columns in
    # # slide 49 (linked above).
    cos = torch.cos(freq_cis_expanded)
    sin = torch.sin(freq_cis_expanded)

    cos_pairs = cos[..., ::2]
    sin_pairs = sin[..., ::2]

    # # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # # key_real, and key_imag.

    query_out_real = query_real * cos_pairs - query_imag * sin_pairs
    query_out_imag = query_real * sin_pairs + query_imag * cos_pairs
    query_out = torch.cat([query_out_real.unsqueeze(-1), query_out_imag.unsqueeze(-1)], dim=-1).flatten(-2)

    key_out_real = key_real * cos_pairs - key_imag * sin_pairs
    key_out_imag = key_real * sin_pairs + key_imag * cos_pairs
    key_out = torch.cat([key_out_real.unsqueeze(-1), key_out_imag.unsqueeze(-1)], dim=-1).flatten(-2)

    return query_out, key_out