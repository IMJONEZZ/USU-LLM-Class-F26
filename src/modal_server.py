MIB = 2**20


def kv_cache_mib(layers, kv_heads, head_dim, tokens, bytes_per_value=2):
    """Memory for the attention cache: one key and one value vector per layer, kv head and token."""
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens / MIB
