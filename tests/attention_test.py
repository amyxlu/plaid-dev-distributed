import torch
from torch import nn
import torch.utils.benchmark as benchmark
import psutil

from plaid.denoisers.modules._attention import Attention
from plaid.datasets import get_test_sharded_batch

C = 1024
kwargs = dict(device="cuda", dtype=torch.bfloat16)

batch = get_test_sharded_batch()
x, mask, go_idx, organism_idx, pfam_id, sample_id, local_path = batch
B, L, _ = x.shape

x = x.to(**kwargs)
mask = mask.bool().cuda()

# arbitrary linear projection
linear = nn.Linear(x.shape[-1], C).to(**kwargs)
torch.nn.init.xavier_uniform_(linear.weight)
x = linear(x)
x += torch.randn_like(x)
print(x.shape)

attn = Attention(dim=C, heads=16, attention_mode="xformers_scaled_dot_product").to(**kwargs)

# Test forward class
_ = attn(x, mask)

# Check if outputs are similar
xformers_memory_efficient_output = attn.xformers_memory_efficient_attention(x, mask).cpu()
xformers_scaled_dot_product_output = attn.xformers_scaled_dot_product_attention(x, mask).cpu()
standard_output = attn.standard_attention(x, mask).cpu()
flash_output = attn.flash_attention_padded(x, mask).cpu()


print("standard_output", standard_output[:2])
print("xformers_memory_efficient_output", xformers_memory_efficient_output[:2])
print("xformers_scaled_dot_product_output", xformers_scaled_dot_product_output[:2])
print("flash_output", flash_output[:2])
print("mask", mask[:2])


# assert torch.allclose(xformers_memory_efficient_output, standard_output, atol=1e-2), "Outputs are not similar!"

# Function to benchmark
def benchmark_attention(attention_fn):
    def forward_pass():
        return attention_fn(x, mask)
    return forward_pass


# Measure time and memory usage
def measure_time_and_memory(fn):
    # Warm-up
    _ = fn(x, mask)
    
    # Measure time using torch.utils.benchmark
    timer = benchmark.Timer(
        stmt="fn(x, mask)",
        globals={"fn": fn, "x": x, "mask": mask}
    )
    
    time_results = timer.timeit(100)
    
    # Measure memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = fn(x, mask)
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    else:
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    
    return time_results.median, memory_usage

standard_time, standard_memory = measure_time_and_memory(attn.standard_attention)
print(f"Standard Multihead Attention - Time: {standard_time:.4f}s, Memory: {standard_memory:.2f}MB")

xformer_mem_eff_time, xformer_mem_eff_memory = measure_time_and_memory(attn.xformers_memory_efficient_attention)
print(f"xFormers Memory Efficient Attention - Time: {xformer_mem_eff_time:.4f}s, Memory: {xformer_mem_eff_memory:.2f}MB")

xformer_sdp_time, xformer_sdp_memory = measure_time_and_memory(attn.xformers_scaled_dot_product_attention)
print(f"xFormers Scaled Dot Product Attention - Time: {xformer_sdp_time:.4f}s, Memory: {xformer_sdp_memory:.2f}MB")

flashattn_time, flashattn_memory = measure_time_and_memory(attn.flash_attention_padded)
print(f"Flash Attention - Time: {flashattn_time:.4f}s, Memory: {flashattn_memory:.2f}MB")

import IPython; IPython.embed()