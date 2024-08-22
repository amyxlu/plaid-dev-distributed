from plaid.denoisers.modules._attention import Attention
import torch
import torch.utils.benchmark as benchmark
import psutil


B, L, C = 4, 8, 1024
kwargs = dict(device="cuda", dtype=torch.bfloat16)

x = torch.randn(B, L, C, **kwargs)
mask = torch.tensor(
    [
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ],
)
mask = mask.bool().cuda()

attn = Attention(dim=C, heads=16, use_xformers=True).to(**kwargs)

# Test forward class
_ = attn(x, mask)

# Check if outputs are similar
xformers_output = attn.xformers_attention(x, mask)
standard_output = attn.attention(x, mask)

assert torch.allclose(xformers_output, standard_output, atol=1e-2), "Outputs are not similar!"

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

# Benchmark standard attention
standard_time, standard_memory = measure_time_and_memory(attn.attention)
print(f"Standard Multihead Attention - Time: {standard_time:.4f}s, Memory: {standard_memory:.2f}MB")

# Benchmark lightweight attention
lightweight_time, lightweight_memory = measure_time_and_memory(attn.xformers_attention)
print(f"xFormers Attention - Time: {lightweight_time:.4f}s, Memory: {lightweight_memory:.2f}MB")
