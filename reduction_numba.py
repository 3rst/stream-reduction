import time
import numpy as np
from numba import cuda
from numba.types import int32

# reduction.kernel.begin
# optimized reduction kernel (Not the native kernel given in numba docs)
@cuda.jit
def array_sum(data, psum):
    shr = cuda.shared.array(1024, int32)  # Shared memory allocation
    t = cuda.threadIdx.x
    size = len(data)
    start = 2 * cuda.blockIdx.x * cuda.blockDim.x

    if start + t < size:
        shr[t] = data[start + t]
    else:
        shr[t] = 0

    if start + cuda.blockDim.x + t < size:
        shr[cuda.blockDim.x + t] = data[start + cuda.blockDim.x + t]
    else:
        shr[cuda.blockDim.x + t] = 0

    s = cuda.blockDim.x
    while s > 0:
        cuda.syncthreads()
        # if t < s:
        if t < s and (t + s) < 2 * cuda.blockDim.x:  # Prevent accessing invalid shared memory
            shr[t] += shr[t + s]
        s //= 2

    if t == 0:
        psum[cuda.blockIdx.x] = shr[t]
# reduction.kernel.end

def generate_input(insize):
    max_int32_safe = 45000  # Since sum(range(46341)) ≈ 2.1 billion (fits in int32)
    full_repeats = insize // max_int32_safe  # Number of full cycles
    remainder = insize % max_int32_safe  # Remaining numbers

    # Create repeated cycles of 1 to max_int32_safe
    input_arr = np.concatenate([np.arange(1, max_int32_safe + 1, dtype=np.int32)] * full_repeats)
    
    # Add remaining numbers if needed
    if remainder > 0:
        input_arr = np.concatenate([input_arr, np.arange(1, remainder + 1, dtype=np.int32)])

    return input_arr

def run_optimized_reduction(insize):
    nthreads = 512
    seg_size = 3000000  # Size of segment processed per stream
    grid_size = (seg_size + (2 * nthreads) - 1) // (2 * nthreads)
    # grid_size = (seg_size // (2 * nthreads))
    print(f"Grid size is: {grid_size}")

    # **Adjust input size to be a multiple of 3 * seg_size**
    # remainder = insize % (3 * seg_size)
    # if remainder != 0:
    #     insize += (3 * seg_size - remainder)  # Round up to nearest multiple

    stream0 = cuda.stream()
    stream1 = cuda.stream()
    stream2 = cuda.stream()

    # input_arr = np.arange(insize, dtype=np.int32)
    input_arr = generate_input(insize)
    psum_host = np.zeros(grid_size, dtype=np.int32)
    # **Start total execution timer**
    total_start = time.time()
    
    a0 = cuda.device_array(seg_size, dtype=np.int32)
    a1 = cuda.device_array(seg_size, dtype=np.int32)
    a2 = cuda.device_array(seg_size, dtype=np.int32)
    
    b0 = cuda.device_array_like(psum_host)
    b1 = cuda.device_array_like(psum_host)
    b2 = cuda.device_array_like(psum_host)

    total_sum = 0
    kernel_time = 0
    
    for i in range(0, insize, 3 * seg_size):
        end0 = min(i + seg_size, insize)
        # if end0 >= insize: break  # Stop if out of bounds
        end1 = min(i + 2 * seg_size, insize)
        # if end1 >= insize: break
        end2 = min(i + 3 * seg_size, insize)
        # if end2 >= insize: break

        # **Transfer Data to GPU Concurrently**
        cuda.to_device(input_arr[i:end0], to=a0, stream=stream0)
        if end1 > end0:
            cuda.to_device(input_arr[end0:end1], to=a1, stream=stream1)
        if end2 > end1:
            cuda.to_device(input_arr[end1:end2], to=a2, stream=stream2)

        # **Launch Kernels Concurrently**
        array_sum[grid_size, nthreads, stream0](a0, b0)
        if end1 > end0:
            array_sum[grid_size, nthreads, stream1](a1, b1)
        if end2 > end1:
            array_sum[grid_size, nthreads, stream2](a2, b2)

        # **Per-stream synchronization instead of full `cuda.synchronize()`**
        stream0.synchronize()
        if end1 > end0:
            stream1.synchronize()
        if end2 > end1:
            stream2.synchronize()

        b0_h = b0.copy_to_host(stream=stream0)
        total_sum += sum(b0_h)
        if end1 > end0:
            b1_h = b1.copy_to_host(stream=stream1)
            total_sum += sum(b1_h)
        if end2 > end1:
            b2_h = b2.copy_to_host(stream=stream2)
            total_sum += sum(b2_h)

    total_end = time.time()
    total_time = (total_end - total_start) * 1000  # Convert to milliseconds
    # Compute expected sum
    num_cycles = insize // 45000
    remainder = insize % 45000
    cycle_sum = (45000 * (45000 + 1)) // 2  # Formula for sum(1 to 45000)
    remainder_sum = (remainder * (remainder + 1)) // 2  # Sum(1 to remainder)
    expected_sum = num_cycles * cycle_sum + remainder_sum
    # expected_ans = sum(input_arr)

    print(f"Kernel execution time: {kernel_time:.3f} ms")
    print(f"Total reduction time (including memory transfers): {total_time:.3f} ms")

def run_naive_reduction_shared(insize):
    """Runs the naive reduction using shared memory & records execution time"""
    nthreads = 512
    nblocks = (insize + nthreads - 1) // nthreads  # Compute number of blocks

    # **Allocate Host & Device Memory**
    input_arr = generate_input(insize)
    psum_host = np.zeros(nblocks, dtype=np.int32)  # Partial sums

    # **Start Total Execution Timer**
    total_start = time.time()
    d_input = cuda.to_device(input_arr)  # Copy input to GPU
    d_psum = cuda.device_array(nblocks, dtype=np.int32)  # Partial sums on GPU

    # **CUDA Events for Timing**
    start_event = cuda.event()
    end_event = cuda.event()

    # **Start Kernel Execution Timer**
    start_event.record()

    # **Launch Naïve Shared Memory Kernel**
    array_sum[nblocks, nthreads](d_input, d_psum)

    # **End Kernel Execution Timer**
    end_event.record()
    end_event.synchronize()

    # **Copy Partial Sums Back to CPU**
    d_psum.copy_to_host(psum_host)
    # **Final Reduction on CPU**
    total_sum = np.sum(psum_host, dtype=np.int64)  # Ensure correct sum computation
    # **Stop Total Execution Timer**
    total_end = time.time()
    
    total_time = (total_end - total_start) * 1000  # Convert to milliseconds
    kernel_time = cuda.event_elapsed_time(start_event, end_event)  # In milliseconds

    # **Compute Expected Sum Safely**
    num_cycles = insize // 45000
    remainder = insize % 45000

    cycle_sum = (45000 * (45000 + 1)) // 2  
    remainder_sum = (remainder * (remainder + 1)) // 2  

    expected_sum = num_cycles * cycle_sum + remainder_sum

    # **Print Results**
    print(f"Kernel execution time: {kernel_time:.3f} ms")
    print(f"Total Reduction Time (including memory transfers): {total_time:.3f} ms")
    # print(f"Final sum: {total_sum}")
    # print(f"Expected sum: {expected_sum}")
    
if __name__ == "__main__":
    input_size = 1500000000   # Change this as needed
    run_optimized_reduction(input_size)
    print("\nRunning naive reduction now\n\n")
    run_naive_reduction_shared(input_size)