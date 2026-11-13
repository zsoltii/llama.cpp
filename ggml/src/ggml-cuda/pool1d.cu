#include "pool1d.cuh"

static __global__ void pool1d_nchw_kernel(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int nc = idx / ow;
    const int cur_ow = idx % ow;

    const float * i_ptr = src + nc * iw;
    float * o_ptr = dst + nc * ow;

    const int start = cur_ow * sw - pw;
    const int b = max(0, start);
    const int e = min(iw, start + kw);

    float res;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0.0f;     break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: return;
    }

    int count = 0;
    for (int i = b; i < e; i++) {
#if __CUDA_ARCH__ >= 350
        float cur = __ldg(i_ptr + i);
#else
        float cur = i_ptr[i];
#endif
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur;                break;
            case GGML_OP_POOL_MAX: res = max(res, cur);       break;
            default: break;
        }
        count++;
    }

    if (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : 0.0f;
    }

    o_ptr[cur_ow] = res;
}

// Non-overlapping windows without padding (e.g. the max over a whole row in build_amax_src1_f16): one block per
// output element reduces its window in parallel. With one thread per output, a 5120 wide window took 21.8 us.
template <int block_size>
static __global__ void pool1d_window_reduce_kernel(
        const int iw, const int ow, const int kw,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int out = blockIdx.x;           // nc*ow + cur_ow
    const int nc  = out / ow;
    const int b   = (out % ow) * kw;
    const int e   = min(iw, b + kw);

    const float * i_ptr = src + (int64_t) nc * iw;

    float res = op == GGML_OP_POOL_MAX ? -FLT_MAX : 0.0f;
    for (int i = b + threadIdx.x; i < e; i += block_size) {
        const float cur = i_ptr[i];
        res = op == GGML_OP_POOL_MAX ? fmaxf(res, cur) : res + cur;
    }

    res = op == GGML_OP_POOL_MAX ? warp_reduce_max(res) : warp_reduce_sum(res);
    if constexpr (block_size > WARP_SIZE) {
        __shared__ float buf[block_size / WARP_SIZE];
        if (threadIdx.x % WARP_SIZE == 0) {
            buf[threadIdx.x / WARP_SIZE] = res;
        }
        __syncthreads();
        if (threadIdx.x >= WARP_SIZE) {
            return;
        }
        res = threadIdx.x < block_size / WARP_SIZE ? buf[threadIdx.x] : (op == GGML_OP_POOL_MAX ? -FLT_MAX : 0.0f);
        res = op == GGML_OP_POOL_MAX ? warp_reduce_max(res) : warp_reduce_sum(res);
    }

    if (threadIdx.x == 0) {
        const int count = e - b;
        dst[out] = op == GGML_OP_POOL_AVG ? (count > 0 ? res / count : 0.0f) : res;
    }
}

static void pool1d_nchw_kernel_f32_f32_cuda(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {
    const int num_blocks = (parallel_elements + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool1d_nchw_kernel<<<block_nums, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(iw, ow, kw, sw, pw, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = src0->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t nr = ggml_nrows(src0);

    const int parallel_elements = (int)(nr * OW);

    if (p0 == 0 && k0 == s0 && k0 >= 64 && (op == GGML_OP_POOL_MAX || op == GGML_OP_POOL_AVG)) {
        constexpr int block_size = 256;
        pool1d_window_reduce_kernel<block_size><<<parallel_elements, block_size, 0, stream>>>(IW, OW, k0, src0_d, dst_d, op);
        return;
    }

    pool1d_nchw_kernel_f32_f32_cuda(IW, OW, k0, s0, p0, parallel_elements, src0_d, dst_d, op, stream);
}
