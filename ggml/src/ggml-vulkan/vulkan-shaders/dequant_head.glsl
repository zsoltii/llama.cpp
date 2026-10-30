#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require

layout (push_constant) uniform parameter
{
    uint M;
    uint K;
    uint stride_a;
    uint stride_b;
    uint nel;
} p;

#include "types.glsl"

// Destination index of the first element of 32-element block ib.
// DEQUANT_TRANSPOSE (f16 scratch for FA with quantized KV): read [HS, NH, KV, NS], write [HS, KV, NH, NS].
// A block never crosses a head boundary (HS % 32 == 0), so the 32 elements stay contiguous.
uint dequant_block_dst(uint ib) {
    const uint e0 = ib * 32;
#ifdef DEQUANT_TRANSPOSE
    const uint HS = p.M, NH = p.K, KVn = p.stride_a;
    return (e0 % HS)
         + ((e0 / (HS * NH)) % KVn) * HS
         + ((e0 / HS) % NH) * (HS * KVn)
         + (e0 / (HS * NH * KVn)) * (HS * KVn * NH);
#else
    return e0;
#endif
}
