/*
 * sm3_hybrid_arm64.c — ARM64 NEON 单流“SIMD 寄存器 + 通用寄存器”混合实现
 *
 * 与 x86 单流混合实现同构：
 *   - 消息扩展：通用寄存器（A32 的标量通路）先解近距递推链头 W[base]，
 *     NEON 128bit 寄存器随后一次算出 W[base..base+3]（P1、ROTL15、ROTL7
 *     与异或全部向量化，ROTL 由 vshl/vshr/vorr 组合实现）；
 *   - 迭代压缩：A..H 串行状态链由通用寄存器执行；
 *   - 字节序：vld1q_u8 + vrev32q_u8 完成大端转换，对任意对齐安全。
 */
#if defined(__aarch64__)

#include "sm3.h"
#include <arm_neon.h>
#include <string.h>

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define P0(x) ((x) ^ ROTL32(x, 9)  ^ ROTL32(x, 17))
#define P1(x) ((x) ^ ROTL32(x, 15) ^ ROTL32(x, 23))

/* NEON 旋转与置换（ARMv8 NEON 无 u32 循环移位单指令，由移位+或合成） */
#define VROL(x, n) vorrq_u32(vshlq_n_u32(x, n), vshrq_n_u32(x, 32 - (n)))
#define VP1(x)     veorq_u32(veorq_u32((x), VROL(x, 15)), VROL(x, 23))

void sm3_compress_hybrid_neon(uint32_t state[8], const uint8_t block[64])
{
    uint32_t W[68];
    uint32_t A, B, C, D, E, F, G, H;
    int j, base;

    /* ---------- 阶段一：NEON 装载并字节序转换 W[0..15] ---------- */
    for (j = 0; j < 16; j += 4) {
        uint8x16_t  vb = vld1q_u8(block + 4 * j);
        uint32x4_t  v  = vreinterpretq_u32_u8(vrev32q_u8(vb));
        vst1q_u32(W + j, v);
    }

    /* ---------- 阶段二：混合消息扩展 W[16..67] ---------- */
    for (base = 16; base < 68; base += 4) {
        /* (1) 通用寄存器先解近距依赖链头 */
        uint32_t w0 = P1(W[base-16] ^ W[base-9] ^ ROTL32(W[base-3], 15))
                    ^ ROTL32(W[base-13], 7) ^ W[base-6];

        /* (2) NEON 一次算 4 个 W */
        uint32_t   v3arr[4] = { W[base-3], W[base-2], W[base-1], w0 };
        uint32x4_t v1 = vld1q_u32(W + base - 16);
        uint32x4_t v2 = vld1q_u32(W + base -  9);
        uint32x4_t v3 = vld1q_u32(v3arr);
        uint32x4_t v4 = vld1q_u32(W + base - 13);
        uint32x4_t v5 = vld1q_u32(W + base -  6);
        uint32x4_t t  = veorq_u32(veorq_u32(v1, v2), VROL(v3, 15));
        t = VP1(t);
        t = veorq_u32(t, veorq_u32(VROL(v4, 7), v5));
        vst1q_u32(W + base, t);
    }

    /* ---------- 阶段三：通用寄存器执行 64 轮迭代压缩 ---------- */
    A = state[0]; B = state[1]; C = state[2]; D = state[3];
    E = state[4]; F = state[5]; G = state[6]; H = state[7];

    for (j = 0; j < 64; j++) {
        uint32_t Tj  = (j < 16) ? 0x79cc4519u : 0x7a879d8au;
        uint32_t SS1 = ROTL32(ROTL32(A, 12) + E + ROTL32(Tj, j & 31), 7);
        uint32_t SS2 = SS1 ^ ROTL32(A, 12);
        uint32_t Wp  = W[j] ^ W[j + 4];
        uint32_t TT1, TT2;
        if (j < 16) {
            TT1 = (A ^ B ^ C) + D + SS2 + Wp;
            TT2 = (E ^ F ^ G) + H + SS1 + W[j];
        } else {
            TT1 = ((A & B) | (A & C) | (B & C)) + D + SS2 + Wp;
            TT2 = ((E & F) | (~E & G)) + H + SS1 + W[j];
        }
        D = C; C = ROTL32(B, 9);  B = A; A = TT1;
        H = G; G = ROTL32(F, 19); F = E; E = P0(TT2);
    }

    state[0] ^= A; state[1] ^= B; state[2] ^= C; state[3] ^= D;
    state[4] ^= E; state[5] ^= F; state[6] ^= G; state[7] ^= H;
}

#endif /* __aarch64__ */
