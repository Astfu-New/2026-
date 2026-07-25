/*
 * sm3_hybrid_x86.c — x86 单流“SIMD 寄存器 + 通用寄存器”混合实现
 *
 * 混合策略（单消息块内）：
 *   - 消息扩展：近距递推 W[j]<-W[j-3] 的头一个字由通用寄存器先行求解，
 *     剩余 3 个字连同冗余的 1 个字由 128bit SIMD 寄存器一次性算出；
 *     SIMD 负责 P1 置换、ROTL(.,15)/ROTL(.,7) 与全部异或。
 *   - 迭代压缩：A..H 状态链本质串行，由通用寄存器执行（每轮 2 次 FF/GG、
 *     P0、5 次 ROTL），W[j] 由 SIMD 扩展结果直接取用。
 */
#if defined(__x86_64__) || defined(_M_X64)

#include "sm3.h"
#include <tmmintrin.h>   /* SSSE3: _mm_shuffle_epi8 */
#include <string.h>

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define P0(x) ((x) ^ ROTL32(x, 9)  ^ ROTL32(x, 17))
#define P1(x) ((x) ^ ROTL32(x, 15) ^ ROTL32(x, 23))

/* SIMD 旋转与置换 */
#define VROTL(x, n) _mm_or_si128(_mm_slli_epi32(x, n), _mm_srli_epi32(x, 32 - (n)))
#define VP1(x)      _mm_xor_si128(_mm_xor_si128((x), VROTL(x, 15)), VROTL(x, 23))

__attribute__((target("ssse3")))
void sm3_compress_hybrid_ssse3(uint32_t state[8], const uint8_t block[64])
{
    uint32_t W[68];
    uint32_t A, B, C, D, E, F, G, H;
    int j, base;

    /* 大端字节序转换掩码（SSSE3 pshufb） */
    const __m128i bswap = _mm_set_epi8(
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);

    /* ---------- 阶段一：SIMD 装载并字节序转换 W[0..15] ---------- */
    for (j = 0; j < 16; j += 4) {
        __m128i v = _mm_loadu_si128((const __m128i *)(block + 4 * j));
        v = _mm_shuffle_epi8(v, bswap);
        _mm_storeu_si128((__m128i *)(W + j), v);
    }

    /* ---------- 阶段二：混合消息扩展 W[16..67] ---------- */
    for (base = 16; base < 68; base += 4) {
        /* (1) 通用寄存器先解近距依赖链头：W[base] */
        uint32_t w0 = P1(W[base-16] ^ W[base-9] ^ ROTL32(W[base-3], 15))
                    ^ ROTL32(W[base-13], 7) ^ W[base-6];

        /* (2) SIMD 一次算 4 个 W（lane3 依赖 (1) 解出的 w0） */
        __m128i v1 = _mm_loadu_si128((const __m128i *)(W + base - 16));
        __m128i v2 = _mm_loadu_si128((const __m128i *)(W + base -  9));
        __m128i v3 = _mm_set_epi32(w0, W[base-1], W[base-2], W[base-3]);
        __m128i v4 = _mm_loadu_si128((const __m128i *)(W + base - 13));
        __m128i v5 = _mm_loadu_si128((const __m128i *)(W + base -  6));
        __m128i t  = _mm_xor_si128(_mm_xor_si128(v1, v2), VROTL(v3, 15));
        t = VP1(t);
        t = _mm_xor_si128(t, _mm_xor_si128(VROTL(v4, 7), v5));
        _mm_storeu_si128((__m128i *)(W + base), t);
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

#endif /* __x86_64__ */
