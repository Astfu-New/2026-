/*
 * sm3_mb8_avx2.c — x86 AVX2 多块并行（multi-buffer，8 路）SM3 实现
 *
 * 与单流混合实现互补：8 条独立消息各占用 __m256i 的一个 32bit lane，
 * 消息扩展与 64 轮压缩全部以 SIMD 逐 lane 语义执行，块间无数据依赖，
 * 充分释放 256bit 寄存器宽度。这是“混合优化”的第二层：
 *   单流 = SIMD(扩展) + 通用寄存器(压缩链)
 *   多块 = SIMD(整块间并行) ，块内仍沿用混合扩展结构
 */
#if defined(__x86_64__) || defined(_M_X64)

#include "sm3.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define VROL(x, n)  _mm256_or_si256(_mm256_slli_epi32(x, n), _mm256_srli_epi32(x, 32 - (n)))
#define VP0(x)      _mm256_xor_si256(_mm256_xor_si256(x, VROL(x, 9)),  VROL(x, 17))
#define VP1(x)      _mm256_xor_si256(_mm256_xor_si256(x, VROL(x, 15)), VROL(x, 23))
#define VFF0(x,y,z) _mm256_xor_si256(_mm256_xor_si256(x, y), z)
#define VFF1(x,y,z) _mm256_or_si256(_mm256_and_si256(x, y), \
                    _mm256_and_si256(_mm256_or_si256(x, y), z))
#define VGG1(x,y,z) _mm256_xor_si256(z, _mm256_and_si256(x, _mm256_xor_si256(y, z)))

static inline uint32_t rd32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

#define SM3_BSWAP256 _mm256_set_epi8( \
    12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3, \
    12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3)

/* 8 条消息的第 b 块联合压缩：消息扩展 + 64 轮，全 SIMD */
__attribute__((target("avx2")))
static void sm3_mb8_compress(__m256i st[8], const uint8_t *const blk[8])
{
    __m256i W[68];
    __m256i A, B, C, D, E, F, G, H;
    const __m256i bswap = SM3_BSWAP256;
    int j;

    /* 转置装载：lane i <-> 消息 i，并完成大端字节序转换 */
    for (j = 0; j < 16; j++) {
        __m256i v = _mm256_set_epi32(
            rd32(blk[7] + 4*j), rd32(blk[6] + 4*j),
            rd32(blk[5] + 4*j), rd32(blk[4] + 4*j),
            rd32(blk[3] + 4*j), rd32(blk[2] + 4*j),
            rd32(blk[1] + 4*j), rd32(blk[0] + 4*j));
        W[j] = _mm256_shuffle_epi8(v, bswap);
    }

    /* 消息扩展：各 lane 独立，无跨 lane 依赖，纯 SIMD */
    for (j = 16; j < 68; j++)
        W[j] = _mm256_xor_si256(
                   VP1(_mm256_xor_si256(
                       _mm256_xor_si256(W[j-16], W[j-9]),
                       VROL(W[j-3], 15))),
                   _mm256_xor_si256(VROL(W[j-13], 7), W[j-6]));

    A = st[0]; B = st[1]; C = st[2]; D = st[3];
    E = st[4]; F = st[5]; G = st[6]; H = st[7];

#define ROUND8(FF, GG) do {                                              \
        uint32_t rt;                                                     \
        __m256i SS1, SS2, TT1, TT2;                                      \
        rt  = ROTL32((j < 16) ? 0x79cc4519u : 0x7a879d8au, j & 31);      \
        SS1 = VROL(_mm256_add_epi32(                                     \
                  _mm256_add_epi32(VROL(A, 12), E),                      \
                  _mm256_set1_epi32((int)rt)), 7);                       \
        SS2 = _mm256_xor_si256(SS1, VROL(A, 12));                        \
        TT1 = _mm256_add_epi32(_mm256_add_epi32(                         \
                  _mm256_add_epi32(FF(A, B, C), D), SS2),                \
              _mm256_xor_si256(W[j], W[j+4]));                           \
        TT2 = _mm256_add_epi32(_mm256_add_epi32(                         \
                  _mm256_add_epi32(GG(E, F, G), H), SS1), W[j]);         \
        D = C; C = VROL(B, 9);  B = A; A = TT1;                          \
        H = G; G = VROL(F, 19); F = E; E = VP0(TT2);                     \
    } while (0)

    for (j = 0; j < 16; j++) ROUND8(VFF0, VFF0);
    for (; j < 64; j++)      ROUND8(VFF1, VGG1);
#undef ROUND8

    st[0] = _mm256_xor_si256(st[0], A); st[1] = _mm256_xor_si256(st[1], B);
    st[2] = _mm256_xor_si256(st[2], C); st[3] = _mm256_xor_si256(st[3], D);
    st[4] = _mm256_xor_si256(st[4], E); st[5] = _mm256_xor_si256(st[5], F);
    st[6] = _mm256_xor_si256(st[6], G); st[7] = _mm256_xor_si256(st[7], H);
}

__attribute__((target("avx2")))
void sm3_hash_mb8_avx2(const uint8_t *const msg[8], size_t len,
                       uint8_t out[8][SM3_DIGEST_SIZE])
{
    static const uint32_t IV[8] = {
        0x7380166fu, 0x4914b2b9u, 0x172442d7u, 0xda8a0600u,
        0xa96f30bcu, 0x163138aau, 0xe38dee4du, 0xb0fb0e4eu
    };
    const __m256i bswap = SM3_BSWAP256;
    const uint64_t bitlen = (uint64_t)len * 8u;
    const size_t nblk   = (len + 1 + 8 + 63) / 64;   /* 每条消息填充后块数 */
    const size_t stride = nblk * 64;
    uint8_t *buf;
    __m256i st[8];
    size_t b;
    int i, w, k;

    buf = (uint8_t *)malloc(stride * 8);
    if (!buf) {  /* 内存不足时回退为逐条串行 */
        for (i = 0; i < 8; i++) sm3_hash(msg[i], len, out[i]);
        return;
    }

    /* 各消息独立 MD 强化填充 */
    for (i = 0; i < 8; i++) {
        uint8_t *p = buf + (size_t)i * stride;
        memcpy(p, msg[i], len);
        p[len] = 0x80;
        memset(p + len + 1, 0, stride - len - 1 - 8);
        for (k = 0; k < 8; k++)
            p[stride - 1 - k] = (uint8_t)(bitlen >> (8 * k));
    }

    for (w = 0; w < 8; w++)
        st[w] = _mm256_set1_epi32((int)IV[w]);

    for (b = 0; b < nblk; b++) {
        const uint8_t *blk[8];
        for (i = 0; i < 8; i++)
            blk[i] = buf + (size_t)i * stride + b * 64;
        sm3_mb8_compress(st, blk);
    }

    /* 逆转置输出摘要（大端字节序） */
    for (w = 0; w < 8; w++) {
        uint32_t tmp[8];
        _mm256_storeu_si256((__m256i *)tmp, _mm256_shuffle_epi8(st[w], bswap));
        for (i = 0; i < 8; i++)
            memcpy(out[i] + 4 * w, &tmp[i], 4);
    }

    free(buf);
}

#endif /* __x86_64__ */
