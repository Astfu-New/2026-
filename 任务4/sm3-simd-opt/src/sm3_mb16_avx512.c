/*
 * sm3_mb16_avx512.c — x86 AVX-512 多块并行（16 路）SM3 实现
 *
 * 16 条独立消息各占用 __m512i 的一个 32bit lane，消息扩展与压缩全程
 * 以 512bit SIMD 执行；ROTL 直接使用 AVX512F 的 vprold 单指令，
 * 字节序转换使用 AVX512BW 的 vpshufb。与 AVX2 的 8 路版本同构，
 * 体现寄存器宽度从 256bit 到 512bit 的扩展性。
 */
#if defined(__x86_64__) || defined(_M_X64)

#include "sm3.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define VROL(x, n)  _mm512_rol_epi32(x, n)
#define VP0(x)      _mm512_xor_si512(_mm512_xor_si512(x, VROL(x, 9)),  VROL(x, 17))
#define VP1(x)      _mm512_xor_si512(_mm512_xor_si512(x, VROL(x, 15)), VROL(x, 23))
#define VFF0(x,y,z) _mm512_xor_si512(_mm512_xor_si512(x, y), z)
#define VFF1(x,y,z) _mm512_or_si512(_mm512_and_si512(x, y), \
                    _mm512_and_si512(_mm512_or_si512(x, y), z))
#define VGG1(x,y,z) _mm512_xor_si512(z, _mm512_and_si512(x, _mm512_xor_si512(y, z)))

static inline uint32_t rd32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

#define SM3_BSWAP512 _mm512_set_epi8( \
    12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3, \
    12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3, \
    12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3, \
    12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3)

__attribute__((target("avx512f,avx512bw")))
static void sm3_mb16_compress(__m512i st[8], const uint8_t *const blk[16])
{
    __m512i W[68];
    __m512i A, B, C, D, E, F, G, H;
    const __m512i bswap = SM3_BSWAP512;
    int j;

    for (j = 0; j < 16; j++) {
        __m512i v = _mm512_set_epi32(
            rd32(blk[15] + 4*j), rd32(blk[14] + 4*j),
            rd32(blk[13] + 4*j), rd32(blk[12] + 4*j),
            rd32(blk[11] + 4*j), rd32(blk[10] + 4*j),
            rd32(blk[9]  + 4*j), rd32(blk[8]  + 4*j),
            rd32(blk[7]  + 4*j), rd32(blk[6]  + 4*j),
            rd32(blk[5]  + 4*j), rd32(blk[4]  + 4*j),
            rd32(blk[3]  + 4*j), rd32(blk[2]  + 4*j),
            rd32(blk[1]  + 4*j), rd32(blk[0]  + 4*j));
        W[j] = _mm512_shuffle_epi8(v, bswap);
    }

    for (j = 16; j < 68; j++)
        W[j] = _mm512_xor_si512(
                   VP1(_mm512_xor_si512(
                       _mm512_xor_si512(W[j-16], W[j-9]),
                       VROL(W[j-3], 15))),
                   _mm512_xor_si512(VROL(W[j-13], 7), W[j-6]));

    A = st[0]; B = st[1]; C = st[2]; D = st[3];
    E = st[4]; F = st[5]; G = st[6]; H = st[7];

#define ROUND16(FF, GG) do {                                             \
        uint32_t rt;                                                     \
        __m512i SS1, SS2, TT1, TT2;                                      \
        rt  = ROTL32((j < 16) ? 0x79cc4519u : 0x7a879d8au, j & 31);      \
        SS1 = VROL(_mm512_add_epi32(                                     \
                  _mm512_add_epi32(VROL(A, 12), E),                      \
                  _mm512_set1_epi32((int)rt)), 7);                       \
        SS2 = _mm512_xor_si512(SS1, VROL(A, 12));                        \
        TT1 = _mm512_add_epi32(_mm512_add_epi32(                         \
                  _mm512_add_epi32(FF(A, B, C), D), SS2),                \
              _mm512_xor_si512(W[j], W[j+4]));                           \
        TT2 = _mm512_add_epi32(_mm512_add_epi32(                         \
                  _mm512_add_epi32(GG(E, F, G), H), SS1), W[j]);         \
        D = C; C = VROL(B, 9);  B = A; A = TT1;                          \
        H = G; G = VROL(F, 19); F = E; E = VP0(TT2);                     \
    } while (0)

    for (j = 0; j < 16; j++) ROUND16(VFF0, VFF0);
    for (; j < 64; j++)      ROUND16(VFF1, VGG1);
#undef ROUND16

    st[0] = _mm512_xor_si512(st[0], A); st[1] = _mm512_xor_si512(st[1], B);
    st[2] = _mm512_xor_si512(st[2], C); st[3] = _mm512_xor_si512(st[3], D);
    st[4] = _mm512_xor_si512(st[4], E); st[5] = _mm512_xor_si512(st[5], F);
    st[6] = _mm512_xor_si512(st[6], G); st[7] = _mm512_xor_si512(st[7], H);
}

__attribute__((target("avx512f,avx512bw")))
void sm3_hash_mb16_avx512(const uint8_t *const msg[16], size_t len,
                          uint8_t out[16][SM3_DIGEST_SIZE])
{
    static const uint32_t IV[8] = {
        0x7380166fu, 0x4914b2b9u, 0x172442d7u, 0xda8a0600u,
        0xa96f30bcu, 0x163138aau, 0xe38dee4du, 0xb0fb0e4eu
    };
    const __m512i bswap = SM3_BSWAP512;
    const uint64_t bitlen = (uint64_t)len * 8u;
    const size_t nblk   = (len + 1 + 8 + 63) / 64;
    const size_t stride = nblk * 64;
    uint8_t *buf;
    __m512i st[8];
    size_t b;
    int i, w, k;

    buf = (uint8_t *)malloc(stride * 16);
    if (!buf) {
        for (i = 0; i < 16; i++) sm3_hash(msg[i], len, out[i]);
        return;
    }

    for (i = 0; i < 16; i++) {
        uint8_t *p = buf + (size_t)i * stride;
        memcpy(p, msg[i], len);
        p[len] = 0x80;
        memset(p + len + 1, 0, stride - len - 1 - 8);
        for (k = 0; k < 8; k++)
            p[stride - 1 - k] = (uint8_t)(bitlen >> (8 * k));
    }

    for (w = 0; w < 8; w++)
        st[w] = _mm512_set1_epi32((int)IV[w]);

    for (b = 0; b < nblk; b++) {
        const uint8_t *blk[16];
        for (i = 0; i < 16; i++)
            blk[i] = buf + (size_t)i * stride + b * 64;
        sm3_mb16_compress(st, blk);
    }

    for (w = 0; w < 8; w++) {
        uint32_t tmp[16];
        _mm512_storeu_si512((void *)tmp, _mm512_shuffle_epi8(st[w], bswap));
        for (i = 0; i < 16; i++)
            memcpy(out[i] + 4 * w, &tmp[i], 4);
    }

    free(buf);
}

#endif /* __x86_64__ */
