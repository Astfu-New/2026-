/*
 * sm3_mb4_arm64.c — ARM64 NEON 多块并行（4 路）SM3 实现
 *
 * 4 条独立消息各占用 uint32x4_t 的一个 32bit lane，消息扩展与 64 轮
 * 压缩全部以 NEON 逐 lane 语义执行；块装载用 vtrn/vzip 系指令完成
 * 4×4 转置。与 x86 AVX2 的 8 路版本同构，构成跨架构的多块并行层。
 */
#if defined(__aarch64__)

#include "sm3.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define VROL(x, n)  vorrq_u32(vshlq_n_u32(x, n), vshrq_n_u32(x, 32 - (n)))
#define VP0(x)      veorq_u32(veorq_u32(x, VROL(x, 9)),  VROL(x, 17))
#define VP1(x)      veorq_u32(veorq_u32(x, VROL(x, 15)), VROL(x, 23))
#define VFF0(x,y,z) veorq_u32(veorq_u32(x, y), z)
#define VFF1(x,y,z) vorrq_u32(vandq_u32(x, y), vandq_u32(vorrq_u32(x, y), z))
#define VGG1(x,y,z) veorq_u32(z, vandq_u32(x, veorq_u32(y, z)))

/* 从 4 条消息的第 b 块装载 4 个 W 字（含大端转换与 4×4 转置）
 * r0..r3: 每条消息 16 字节（4 个 W 字），lane i <-> 消息 i */
static inline void mb4_load4(uint32x4_t W[4], const uint8_t *const blk[4], int j4)
{
    uint32x4_t r0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk[0] + 16 * j4)));
    uint32x4_t r1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk[1] + 16 * j4)));
    uint32x4_t r2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk[2] + 16 * j4)));
    uint32x4_t r3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk[3] + 16 * j4)));
    uint32x4x2_t t0 = vtrnq_u32(r0, r1);   /* [m0w0,m1w0,m0w2,m1w2] ... */
    uint32x4x2_t t1 = vtrnq_u32(r2, r3);
    W[0] = vcombine_u32(vget_low_u32 (t0.val[0]), vget_low_u32 (t1.val[0]));
    W[1] = vcombine_u32(vget_low_u32 (t0.val[1]), vget_low_u32 (t1.val[1]));
    W[2] = vcombine_u32(vget_high_u32(t0.val[0]), vget_high_u32(t1.val[0]));
    W[3] = vcombine_u32(vget_high_u32(t0.val[1]), vget_high_u32(t1.val[1]));
}

static void sm3_mb4_compress(uint32x4_t st[8], const uint8_t *const blk[4])
{
    uint32x4_t W[68];
    uint32x4_t A, B, C, D, E, F, G, H;
    int j;

    for (j = 0; j < 16; j += 4)
        mb4_load4(&W[j], blk, j / 4);

    for (j = 16; j < 68; j++)
        W[j] = veorq_u32(
                   VP1(veorq_u32(
                       veorq_u32(W[j-16], W[j-9]),
                       VROL(W[j-3], 15))),
                   veorq_u32(VROL(W[j-13], 7), W[j-6]));

    A = st[0]; B = st[1]; C = st[2]; D = st[3];
    E = st[4]; F = st[5]; G = st[6]; H = st[7];

#define ROUND4(FF, GG) do {                                              \
        uint32_t rt;                                                     \
        uint32x4_t SS1, SS2, TT1, TT2;                                   \
        rt  = ROTL32((j < 16) ? 0x79cc4519u : 0x7a879d8au, j & 31);      \
        SS1 = VROL(vaddq_u32(vaddq_u32(VROL(A, 12), E),                  \
                   vdupq_n_u32(rt)), 7);                                 \
        SS2 = veorq_u32(SS1, VROL(A, 12));                               \
        TT1 = vaddq_u32(vaddq_u32(vaddq_u32(FF(A, B, C), D), SS2),       \
              veorq_u32(W[j], W[j+4]));                                  \
        TT2 = vaddq_u32(vaddq_u32(vaddq_u32(GG(E, F, G), H), SS1),       \
              W[j]);                                                     \
        D = C; C = VROL(B, 9);  B = A; A = TT1;                          \
        H = G; G = VROL(F, 19); F = E; E = VP0(TT2);                     \
    } while (0)

    for (j = 0; j < 16; j++) ROUND4(VFF0, VFF0);
    for (; j < 64; j++)      ROUND4(VFF1, VGG1);
#undef ROUND4

    st[0] = veorq_u32(st[0], A); st[1] = veorq_u32(st[1], B);
    st[2] = veorq_u32(st[2], C); st[3] = veorq_u32(st[3], D);
    st[4] = veorq_u32(st[4], E); st[5] = veorq_u32(st[5], F);
    st[6] = veorq_u32(st[6], G); st[7] = veorq_u32(st[7], H);
}

void sm3_hash_mb4_neon(const uint8_t *const msg[4], size_t len,
                       uint8_t out[4][SM3_DIGEST_SIZE])
{
    static const uint32_t IV[8] = {
        0x7380166fu, 0x4914b2b9u, 0x172442d7u, 0xda8a0600u,
        0xa96f30bcu, 0x163138aau, 0xe38dee4du, 0xb0fb0e4eu
    };
    const uint64_t bitlen = (uint64_t)len * 8u;
    const size_t nblk   = (len + 1 + 8 + 63) / 64;
    const size_t stride = nblk * 64;
    uint8_t *buf;
    uint32x4_t st[8];
    size_t b;
    int i, w, k;

    buf = (uint8_t *)malloc(stride * 4);
    if (!buf) {
        for (i = 0; i < 4; i++) sm3_hash(msg[i], len, out[i]);
        return;
    }

    for (i = 0; i < 4; i++) {
        uint8_t *p = buf + (size_t)i * stride;
        memcpy(p, msg[i], len);
        p[len] = 0x80;
        memset(p + len + 1, 0, stride - len - 1 - 8);
        for (k = 0; k < 8; k++)
            p[stride - 1 - k] = (uint8_t)(bitlen >> (8 * k));
    }

    for (w = 0; w < 8; w++)
        st[w] = vdupq_n_u32(IV[w]);

    for (b = 0; b < nblk; b++) {
        const uint8_t *blk[4];
        for (i = 0; i < 4; i++)
            blk[i] = buf + (size_t)i * stride + b * 64;
        sm3_mb4_compress(st, blk);
    }

    for (w = 0; w < 8; w++) {
        uint32_t tmp[4];
        /* lane 内字节反转回大端内存序后转置写出 */
        vst1q_u32(tmp, vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(st[w]))));
        for (i = 0; i < 4; i++)
            memcpy(out[i] + 4 * w, &tmp[i], 4);
    }

    free(buf);
}

#endif /* __aarch64__ */
