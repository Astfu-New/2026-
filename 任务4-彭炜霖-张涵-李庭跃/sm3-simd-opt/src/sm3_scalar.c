/*
 * sm3_scalar.c — SM3 标量参考实现（纯通用寄存器）
 * 严格遵循 GB/T 32905-2016，用作正确性基准与性能对比基线。
 */
#include "sm3.h"
#include <string.h>

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

/* 初始向量 IV */
static const uint32_t SM3_IV[8] = {
    0x7380166fu, 0x4914b2b9u, 0x172442d7u, 0xda8a0600u,
    0xa96f30bcu, 0x163138aau, 0xe38dee4du, 0xb0fb0e4eu
};

/* 置换函数 */
#define P0(x) ((x) ^ ROTL32(x, 9) ^ ROTL32(x, 17))
#define P1(x) ((x) ^ ROTL32(x, 15) ^ ROTL32(x, 23))

/* 布尔函数 */
#define FF0(x, y, z) ((x) ^ (y) ^ (z))
#define FF1(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define GG0(x, y, z) ((x) ^ (y) ^ (z))
#define GG1(x, y, z) (((x) & (y)) | (~(x) & (z)))

#define TJ0 0x79cc4519u   /* j = 0..15  */
#define TJ1 0x7a879d8au   /* j = 16..63 */

void sm3_compress_scalar(uint32_t state[8], const uint8_t block[SM3_BLOCK_SIZE])
{
    uint32_t W[68];
    uint32_t A, B, C, D, E, F, G, H;
    int j;

    /* 消息扩展：B -> W[0..67]（大端装入） */
    for (j = 0; j < 16; j++)
        W[j] = ((uint32_t)block[4*j]     << 24) |
               ((uint32_t)block[4*j + 1] << 16) |
               ((uint32_t)block[4*j + 2] <<  8) |
               ((uint32_t)block[4*j + 3]);

    for (j = 16; j < 68; j++)
        W[j] = P1(W[j-16] ^ W[j-9] ^ ROTL32(W[j-3], 15))
             ^ ROTL32(W[j-13], 7) ^ W[j-6];

    A = state[0]; B = state[1]; C = state[2]; D = state[3];
    E = state[4]; F = state[5]; G = state[6]; H = state[7];

    /* 迭代压缩 */
    for (j = 0; j < 64; j++) {
        uint32_t Tj  = (j < 16) ? TJ0 : TJ1;
        uint32_t SS1 = ROTL32(ROTL32(A, 12) + E + ROTL32(Tj, j & 31), 7);
        uint32_t SS2 = SS1 ^ ROTL32(A, 12);
        uint32_t Wp  = W[j] ^ W[j + 4];
        uint32_t TT1, TT2;
        if (j < 16) {
            TT1 = FF0(A, B, C) + D + SS2 + Wp;
            TT2 = GG0(E, F, G) + H + SS1 + W[j];
        } else {
            TT1 = FF1(A, B, C) + D + SS2 + Wp;
            TT2 = GG1(E, F, G) + H + SS1 + W[j];
        }
        D = C; C = ROTL32(B, 9);  B = A; A = TT1;
        H = G; G = ROTL32(F, 19); F = E; E = P0(TT2);
    }

    state[0] ^= A; state[1] ^= B; state[2] ^= C; state[3] ^= D;
    state[4] ^= E; state[5] ^= F; state[6] ^= G; state[7] ^= H;
}

/* 通用一次性哈希骨架：cf 为压缩函数指针，供标量/混合两版复用（跨文件共享） */
void sm3_hash_cf(void (*cf)(uint32_t *, const uint8_t *),
                 const uint8_t *data, size_t len, uint8_t out[32])
{
    uint32_t st[8];
    uint8_t  tail[2 * SM3_BLOCK_SIZE];
    uint64_t bitlen = (uint64_t)len * 8u;
    size_t   t, i;

    memcpy(st, SM3_IV, sizeof(st));

    while (len >= SM3_BLOCK_SIZE) {
        cf(st, data);
        data += SM3_BLOCK_SIZE;
        len  -= SM3_BLOCK_SIZE;
    }

    /* MD 强化填充：1 || 0...0 || 64bit 大端长度 */
    memcpy(tail, data, len);
    t = len;
    tail[t++] = 0x80;
    if (t > 56) {
        memset(tail + t, 0, SM3_BLOCK_SIZE - t);
        cf(st, tail);
        t = 0;
    }
    memset(tail + t, 0, 56 - t);
    for (i = 0; i < 8; i++)
        tail[56 + i] = (uint8_t)(bitlen >> (56 - 8 * i));
    cf(st, tail);

    for (i = 0; i < 8; i++) {
        out[4*i]     = (uint8_t)(st[i] >> 24);
        out[4*i + 1] = (uint8_t)(st[i] >> 16);
        out[4*i + 2] = (uint8_t)(st[i] >>  8);
        out[4*i + 3] = (uint8_t)(st[i]);
    }
}

void sm3_hash_scalar(const uint8_t *data, size_t len, uint8_t out[SM3_DIGEST_SIZE])
{
    sm3_hash_cf(sm3_compress_scalar, data, len, out);
}
