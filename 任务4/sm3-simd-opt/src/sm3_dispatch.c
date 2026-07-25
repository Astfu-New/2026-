/*
 * sm3_dispatch.c — 平台调度层
 *   - 流式 API（sm3_init/update/final）
 *   - 单流混合压缩的运行时分派（x86: cpuid；ARM64: NEON 标配）
 *   - 多块并行接口的分派与安全回退（任何平台均可正确运行）
 */
#include "sm3.h"
#include <string.h>

/* 跨文件共享：一次性哈希骨架（定义于 sm3_scalar.c） */
void sm3_hash_cf(void (*cf)(uint32_t *, const uint8_t *),
                 const uint8_t *data, size_t len, uint8_t out[32]);

static const uint32_t SM3_IV[8] = {
    0x7380166fu, 0x4914b2b9u, 0x172442d7u, 0xda8a0600u,
    0xa96f30bcu, 0x163138aau, 0xe38dee4du, 0xb0fb0e4eu
};

/* ---------------- 平台能力检测与优化实现声明 ---------------- */
#if defined(__x86_64__) || defined(_M_X64)

void sm3_compress_hybrid_ssse3(uint32_t state[8], const uint8_t block[64]);
void sm3_hash_mb8_avx2(const uint8_t *const msg[8], size_t len, uint8_t out[8][32]);
void sm3_hash_mb16_avx512(const uint8_t *const msg[16], size_t len, uint8_t out[16][32]);

static int has_ssse3(void)
{
    static int v = -1;
    if (v < 0) v = __builtin_cpu_supports("ssse3");
    return v;
}
static int has_avx2(void)
{
    static int v = -1;
    if (v < 0) v = __builtin_cpu_supports("avx2");
    return v;
}
static int has_avx512fbw(void)
{
    static int v = -1;
    if (v < 0) v = __builtin_cpu_supports("avx512f") &&
                   __builtin_cpu_supports("avx512bw");
    return v;
}

#elif defined(__aarch64__)

void sm3_compress_hybrid_neon(uint32_t state[8], const uint8_t block[64]);
void sm3_hash_mb4_neon(const uint8_t *const msg[4], size_t len, uint8_t out[4][32]);

#endif

/* ---------------- 单流混合压缩分派 ---------------- */
void sm3_compress_hybrid(uint32_t state[8], const uint8_t block[SM3_BLOCK_SIZE])
{
#if defined(__x86_64__) || defined(_M_X64)
    if (has_ssse3()) {
        sm3_compress_hybrid_ssse3(state, block);
        return;
    }
#elif defined(__aarch64__)
    sm3_compress_hybrid_neon(state, block);
    return;
#endif
    sm3_compress_scalar(state, block);   /* 回退：纯通用寄存器 */
}

const char *sm3_impl_name(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    return has_ssse3() ? "hybrid-SSSE3 (SIMD扩展 + GPR压缩)" : "scalar (GPR)";
#elif defined(__aarch64__)
    return "hybrid-NEON (SIMD扩展 + GPR压缩)";
#else
    return "scalar (GPR)";
#endif
}

/* ---------------- 一次性哈希 ---------------- */
void sm3_hash_hybrid(const uint8_t *data, size_t len, uint8_t out[SM3_DIGEST_SIZE])
{
    sm3_hash_cf(sm3_compress_hybrid, data, len, out);
}

void sm3_hash(const uint8_t *data, size_t len, uint8_t out[SM3_DIGEST_SIZE])
{
    sm3_hash_hybrid(data, len, out);
}

/* ---------------- 流式 API ---------------- */
void sm3_init(sm3_ctx *ctx)
{
    memcpy(ctx->state, SM3_IV, sizeof(ctx->state));
    ctx->total_len = 0;
    ctx->buf_len   = 0;
}

void sm3_update(sm3_ctx *ctx, const uint8_t *data, size_t len)
{
    ctx->total_len += len;

    if (ctx->buf_len) {                          /* 先填满残余缓冲 */
        size_t need = SM3_BLOCK_SIZE - ctx->buf_len;
        if (need > len) need = len;
        memcpy(ctx->buf + ctx->buf_len, data, need);
        ctx->buf_len += need;
        data += need;
        len  -= need;
        if (ctx->buf_len == SM3_BLOCK_SIZE) {
            sm3_compress_hybrid(ctx->state, ctx->buf);
            ctx->buf_len = 0;
        }
    }
    while (len >= SM3_BLOCK_SIZE) {
        sm3_compress_hybrid(ctx->state, data);
        data += SM3_BLOCK_SIZE;
        len  -= SM3_BLOCK_SIZE;
    }
    if (len) {
        memcpy(ctx->buf, data, len);
        ctx->buf_len = len;
    }
}

void sm3_final(sm3_ctx *ctx, uint8_t out[SM3_DIGEST_SIZE])
{
    uint64_t bitlen = ctx->total_len * 8u;
    size_t i;

    ctx->buf[ctx->buf_len++] = 0x80;
    if (ctx->buf_len > 56) {
        memset(ctx->buf + ctx->buf_len, 0, SM3_BLOCK_SIZE - ctx->buf_len);
        sm3_compress_hybrid(ctx->state, ctx->buf);
        ctx->buf_len = 0;
    }
    memset(ctx->buf + ctx->buf_len, 0, 56 - ctx->buf_len);
    for (i = 0; i < 8; i++)
        ctx->buf[56 + i] = (uint8_t)(bitlen >> (56 - 8 * i));
    sm3_compress_hybrid(ctx->state, ctx->buf);

    for (i = 0; i < 8; i++) {
        out[4*i]     = (uint8_t)(ctx->state[i] >> 24);
        out[4*i + 1] = (uint8_t)(ctx->state[i] >> 16);
        out[4*i + 2] = (uint8_t)(ctx->state[i] >>  8);
        out[4*i + 3] = (uint8_t)(ctx->state[i]);
    }
}

/* ---------------- 多块并行分派（含安全回退） ---------------- */
int sm3_mb_max_lanes(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512fbw()) return 16;
    if (has_avx2())      return 8;
    return 0;
#elif defined(__aarch64__)
    return 4;
#else
    return 0;
#endif
}

void sm3_hash_mb4(const uint8_t *const msg[4], size_t len, uint8_t out[4][SM3_DIGEST_SIZE])
{
#if defined(__aarch64__)
    sm3_hash_mb4_neon(msg, len, out);
    return;
#else
    int i;
    for (i = 0; i < 4; i++) sm3_hash(msg[i], len, out[i]);
#endif
}

void sm3_hash_mb8(const uint8_t *const msg[8], size_t len, uint8_t out[8][SM3_DIGEST_SIZE])
{
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx2()) {
        sm3_hash_mb8_avx2(msg, len, out);
        return;
    }
#endif
    {
        int i;
        for (i = 0; i < 8; i++) sm3_hash(msg[i], len, out[i]);
    }
}

void sm3_hash_mb16(const uint8_t *const msg[16], size_t len, uint8_t out[16][SM3_DIGEST_SIZE])
{
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512fbw()) {
        sm3_hash_mb16_avx512(msg, len, out);
        return;
    }
    if (has_avx2()) {               /* 无 AVX-512 时拆成两批 8 路 */
        sm3_hash_mb8_avx2(msg,     len, out);
        sm3_hash_mb8_avx2(msg + 8, len, out + 8);
        return;
    }
#endif
    {
        int i;
        for (i = 0; i < 16; i++) sm3_hash(msg[i], len, out[i]);
    }
}
