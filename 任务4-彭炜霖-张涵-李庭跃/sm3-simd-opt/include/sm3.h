/*
 * sm3.h — SM3 密码杂凑算法（GB/T 32905-2016）
 * SIMD 寄存器 + 通用寄存器混合优化实现
 * 支持架构：x86-64（SSSE3 / AVX2 / AVX512F+BW）、ARM64（NEON）
 */
#ifndef SM3_H
#define SM3_H

#include <stdint.h>
#include <stddef.h>

#define SM3_DIGEST_SIZE 32
#define SM3_BLOCK_SIZE  64

#ifdef __cplusplus
extern "C" {
#endif

/* 流式上下文（内部按平台自动选用最优单流实现） */
typedef struct {
    uint32_t state[8];
    uint64_t total_len;             /* 已处理消息总字节数 */
    uint8_t  buf[SM3_BLOCK_SIZE];   /* 未满一块的残余 */
    size_t   buf_len;
} sm3_ctx;

void sm3_init(sm3_ctx *ctx);
void sm3_update(sm3_ctx *ctx, const uint8_t *data, size_t len);
void sm3_final(sm3_ctx *ctx, uint8_t out[SM3_DIGEST_SIZE]);

/* 一次性哈希：自动分派到当前平台最优实现（混合优化版） */
void sm3_hash(const uint8_t *data, size_t len, uint8_t out[SM3_DIGEST_SIZE]);

/* 指定实现的版本（用于性能对比 / 正确性验证） */
void sm3_hash_scalar(const uint8_t *data, size_t len, uint8_t out[SM3_DIGEST_SIZE]);
void sm3_hash_hybrid(const uint8_t *data, size_t len, uint8_t out[SM3_DIGEST_SIZE]);

/* 单块压缩函数 */
void sm3_compress_scalar(uint32_t state[8], const uint8_t block[SM3_BLOCK_SIZE]);
void sm3_compress_hybrid(uint32_t state[8], const uint8_t block[SM3_BLOCK_SIZE]);

/* ---- 多块并行（multi-buffer）接口：len 为每条消息长度（等长） ---- */
/* 返回当前平台块间并行最大路数：16(AVX512)/8(AVX2)/4(NEON)/0(无) */
int  sm3_mb_max_lanes(void);
void sm3_hash_mb4 (const uint8_t *const msg[4],  size_t len, uint8_t out[4][SM3_DIGEST_SIZE]);
void sm3_hash_mb8 (const uint8_t *const msg[8],  size_t len, uint8_t out[8][SM3_DIGEST_SIZE]);
void sm3_hash_mb16(const uint8_t *const msg[16], size_t len, uint8_t out[16][SM3_DIGEST_SIZE]);

/* 当前实际选用的实现名称（调试用） */
const char *sm3_impl_name(void);

#ifdef __cplusplus
}
#endif
#endif /* SM3_H */
