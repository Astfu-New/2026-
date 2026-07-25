/*
 * test_sm3.c — SM3 正确性验证
 *  1) GB/T 32905-2016 标准测试向量（abc / abcd×16）
 *  2) 随机长消息：标量 vs 混合单流 vs 多块并行 结果一致性
 *  3) 流式分段喂入 vs 一次性哈希 一致性
 */
#include "sm3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void hexprint(const uint8_t *d, size_t n, char *s)
{
    size_t i;
    for (i = 0; i < n; i++) sprintf(s + 2 * i, "%02x", d[i]);
    s[2 * n] = '\0';
}

static void check_vec(const char *name, const uint8_t *msg, size_t len,
                      const char *expect)
{
    uint8_t out[32];
    char got[65];
    sm3_hash(msg, len, out);
    hexprint(out, 32, got);
    if (strcmp(got, expect) == 0) {
        printf("[PASS] %-28s %s\n", name, got);
    } else {
        printf("[FAIL] %-28s\n  got:    %s\n  expect: %s\n", name, got, expect);
        g_fail = 1;
    }
}

/* 确定性伪随机填充 */
static void fill_rand(uint8_t *p, size_t n, uint32_t seed)
{
    uint32_t x = seed ? seed : 1u;
    size_t i;
    for (i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        p[i] = (uint8_t)x;
    }
}

int main(void)
{
    printf("== 当前单流实现: %s ; 多块最大路数: %d ==\n\n",
           sm3_impl_name(), sm3_mb_max_lanes());

    /* ---- 1. 标准测试向量（GB/T 32905-2016 附录示例） ---- */
    check_vec("abc", (const uint8_t *)"abc", 3,
              "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0");

    {
        uint8_t m[64];
        int i;
        for (i = 0; i < 16; i++) memcpy(m + 4 * i, "abcd", 4);
        check_vec("abcd x 16 (64 bytes)", m, 64,
                  "debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732");
    }

    /* ---- 2. 随机 1 MiB：各实现一致性 ---- */
    {
        const size_t LEN = 1u << 20;
        uint8_t *buf = malloc(LEN);
        uint8_t d_scalar[32], d_hybrid[32], d_stream[32];
        char h_scalar[65], h_hybrid[65];
        const uint8_t *msgs[16];
        uint8_t d_mb[16][32];
        int i, ok;
        sm3_ctx ctx;

        fill_rand(buf, LEN, 0x12345678u);

        sm3_hash_scalar(buf, LEN, d_scalar);
        sm3_hash_hybrid(buf, LEN, d_hybrid);
        hexprint(d_scalar, 32, h_scalar);
        hexprint(d_hybrid, 32, h_hybrid);

        if (memcmp(d_scalar, d_hybrid, 32) == 0)
            printf("[PASS] scalar vs hybrid (1 MiB)  %s\n", h_scalar);
        else {
            printf("[FAIL] scalar vs hybrid (1 MiB)\n  scalar: %s\n  hybrid: %s\n",
                   h_scalar, h_hybrid);
            g_fail = 1;
        }

        /* 流式分段（7/64/1000 字节混合粒度） vs 一次性 */
        sm3_init(&ctx);
        {
            size_t off = 0;
            const size_t grains[] = { 7, 64, 1000, 3, 128, 65536 };
            int g = 0;
            while (off < LEN) {
                size_t step = grains[g++ % 6];
                if (step > LEN - off) step = LEN - off;
                sm3_update(&ctx, buf + off, step);
                off += step;
            }
        }
        sm3_final(&ctx, d_stream);
        if (memcmp(d_stream, d_scalar, 32) == 0)
            printf("[PASS] streaming update/final (1 MiB, 混合粒度)\n");
        else { printf("[FAIL] streaming mismatch\n"); g_fail = 1; }

        /* 多块并行：16 条相同消息，各路输出均应等于标量结果 */
        for (i = 0; i < 16; i++) msgs[i] = buf;
        sm3_hash_mb4 (msgs, LEN, d_mb);            /* 用前 4 槽 */
        ok = 1;
        for (i = 0; i < 4; i++) if (memcmp(d_mb[i], d_scalar, 32)) ok = 0;
        printf(ok ? "[PASS] mb4  (4 lanes 一致)\n"
                  : "[FAIL] mb4\n");
        if (!ok) g_fail = 1;

        memset(d_mb, 0, sizeof(d_mb));
        sm3_hash_mb8 (msgs, LEN, d_mb);
        ok = 1;
        for (i = 0; i < 8; i++) if (memcmp(d_mb[i], d_scalar, 32)) ok = 0;
        printf(ok ? "[PASS] mb8  (8 lanes 一致)\n"
                  : "[FAIL] mb8\n");
        if (!ok) g_fail = 1;

        memset(d_mb, 0, sizeof(d_mb));
        sm3_hash_mb16(msgs, LEN, d_mb);
        ok = 1;
        for (i = 0; i < 16; i++) if (memcmp(d_mb[i], d_scalar, 32)) ok = 0;
        printf(ok ? "[PASS] mb16 (16 lanes 一致)\n"
                  : "[FAIL] mb16\n");
        if (!ok) g_fail = 1;

        free(buf);
    }

    printf("\n%s\n", g_fail ? "*** 存在失败用例 ***" : "全部测试通过 ✓");
    return g_fail;
}
