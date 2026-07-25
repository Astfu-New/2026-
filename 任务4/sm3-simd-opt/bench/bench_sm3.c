/*
 * bench_sm3.c — SM3 性能基准
 *  单流吞吐：scalar vs 混合实现（256 MiB）
 *  多块吞吐：mb4 / mb8 / mb16 vs 等数据量单流串行
 * 结果输出为 MiB/s 与相对加速比。
 */
#define _POSIX_C_SOURCE 199309L
#include "sm3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_rand(uint8_t *p, size_t n, uint32_t seed)
{
    uint32_t x = seed ? seed : 1u;
    size_t i;
    for (i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        p[i] = (uint8_t)x;
    }
}

typedef void (*hash_fn)(const uint8_t *, size_t, uint8_t *);

static double bench_single(hash_fn fn, const uint8_t *buf, size_t len, int reps)
{
    double best = 1e30;
    uint8_t out[32];
    int r;
    for (r = 0; r < reps; r++) {
        double t0 = now_sec();
        fn(buf, len, out);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    return best;   /* 秒 */
}

int main(void)
{
    const size_t SINGLE_LEN = 256u << 20;          /* 256 MiB */
    const int    REPS = 3;
    uint8_t *buf = malloc(SINGLE_LEN);
    int lanes;
    double t, t_ref, mib;

    if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }
    fill_rand(buf, SINGLE_LEN, 0xdeadbeefu);

    printf("实现: %s | mb lanes: %d | 预热后取 %d 次最优\n\n",
           sm3_impl_name(), sm3_mb_max_lanes(), REPS);

    /* ---------------- 单流吞吐 ---------------- */
    mib = (double)SINGLE_LEN / (1024.0 * 1024.0);
    printf("---- 单流吞吐（%zu MiB） ----\n", SINGLE_LEN >> 20);

    t_ref = bench_single(sm3_hash_scalar, buf, SINGLE_LEN, REPS);
    printf("%-40s %8.3f s  %8.2f MiB/s  (基线 1.00x)\n",
           "scalar (通用寄存器)", t_ref, mib / t_ref);

    t = bench_single(sm3_hash_hybrid, buf, SINGLE_LEN, REPS);
    printf("%-40s %8.3f s  %8.2f MiB/s  (加速 %.2fx)\n",
           "hybrid (SIMD扩展+GPR压缩)", t, mib / t, t_ref / t);

    /* ---------------- 多块吞吐 ---------------- */
    lanes = sm3_mb_max_lanes();
    if (lanes >= 4) {
        /* 每条消息长度 = SINGLE_LEN / lanes，总数据量恒定 256 MiB */
        size_t mlen = SINGLE_LEN / 16;
        const uint8_t *msgs[16];
        uint8_t outs[16][32];
        double t0, dt, dt_ref, total_mib;
        int i;

        for (i = 0; i < 16; i++) msgs[i] = buf + (size_t)i * mlen;
        total_mib = 16.0 * (double)mlen / (1024.0 * 1024.0);

        printf("\n---- 多块并行吞吐（16 条 x %zu MiB = %.0f MiB 总数据） ----\n",
               mlen >> 20, total_mib);

        /* 基线：16 条逐条单流 */
        t0 = now_sec();
        for (i = 0; i < 16; i++) sm3_hash(msgs[i], mlen, outs[i]);
        dt_ref = now_sec() - t0;
        printf("%-40s %8.3f s  %8.2f MiB/s  (基线 1.00x)\n",
               "16x 单流串行 (hybrid)", dt_ref, total_mib / dt_ref);

        /* 预热 + 计时：mb4 */
        sm3_hash_mb4(msgs, mlen, outs);
        t0 = now_sec();
        for (i = 0; i < 16; i += 4) sm3_hash_mb4(msgs + i, mlen, outs + i);
        dt = now_sec() - t0;
        printf("%-40s %8.3f s  %8.2f MiB/s  (加速 %.2fx)\n",
               "mb4  (NEON 4路)", dt, total_mib / dt, dt_ref / dt);

        /* mb8 */
        sm3_hash_mb8(msgs, mlen, outs);
        t0 = now_sec();
        for (i = 0; i < 16; i += 8) sm3_hash_mb8(msgs + i, mlen, outs + i);
        dt = now_sec() - t0;
        printf("%-40s %8.3f s  %8.2f MiB/s  (加速 %.2fx)\n",
               "mb8  (AVX2 8路)", dt, total_mib / dt, dt_ref / dt);

        /* mb16 */
        sm3_hash_mb16(msgs, mlen, outs);
        t0 = now_sec();
        sm3_hash_mb16(msgs, mlen, outs);
        dt = now_sec() - t0;
        printf("%-40s %8.3f s  %8.2f MiB/s  (加速 %.2fx)\n",
               "mb16 (AVX512 16路)", dt, total_mib / dt, dt_ref / dt);
    }

    free(buf);
    return 0;
}
