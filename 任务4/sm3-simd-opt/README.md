# SM3 密码杂凑算法 —— SIMD 寄存器 + 通用寄存器混合优化实现

基于 GB/T 32905-2016 标准，实现 SM3 的**混合优化**方案，覆盖
**ARM64（NEON）** 与 **x86-64（SSSE3 / AVX2 / AVX512F+BW）** 两类指令集架构。

## 优化设计（混合算法）

| 层次 | 名称 | 通用寄存器承担 | SIMD 寄存器承担 |
|------|------|----------------|------------------|
| L1 | 单流块内混合 | 64 轮迭代压缩（A–H 串行状态链）、消息扩展近距依赖链头 W[base] | 消息扩展批量计算（P1 置换、ROTL15/ROTL7、异或，4 字/次），大端字节序转换 |
| L2 | 多块并行 multi-buffer | 常数生成、循环控制 | 4 路(NEON)/8 路(AVX2)/16 路(AVX512) 消息各占用一个 32bit lane，扩展+压缩全程 SIMD |

设计依据：SM3 压缩函数 A→B→…→H 的链式依赖在单条消息内本质串行，强行向量化
收益甚微；而消息扩展中 W[j] 对 W[j-3] 的近距依赖可通过“标量解链头 + SIMD 算四个”
的方式打破。多条独立消息之间则完全无依赖，可用宽 SIMD 寄存器做块间数据级并行。

## 目录结构

```
sm3-simd-opt/
├── Makefile                  # 按编译目标三元组自动选择平台源文件
├── README.md
├── include/sm3.h             # 公共 API（流式/一次性/多块并行）
├── src/
│   ├── sm3_scalar.c          # 标量参考实现（正确性基准与性能基线）
│   ├── sm3_dispatch.c        # 运行时分派（cpuid/架构）、流式 API、安全回退
│   ├── sm3_hybrid_x86.c      # x86 单流混合（SSSE3 128bit）
│   ├── sm3_mb8_avx2.c        # x86 AVX2 多块并行 8 路
│   ├── sm3_mb16_avx512.c     # x86 AVX-512 多块并行 16 路
│   ├── sm3_hybrid_arm64.c    # ARM64 NEON 单流混合（128bit）
│   └── sm3_mb4_arm64.c       # ARM64 NEON 多块并行 4 路
├── test/test_sm3.c           # 标准测试向量 + 跨实现一致性 + 流式分段
└── bench/bench_sm3.c         # 吞吐基准（MiB/s 与加速比）
```

## 构建与运行

```bash
# x86-64 本机
make            # 生成 sm3test / sm3bench
make test       # 正确性验证（GB/T 32905-2016 标准向量）
make bench      # 性能基准

# ARM64 交叉编译（以 Arm GNU Toolchain 为例）
make CC=aarch64-none-linux-gnu-gcc
# 在 ARM64 设备上直接运行；或用 qemu-user 模拟：
qemu-aarch64 -L <sysroot> ./sm3test
```

x86 二进制内置运行时 cpuid 分派：无 AVX2/AVX-512 的机器自动回退，
任何 x86-64 机器上均可正确运行；ARM64 的 NEON 为架构标配，直接启用。

## API 速览

```c
uint8_t out[32];
sm3_hash(data, len, out);                 // 一次性（自动最优实现）
sm3_init(&ctx); sm3_update(&ctx, d, n); sm3_final(&ctx, out);  // 流式
sm3_hash_mb8 (msgs, len, outs);           // 8 条等长消息并行（AVX2）
sm3_hash_mb16(msgs, len, outs);           // 16 条（AVX-512）
sm3_hash_mb4 (msgs, len, outs);           // 4 条（ARM64 NEON）
```

## 正确性

- GB/T 32905-2016 附录示例：`abc` → `66c7f0f4…8f4ba8e0`；
  `abcd×16`（64 字节）→ `debe9ff9…9c0c5732`
- 1 MiB 随机消息：标量 / 混合 / NEON / AVX2 / AVX512 输出完全一致；
  流式任意粒度分段喂入与一次性结果一致。
