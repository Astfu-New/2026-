# SM3 SIMD + 通用寄存器混合优化

基于 GB/T 32905-2016 的 SM3 杂凑算法优化实现，支持 **x86（SSSE3 / AVX2 / AVX-512）** 与 **ARM64（NEON）**，内置运行时调度，任何机器上均可正确运行。

## 环境要求

- Linux / macOS：`gcc` + `make`（macOS 先运行 `xcode-select --install`）
- Windows：建议在 VSCode 中通过 **WSL** 打开本项目

## 快速开始

```bash
make          # 编译
make test     # 运行测试用例（标准向量 + 一致性验证）
make bench    # 运行性能基准
```

看到 `[PASS] abc`、`[PASS] abcd x 16` 及“全部测试通过 ✓”即成功。

## 目录结构

```
include/sm3.h        # 公共 API
src/sm3_scalar.c     # 标量参考实现
src/sm3_dispatch.c   # 运行时调度与回退
src/sm3_hybrid_x86.c # x86 单流混合（SSSE3）
src/sm3_mb8_avx2.c   # AVX2 多块 8 路
src/sm3_mb16_avx512.c# AVX-512 多块 16 路
src/sm3_hybrid_arm64.c # ARM64 NEON 单流混合
src/sm3_mb4_arm64.c  # NEON 多块 4 路
test/test_sm3.c      # 测试用例
bench/bench_sm3.c    # 性能基准
```

## 常见问题

- CPU 不支持 AVX-512 / AVX2 时会自动回退，测试依然全部通过，属正常现象。
- ARM64 交叉编译：`make CC=aarch64-linux-gnu-gcc`
- VSCode 一键构建：按 `` Ctrl+` `` 打开终端执行上述命令即可。
