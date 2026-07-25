# SM3 SIMD+GPR 混合优化工程 Makefile
# 本机构建：      make            （自动按编译目标三元组选择源文件）
# 运行测试：      make test
# 运行基准：      make bench
# 交叉编译ARM64： make CC=aarch64-linux-gnu-gcc
CC      ?= gcc
CFLAGS  ?= -O3 -Wall -Wextra -std=c11
INCLUDE := -Iinclude

# 依据编译器目标三元组（而非主机 uname）选择平台源文件，交叉编译同样适用
TARGET  := $(shell $(CC) -dumpmachine 2>/dev/null)

SRCS := src/sm3_scalar.c src/sm3_dispatch.c
ifneq (,$(findstring aarch64,$(TARGET)))
  SRCS += src/sm3_hybrid_arm64.c src/sm3_mb4_arm64.c
else ifneq (,$(findstring x86_64,$(TARGET)))
  SRCS += src/sm3_hybrid_x86.c src/sm3_mb8_avx2.c src/sm3_mb16_avx512.c
else ifneq (,$(findstring x86-64,$(TARGET)))
  SRCS += src/sm3_hybrid_x86.c src/sm3_mb8_avx2.c src/sm3_mb16_avx512.c
endif

OUTDIR  ?= .

all: $(OUTDIR)/sm3test $(OUTDIR)/sm3bench

$(OUTDIR)/sm3test: $(SRCS) test/test_sm3.c include/sm3.h
	$(CC) $(CFLAGS) $(INCLUDE) $(SRCS) test/test_sm3.c -o $@

$(OUTDIR)/sm3bench: $(SRCS) bench/bench_sm3.c include/sm3.h
	$(CC) $(CFLAGS) $(INCLUDE) $(SRCS) bench/bench_sm3.c -o $@

test: $(OUTDIR)/sm3test
	$(OUTDIR)/sm3test

bench: $(OUTDIR)/sm3bench
	$(OUTDIR)/sm3bench

clean:
	rm -f $(OUTDIR)/sm3test $(OUTDIR)/sm3bench sm3test sm3bench

.PHONY: all test bench clean
