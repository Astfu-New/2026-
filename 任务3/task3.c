#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <wmmintrin.h>  //AES-NI
#include <tmmintrin.h>  //SSSE3 (Shuffle)
#include <x86intrin.h>  //RDTSC

#define BLOCK_SIZE 16
#define TEST_SIZE (1024 * 1024 * 16) //16MB测试数据

static inline __m128i aes_128_key_exp(__m128i key, __m128i keygened) {
    keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3, 3, 3, 3));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, keygened);
}

void aes128_load_key(uint8_t* enc_key, __m128i* key_schedule) {
    key_schedule[0] = _mm_loadu_si128((const __m128i*) enc_key);
    key_schedule[1] = aes_128_key_exp(key_schedule[0], _mm_aeskeygenassist_si128(key_schedule[0], 0x01));
    key_schedule[2] = aes_128_key_exp(key_schedule[1], _mm_aeskeygenassist_si128(key_schedule[1], 0x02));
    key_schedule[3] = aes_128_key_exp(key_schedule[2], _mm_aeskeygenassist_si128(key_schedule[2], 0x04));
    key_schedule[4] = aes_128_key_exp(key_schedule[3], _mm_aeskeygenassist_si128(key_schedule[3], 0x08));
    key_schedule[5] = aes_128_key_exp(key_schedule[4], _mm_aeskeygenassist_si128(key_schedule[4], 0x10));
    key_schedule[6] = aes_128_key_exp(key_schedule[5], _mm_aeskeygenassist_si128(key_schedule[5], 0x20));
    key_schedule[7] = aes_128_key_exp(key_schedule[6], _mm_aeskeygenassist_si128(key_schedule[6], 0x40));
    key_schedule[8] = aes_128_key_exp(key_schedule[7], _mm_aeskeygenassist_si128(key_schedule[7], 0x80));
    key_schedule[9] = aes_128_key_exp(key_schedule[8], _mm_aeskeygenassist_si128(key_schedule[8], 0x1B));
    key_schedule[10] = aes_128_key_exp(key_schedule[9], _mm_aeskeygenassist_si128(key_schedule[9], 0x36));
}

//T-Table优化示意
uint32_t T0[256], T1[256], T2[256], T3[256];
void aes_round_ttable(uint32_t* state, const uint32_t* round_key) {
    uint32_t t0 = state[0], t1 = state[1], t2 = state[2], t3 = state[3];
    state[0] = T0[t0 >> 24] ^ T1[(t1 >> 16) & 0xFF] ^ T2[(t2 >> 8) & 0xFF] ^ T3[t3 & 0xFF] ^ round_key[0];
    state[1] = T0[t1 >> 24] ^ T1[(t2 >> 16) & 0xFF] ^ T2[(t3 >> 8) & 0xFF] ^ T3[t0 & 0xFF] ^ round_key[1];
    state[2] = T0[t2 >> 24] ^ T1[(t3 >> 16) & 0xFF] ^ T2[(t0 >> 8) & 0xFF] ^ T3[t1 & 0xFF] ^ round_key[2];
    state[3] = T0[t3 >> 24] ^ T1[(t0 >> 16) & 0xFF] ^ T2[(t1 >> 8) & 0xFF] ^ T3[t2 & 0xFF] ^ round_key[3];
}

//CTR模式：结合AES-NI与Shuffle并行4块加密
static const __m128i BSWAP_EPUB8 = { 0x0001020304050607, 0x08090A0B0C0D0E0F };

void aes128_ctr_4blocks_ni(__m128i* pt, __m128i* ct, const __m128i* round_keys, __m128i* iv_counter) {
    __m128i bswap_mask = _mm_loadu_si128((__m128i*) & BSWAP_EPUB8);
    __m128i ctr0 = _mm_shuffle_epi8(*iv_counter, bswap_mask);

    __m128i one = _mm_set_epi64x(0, 1);
    __m128i ctr1 = _mm_add_epi64(ctr0, one);
    __m128i ctr2 = _mm_add_epi64(ctr1, one);
    __m128i ctr3 = _mm_add_epi64(ctr2, one);

    __m128i block0 = _mm_shuffle_epi8(ctr0, bswap_mask);
    __m128i block1 = _mm_shuffle_epi8(ctr1, bswap_mask);
    __m128i block2 = _mm_shuffle_epi8(ctr2, bswap_mask);
    __m128i block3 = _mm_shuffle_epi8(ctr3, bswap_mask);

    block0 = _mm_xor_si128(block0, round_keys[0]);
    block1 = _mm_xor_si128(block1, round_keys[0]);
    block2 = _mm_xor_si128(block2, round_keys[0]);
    block3 = _mm_xor_si128(block3, round_keys[0]);

    for (int i = 1; i < 10; ++i) {
        block0 = _mm_aesenc_si128(block0, round_keys[i]);
        block1 = _mm_aesenc_si128(block1, round_keys[i]);
        block2 = _mm_aesenc_si128(block2, round_keys[i]);
        block3 = _mm_aesenc_si128(block3, round_keys[i]);
    }

    block0 = _mm_aesenclast_si128(block0, round_keys[10]);
    block1 = _mm_aesenclast_si128(block1, round_keys[10]);
    block2 = _mm_aesenclast_si128(block2, round_keys[10]);
    block3 = _mm_aesenclast_si128(block3, round_keys[10]);

    _mm_storeu_si128(&ct[0], _mm_xor_si128(block0, _mm_loadu_si128(&pt[0])));
    _mm_storeu_si128(&ct[1], _mm_xor_si128(block1, _mm_loadu_si128(&pt[1])));
    _mm_storeu_si128(&ct[2], _mm_xor_si128(block2, _mm_loadu_si128(&pt[2])));
    _mm_storeu_si128(&ct[3], _mm_xor_si128(block3, _mm_loadu_si128(&pt[3])));

    *iv_counter = _mm_shuffle_epi8(_mm_add_epi64(ctr3, one), bswap_mask);
}

//XTS模式：GF(2^128) Tweak 生成与并行加密

static inline __m128i next_tweak(__m128i t) {
    __m128i res, mask;
    //获取最高位，判断是否溢出需要异或0x87
    mask = _mm_srai_epi32(_mm_shuffle_epi32(t, _MM_SHUFFLE(3, 3, 3, 3)), 31);
    mask = _mm_and_si128(mask, _mm_set_epi32(0, 0, 0, 0x87));

    __m128i t_shifted = _mm_slli_epi64(t, 1);
    __m128i t_carry = _mm_srli_epi64(_mm_slli_si128(t, 8), 63);
    res = _mm_or_si128(t_shifted, t_carry);

    return _mm_xor_si128(res, mask);
}

void aes128_xts_4blocks_ni(__m128i* pt, __m128i* ct, const __m128i* round_keys, __m128i* tweak) {
    __m128i t0 = *tweak;
    __m128i t1 = next_tweak(t0);
    __m128i t2 = next_tweak(t1);
    __m128i t3 = next_tweak(t2);

    __m128i b0 = _mm_xor_si128(_mm_loadu_si128(&pt[0]), t0);
    __m128i b1 = _mm_xor_si128(_mm_loadu_si128(&pt[1]), t1);
    __m128i b2 = _mm_xor_si128(_mm_loadu_si128(&pt[2]), t2);
    __m128i b3 = _mm_xor_si128(_mm_loadu_si128(&pt[3]), t3);

    b0 = _mm_xor_si128(b0, round_keys[0]);
    b1 = _mm_xor_si128(b1, round_keys[0]);
    b2 = _mm_xor_si128(b2, round_keys[0]);
    b3 = _mm_xor_si128(b3, round_keys[0]);

    for (int i = 1; i < 10; ++i) {
        b0 = _mm_aesenc_si128(b0, round_keys[i]);
        b1 = _mm_aesenc_si128(b1, round_keys[i]);
        b2 = _mm_aesenc_si128(b2, round_keys[i]);
        b3 = _mm_aesenc_si128(b3, round_keys[i]);
    }
    b0 = _mm_aesenclast_si128(b0, round_keys[10]);
    b1 = _mm_aesenclast_si128(b1, round_keys[10]);
    b2 = _mm_aesenclast_si128(b2, round_keys[10]);
    b3 = _mm_aesenclast_si128(b3, round_keys[10]);

    _mm_storeu_si128(&ct[0], _mm_xor_si128(b0, t0));
    _mm_storeu_si128(&ct[1], _mm_xor_si128(b1, t1));
    _mm_storeu_si128(&ct[2], _mm_xor_si128(b2, t2));
    _mm_storeu_si128(&ct[3], _mm_xor_si128(b3, t3));

    *tweak = next_tweak(t3);
}

uint8_t data_in[TEST_SIZE] __attribute__((aligned(16)));
uint8_t data_out[TEST_SIZE] __attribute__((aligned(16)));

int main() {
    printf("Starting AES Optimization Benchmark (Test Size: %d MB)...\n", TEST_SIZE / 1024 / 1024);

    //初始化密钥
    uint8_t key[16] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                       0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F };
    __m128i round_keys[11];
    aes128_load_key(key, round_keys);

    __m128i iv = _mm_set_epi32(0, 0, 0, 0);
    __m128i tweak = _mm_set_epi32(0, 0, 0, 1);

    uint64_t start, end;
    double cycles_per_byte;

    //测试CTR并行模式
    start = __rdtsc();
    for (int i = 0; i < TEST_SIZE; i += 64) {
        aes128_ctr_4blocks_ni((__m128i*)(data_in + i), (__m128i*)(data_out + i), round_keys, &iv);
    }
    end = __rdtsc();
    cycles_per_byte = (double)(end - start) / TEST_SIZE;
    printf("AES-128 CTR (4-Block Parallel): %.2f cpb\n", cycles_per_byte);

    //测试XTS并行模式
    start = __rdtsc();
    for (int i = 0; i < TEST_SIZE; i += 64) {
        aes128_xts_4blocks_ni((__m128i*)(data_in + i), (__m128i*)(data_out + i), round_keys, &tweak);
    }
    end = __rdtsc();
    cycles_per_byte = (double)(end - start) / TEST_SIZE;
    printf("AES-128 XTS (4-Block Parallel): %.2f cpb\n", cycles_per_byte);

    system("pause");
    return 0;
}