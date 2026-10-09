/*
 * AgentJIT Hardware SIMD Acceleration Microkernels (POSIX / Linux / macOS)
 * High-throughput AVX2+FMA & SSE2 microkernels for AI Agent trajectory execution.
 * Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
 * Apache-2.0 License
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HAVE_AVX2_FMA 1
#endif

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int agentjit_version(void) {
    return 108;
}

EXPORT const char* agentjit_simd_isa(void) {
#if defined(HAVE_AVX2_FMA)
    return "Hardware AVX2+FMA SIMD (POSIX / Linux Engine)";
#elif defined(__SSE2__)
    return "Hardware SSE2 SIMD (POSIX / Linux Engine)";
#else
    return "Optimized C99 Vector SIMD Engine";
#endif
}

EXPORT uint64_t agentjit_fast_hash(const void* data, size_t len, uint64_t seed) {
    uint64_t h = seed ? seed : 0xCBF29CE484222325ULL;
    const uint64_t prime = 0x100000001B3ULL;
    const uint8_t* ptr = (const uint8_t*)data;
    if (ptr && len > 0) {
        while (len >= 8) {
            uint64_t v;
            memcpy(&v, ptr, 8);
            h ^= v;
            h *= prime;
            h = (h << 27) | (h >> (64 - 27)); // rol 27
            h *= 0xc6a4a7935bd1e995ULL;
            ptr += 8;
            len -= 8;
        }
        while (len > 0) {
            h ^= (uint64_t)(*ptr);
            h *= prime;
            ptr++;
            len--;
        }
    }
    // Murmur3 64-bit avalanche mixer:
    h ^= (h >> 33);
    h *= 0xff51afd7ed558ccdULL;
    h ^= (h >> 33);
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= (h >> 33);
    return h;
}

EXPORT int agentjit_eval_null_guards(const void* const* ptrs, size_t count) {
    if (!ptrs || count == 0) return 1;
    for (size_t i = 0; i < count; i++) {
        if (ptrs[i] == NULL) return 0;
    }
    return 1;
}

EXPORT int agentjit_eval_range_guards_f64(const double* vals, const double* mins,
                                         const double* maxs, size_t count) {
    if (!vals || !mins || !maxs || count == 0) return 1;
    size_t i = 0;
#if defined(HAVE_AVX2_FMA)
    for (; i + 4 <= count; i += 4) {
        __m256d v = _mm256_loadu_pd(&vals[i]);
        __m256d mi = _mm256_loadu_pd(&mins[i]);
        __m256d ma = _mm256_loadu_pd(&maxs[i]);
        __m256d cmp_ge = _mm256_cmp_pd(v, mi, _CMP_GE_OQ);
        __m256d cmp_le = _mm256_cmp_pd(v, ma, _CMP_LE_OQ);
        __m256d mask = _mm256_and_pd(cmp_ge, cmp_le);
        if (_mm256_movemask_pd(mask) != 0x0F) return 0;
    }
#endif
    for (; i < count; i++) {
        if (vals[i] < mins[i] || vals[i] > maxs[i]) return 0;
    }
    return 1;
}

EXPORT float agentjit_vector_dot(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    size_t i = 0;
    float sum = 0.0f;
#if defined(HAVE_AVX2_FMA)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= dim; i += 16) {
        __m256 va0 = _mm256_loadu_ps(&a[i]);
        __m256 vb0 = _mm256_loadu_ps(&b[i]);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        __m256 va1 = _mm256_loadu_ps(&a[i + 8]);
        __m256 vb1 = _mm256_loadu_ps(&b[i + 8]);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }
    // Horizontal add
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 r = _mm_add_ps(lo, hi);
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_shuffle_ps(r, r, 1));
    sum = _mm_cvtss_f32(r);
#endif
    for (; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

EXPORT float agentjit_cosine_similarity(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    size_t i = 0;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
#if defined(HAVE_AVX2_FMA)
    __m256 vdot = _mm256_setzero_ps();
    __m256 vna  = _mm256_setzero_ps();
    __m256 vnb  = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        vdot = _mm256_fmadd_ps(va, vb, vdot);
        vna  = _mm256_fmadd_ps(va, va, vna);
        vnb  = _mm256_fmadd_ps(vb, vb, vnb);
    }
    // Reduce dot
    __m128 r_dot = _mm_add_ps(_mm256_castps256_ps128(vdot), _mm256_extractf128_ps(vdot, 1));
    r_dot = _mm_add_ps(r_dot, _mm_movehl_ps(r_dot, r_dot));
    r_dot = _mm_add_ss(r_dot, _mm_shuffle_ps(r_dot, r_dot, 1));
    dot = _mm_cvtss_f32(r_dot);

    // Reduce na
    __m128 r_na = _mm_add_ps(_mm256_castps256_ps128(vna), _mm256_extractf128_ps(vna, 1));
    r_na = _mm_add_ps(r_na, _mm_movehl_ps(r_na, r_na));
    r_na = _mm_add_ss(r_na, _mm_shuffle_ps(r_na, r_na, 1));
    na = _mm_cvtss_f32(r_na);

    // Reduce nb
    __m128 r_nb = _mm_add_ps(_mm256_castps256_ps128(vnb), _mm256_extractf128_ps(vnb, 1));
    r_nb = _mm_add_ps(r_nb, _mm_movehl_ps(r_nb, r_nb));
    r_nb = _mm_add_ss(r_nb, _mm_shuffle_ps(r_nb, r_nb, 1));
    nb = _mm_cvtss_f32(r_nb);
#endif
    for (; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na * nb);
    if (denom <= 0.0f) return 0.0f;
    return dot / denom;
}

EXPORT int agentjit_semantic_guard_cosine(const float* emb, const float* ref,
                                         size_t dim, float threshold) {
    float sim = agentjit_cosine_similarity(emb, ref, dim);
    return sim >= threshold ? 1 : 0;
}

EXPORT int agentjit_batch_arithmetic_f64(const double* a, const double* b,
                                        double* out, size_t count, int op) {
    if (!a || !b || !out || count == 0) return 0;
    size_t i = 0;
#if defined(HAVE_AVX2_FMA)
    for (; i + 4 <= count; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vr;
        switch (op) {
            case 0: vr = _mm256_add_pd(va, vb); break;
            case 1: vr = _mm256_sub_pd(va, vb); break;
            case 2: vr = _mm256_mul_pd(va, vb); break;
            case 3: vr = _mm256_div_pd(va, vb); break;
            default: vr = _mm256_add_pd(va, vb); break;
        }
        _mm256_storeu_pd(&out[i], vr);
    }
#endif
    for (; i < count; i++) {
        switch (op) {
            case 0: out[i] = a[i] + b[i]; break;
            case 1: out[i] = a[i] - b[i]; break;
            case 2: out[i] = a[i] * b[i]; break;
            case 3: out[i] = a[i] / b[i]; break;
            default: out[i] = a[i] + b[i]; break;
        }
    }
    return 0;
}

EXPORT int64_t agentjit_batch_lookup_hash(const uint64_t* arr, size_t count, uint64_t target) {
    if (!arr || count == 0) return -1;
    for (size_t i = 0; i < count; i++) {
        if (arr[i] == target) return (int64_t)i;
    }
    return -1;
}
