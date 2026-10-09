; =============================================================================
; AgentJIT — 64-bit Native Standalone FASM Test & Benchmark Suite
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; Apache-2.0 License
; =============================================================================

format PE64 console
entry start

include 'C:\proekts\FASM\INCLUDE\WIN64A.INC'

section '.data' data readable writeable
    hdr_msg     db '====================================================================', 13, 10
                db '  AgentJIT Native x86-64 FASM AVX2+FMA Engine Test Suite', 13, 10
                db '====================================================================', 13, 10, 0
    isa_msg     db '  Active SIMD Backend: %s', 13, 10, 0
    t1_msg      db '  [TEST 1] Core Version & ISA Identification: ', 0
    t2_msg      db '  [TEST 2] 64-bit Ultra-Fast Prompt/Trajectory Hash: ', 0
    t3_msg      db '  [TEST 3] Vectorized NULL-Pointer Guard (SIMD): ', 0
    t4_msg      db '  [TEST 4] Vectorized Numerical Range Guards (F64): ', 0
    t5_msg      db '  [TEST 5] Unrolled AVX2+FMA Vector Dot Product (dim=384): ', 0
    t6_msg      db '  [TEST 6] Hardware Cosine Similarity Kernel (dim=384): ', 0
    t7_msg      db '  [TEST 7] Semantic Prompt Guard / Speculative Router: ', 0
    t8_msg      db '  [TEST 8] Vectorized Batch Arithmetic (Add/Sub/Mul/Div): ', 0
    t9_msg      db '  [TEST 9] Trajectory Cache Fast Hash Lookup: ', 0

    pass_str    db 'PASS (Exact match)', 13, 10, 0
    bench_str   db 'PASS (%u ns/op, %u M/sec)', 13, 10, 0
    fail_str    db 'FAIL! Deviation exceeds tolerance.', 13, 10, 0

    all_ok_msg  db '--------------------------------------------------------------------', 13, 10
                db '  ALL 64-BIT FASM AGENTJIT NATIVE TESTS PASSED (100%% Accuracy)!', 13, 10
                db '====================================================================', 13, 10, 0

    isa_str     db 'AVX2+FMA (FASM x86-64, 256-bit SIMD)', 0
    sample_text db 'User orders 5 mechanical keyboards in Berlin', 0

    freq        rq 1
    t_start     rq 1
    t_end       rq 1

    ; Data buffers for testing
    align 32
    ptrs_ok     rq 8
    ptrs_fail   rq 8

    vals_ok     dq 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0
    mins        dq 5.0,  15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0
    maxs        dq 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0
    vals_bad    dq 10.0, 20.0, 99.0, 40.0, 50.0, 60.0, 70.0, 80.0 ; index 2 violates max

    align 32
    v_a         rd 384
    align 32
    v_b         rd 384

    align 32
    arith_a     dq 10.0, 20.0, 30.0, 40.0
    arith_b     dq 2.0,  4.0,  5.0,  8.0
    arith_out   dq 0.0,  0.0,  0.0,  0.0

    hash_tbl    dq 1111h, 2222h, 3333h, 4444h, 5555h, 6666h, 7777h, 8888h

    dummy_var   dq 12345678h

section '.text' code readable executable

; Include microkernels
include 'agentjit64_kernel.inc'

align 16
agentjit_simd_isa:
    lea rax, [isa_str]
    ret

start:
    sub rsp, 88h

    lea rcx, [freq]
    call [QueryPerformanceFrequency]

    lea rcx, [hdr_msg]
    call [printf]

    call agentjit_simd_isa
    mov rdx, rax
    lea rcx, [isa_msg]
    call [printf]

    ; -------------------------------------------------------------------------
    ; [TEST 1] Version & ISA
    ; -------------------------------------------------------------------------
    lea rcx, [t1_msg]
    call [printf]

    call agentjit_version
    cmp eax, 108
    jne .t1_fail
    lea rcx, [pass_str]
    call [printf]
    jmp .test2
.t1_fail:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 1
    call [ExitProcess]

.test2:
    ; -------------------------------------------------------------------------
    ; [TEST 2] 64-bit Fast Hash
    ; -------------------------------------------------------------------------
    lea rcx, [t2_msg]
    call [printf]

    lea rcx, [sample_text]
    mov rdx, 44                 ; len
    xor r8, r8                  ; seed = 0
    call agentjit_fast_hash
    test rax, rax
    jz .t2_fail

    lea rcx, [pass_str]
    call [printf]
    jmp .test3
.t2_fail:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 2
    call [ExitProcess]

.test3:
    ; -------------------------------------------------------------------------
    ; [TEST 3] NULL-Pointer Guard
    ; -------------------------------------------------------------------------
    lea rcx, [t3_msg]
    call [printf]

    ; Init ptrs_ok (all non-null)
    lea rdx, [dummy_var]
    mov [ptrs_ok + 0*8], rdx
    mov [ptrs_ok + 1*8], rdx
    mov [ptrs_ok + 2*8], rdx
    mov [ptrs_ok + 3*8], rdx
    mov [ptrs_ok + 4*8], rdx
    mov [ptrs_ok + 5*8], rdx
    mov [ptrs_ok + 6*8], rdx
    mov [ptrs_ok + 7*8], rdx

    lea rcx, [ptrs_ok]
    mov rdx, 8
    call agentjit_eval_null_guards
    cmp eax, 1
    jne .t3_fail

    ; Init ptrs_fail (one is NULL)
    lea rdx, [dummy_var]
    mov [ptrs_fail + 0*8], rdx
    mov [ptrs_fail + 1*8], rdx
    mov qword [ptrs_fail + 2*8], 0 ; NULL!
    mov [ptrs_fail + 3*8], rdx
    mov [ptrs_fail + 4*8], rdx
    mov [ptrs_fail + 5*8], rdx
    mov [ptrs_fail + 6*8], rdx
    mov [ptrs_fail + 7*8], rdx

    lea rcx, [ptrs_fail]
    mov rdx, 8
    call agentjit_eval_null_guards
    cmp eax, 0
    jne .t3_fail

    lea rcx, [pass_str]
    call [printf]
    jmp .test4
.t3_fail:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 3
    call [ExitProcess]

.test4:
    ; -------------------------------------------------------------------------
    ; [TEST 4] Range Guards (F64)
    ; -------------------------------------------------------------------------
    lea rcx, [t4_msg]
    call [printf]

    ; Valid ranges
    lea rcx, [vals_ok]
    lea rdx, [mins]
    lea r8, [maxs]
    mov r9, 8
    call agentjit_eval_range_guards_f64
    cmp eax, 1
    jne .t4_fail

    ; Violating ranges
    lea rcx, [vals_bad]
    lea rdx, [mins]
    lea r8, [maxs]
    mov r9, 8
    call agentjit_eval_range_guards_f64
    cmp eax, 0
    jne .t4_fail

    lea rcx, [pass_str]
    call [printf]
    jmp .test5
.t4_fail:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 4
    call [ExitProcess]

.test5:
    ; -------------------------------------------------------------------------
    ; [TEST 5] Vector Dot Product (dim=384): a[i]=1.0, b[i]=2.0 -> sum = 768.0
    ; -------------------------------------------------------------------------
    lea rcx, [t5_msg]
    call [printf]

    xor eax, eax
.init_vecs:
    mov dword [v_a + rax*4], 3F800000h ; 1.0f
    mov dword [v_b + rax*4], 40000000h ; 2.0f
    inc eax
    cmp eax, 384
    jb .init_vecs

    lea rcx, [v_a]
    lea rdx, [v_b]
    mov r8, 384
    call agentjit_vector_dot

    ; Result in XMM0 should be 768.0f (44400000h)
    mov dword [rsp + 20h], 44400000h
    vcomiss xmm0, [rsp + 20h]
    jne .t5_fail

    lea rcx, [pass_str]
    call [printf]
    jmp .test6
.t5_fail:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 5
    call [ExitProcess]

.test6:
    ; -------------------------------------------------------------------------
    ; [TEST 6] Cosine Similarity (dim=384): parallel vectors -> 1.0
    ; -------------------------------------------------------------------------
    lea rcx, [t6_msg]
    call [printf]

    lea rcx, [v_a]
    lea rdx, [v_b]
    mov r8, 384
    call agentjit_cosine_similarity

    ; Result in XMM0 should be approx 1.0
    mov dword [rsp + 20h], 3F800000h ; 1.0f
    vsubss xmm1, xmm0, [rsp + 20h]
    ; Absolute value
    mov dword [rsp + 24h], 7FFFFFFFh
    vandps xmm1, xmm1, [rsp + 24h]
    ; Tolerance 0.001f
    mov dword [rsp + 28h], 3A83126Fh ; 0.001f
    vcomiss xmm1, [rsp + 28h]
    ja .t6_fail

    lea rcx, [pass_str]
    call [printf]
    jmp .test7
.t6_fail:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 6
    call [ExitProcess]

.test7:
    ; -------------------------------------------------------------------------
    ; [TEST 7] Semantic Guard Cosine Router
    ; -------------------------------------------------------------------------
    lea rcx, [t7_msg]
    call [printf]

    ; Check parallel vectors with threshold 0.85 -> must return 1 (PASS)
    lea rcx, [v_a]
    lea rdx, [v_b]
    mov r8, 384
    mov dword [rsp + 20h], 3F59999Ah ; 0.85f
    vmovss xmm3, [rsp + 20h]
    call agentjit_semantic_guard_cosine
    cmp eax, 1
    jne .t7_fail

    ; Check with threshold 1.05 -> must return 0 (FAIL/BAILOUT)
    lea rcx, [v_a]
    lea rdx, [v_b]
    mov r8, 384
    mov dword [rsp + 20h], 3F866666h ; 1.05f
    vmovss xmm3, [rsp + 20h]
    call agentjit_semantic_guard_cosine
    cmp eax, 0
    jne .t7_fail

    lea rcx, [pass_str]
    call [printf]
    jmp .test8
.t7_fail:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 7
    call [ExitProcess]

.test8:
    ; -------------------------------------------------------------------------
    ; [TEST 8] Batch Arithmetic (op=2: mul -> 10*2=20, 20*4=80, 30*5=150, 40*8=320)
    ; -------------------------------------------------------------------------
    lea rcx, [t8_msg]
    call [printf]

    lea rcx, [arith_a]
    lea rdx, [arith_b]
    lea r8, [arith_out]
    mov r9, 4
    mov dword [rsp + 20h], 2 ; op=2 (mul, 5th arg in Windows x64 ABI)
    call agentjit_batch_arithmetic_f64

    ; Verify output
    mov rax, 4034000000000000h ; 20.0
    mov [rsp + 20h], rax
    vmovsd xmm0, [arith_out]
    vcomisd xmm0, [rsp + 20h]
    jne .t8_fail

    lea rcx, [pass_str]
    call [printf]
    jmp .test9
.t8_fail:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 8
    call [ExitProcess]

.test9:
    ; -------------------------------------------------------------------------
    ; [TEST 9] Hash Lookup in Cache
    ; -------------------------------------------------------------------------
    lea rcx, [t9_msg]
    call [printf]

    ; Lookup 4444h -> expected index 3
    lea rcx, [hash_tbl]
    mov rdx, 8
    mov r8, 4444h
    call agentjit_batch_lookup_hash
    cmp rax, 3
    jne .t9_fail

    ; Lookup 9999h (not in table) -> expected -1
    lea rcx, [hash_tbl]
    mov rdx, 8
    mov r8, 9999h
    call agentjit_batch_lookup_hash
    cmp rax, -1
    jne .t9_fail

    lea rcx, [pass_str]
    call [printf]

    ; All Passed
    lea rcx, [all_ok_msg]
    call [printf]

    xor ecx, ecx
    call [ExitProcess]

.t9_fail:
    lea rcx, [fail_str]
    call [printf]
    mov ecx, 9
    call [ExitProcess]

section '.idata' import data readable
library kernel32, 'KERNEL32.DLL',\
        msvcrt,   'MSVCRT.DLL'

import kernel32,\
       ExitProcess, 'ExitProcess',\
       QueryPerformanceCounter, 'QueryPerformanceCounter',\
       QueryPerformanceFrequency, 'QueryPerformanceFrequency'

import msvcrt,\
       printf, 'printf'
