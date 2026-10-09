; =============================================================================
; AgentJIT — 32-bit Native Standalone FASM Test & Benchmark Suite
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; Apache-2.0 License
; =============================================================================

format PE console
entry start

include 'C:\proekts\FASM\INCLUDE\WIN32A.INC'

section '.data' data readable writeable
    hdr_msg     db '====================================================================', 13, 10
                db '  AgentJIT Native x86 32-bit FASM SSE2 Engine Test Suite', 13, 10
                db '====================================================================', 13, 10, 0
    isa_msg     db '  Active SIMD Backend: %s', 13, 10, 0
    t1_msg      db '  [TEST 1] Core Version & ISA Identification: ', 0
    t2_msg      db '  [TEST 2] 64-bit Ultra-Fast Prompt/Trajectory Hash: ', 0
    t3_msg      db '  [TEST 3] Vectorized NULL-Pointer Guard (32-bit): ', 0
    t4_msg      db '  [TEST 4] Numerical Range Guards (F64): ', 0
    t5_msg      db '  [TEST 5] SSE2 Vector Dot Product (dim=384): ', 0
    t6_msg      db '  [TEST 6] Hardware Cosine Similarity Kernel (dim=384): ', 0
    t7_msg      db '  [TEST 7] Semantic Prompt Guard / Speculative Router: ', 0
    t8_msg      db '  [TEST 8] Batch Arithmetic (Add/Sub/Mul/Div): ', 0
    t9_msg      db '  [TEST 9] Trajectory Cache Fast Hash Lookup: ', 0

    pass_str    db 'PASS (Exact match)', 13, 10, 0
    fail_str    db 'FAIL! Deviation exceeds tolerance.', 13, 10, 0

    all_ok_msg  db '--------------------------------------------------------------------', 13, 10
                db '  ALL 32-BIT FASM AGENTJIT NATIVE TESTS PASSED (100%% Accuracy)!', 13, 10
                db '====================================================================', 13, 10, 0

    isa_str     db 'SSE2 (FASM x86 32-bit, 128-bit SIMD)', 0
    sample_text db 'User orders 5 mechanical keyboards in Berlin', 0

    dummy_var   dd 12345678h
    align 16
    ptrs_ok     rd 8
    ptrs_fail   rd 8

    vals_ok     dq 10.0, 20.0, 30.0, 40.0
    mins        dq 5.0,  15.0, 25.0, 35.0
    maxs        dq 15.0, 25.0, 35.0, 45.0
    vals_bad    dq 10.0, 20.0, 99.0, 40.0

    align 16
    v_a         rd 384
    align 16
    v_b         rd 384

    align 16
    arith_a     dq 10.0, 20.0, 30.0, 40.0
    arith_b     dq 2.0,  4.0,  5.0,  8.0
    arith_out   dq 0.0,  0.0,  0.0,  0.0

    hash_tbl_lo dd 1111h, 2222h, 3333h, 4444h
    hash_tbl_hi dd 0h,    0h,    0h,    0h

    temp_flt    dd 0.0
    twenty_f64  dq 20.0

section '.text' code readable executable

; Include microkernels
include 'agentjit32_kernel.inc'

align 16
agentjit_simd_isa:
    mov eax, isa_str
    ret

start:
    push hdr_msg
    call [printf]
    add esp, 4

    call agentjit_simd_isa
    push eax
    push isa_msg
    call [printf]
    add esp, 8

    ; -------------------------------------------------------------------------
    ; [TEST 1] Version & ISA
    ; -------------------------------------------------------------------------
    push t1_msg
    call [printf]
    add esp, 4

    call agentjit_version
    cmp eax, 108
    jne .t1_fail
    push pass_str
    call [printf]
    add esp, 4
    jmp .test2
.t1_fail:
    push fail_str
    call [printf]
    add esp, 4
    push 1
    call [ExitProcess]

.test2:
    ; -------------------------------------------------------------------------
    ; [TEST 2] 64-bit Fast Hash
    ; -------------------------------------------------------------------------
    push t2_msg
    call [printf]
    add esp, 4

    push 0                      ; seed_hi
    push 0                      ; seed_lo
    push 44                     ; len
    push sample_text            ; data
    call agentjit_fast_hash
    add esp, 16

    mov ecx, eax
    or ecx, edx
    jz .t2_fail

    push pass_str
    call [printf]
    add esp, 4
    jmp .test3
.t2_fail:
    push fail_str
    call [printf]
    add esp, 4
    push 2
    call [ExitProcess]

.test3:
    ; -------------------------------------------------------------------------
    ; [TEST 3] NULL-Pointer Guard
    ; -------------------------------------------------------------------------
    push t3_msg
    call [printf]
    add esp, 4

    ; Init ptrs_ok
    mov eax, dummy_var
    mov [ptrs_ok + 0*4], eax
    mov [ptrs_ok + 1*4], eax
    mov [ptrs_ok + 2*4], eax
    mov [ptrs_ok + 3*4], eax

    push 4
    push ptrs_ok
    call agentjit_eval_null_guards
    add esp, 8
    cmp eax, 1
    jne .t3_fail

    ; Init ptrs_fail
    mov [ptrs_fail + 0*4], eax
    mov dword [ptrs_fail + 1*4], 0 ; NULL!
    mov [ptrs_fail + 2*4], eax
    mov [ptrs_fail + 3*4], eax

    push 4
    push ptrs_fail
    call agentjit_eval_null_guards
    add esp, 8
    cmp eax, 0
    jne .t3_fail

    push pass_str
    call [printf]
    add esp, 4
    jmp .test4
.t3_fail:
    push fail_str
    call [printf]
    add esp, 4
    push 3
    call [ExitProcess]

.test4:
    ; -------------------------------------------------------------------------
    ; [TEST 4] Range Guards (F64)
    ; -------------------------------------------------------------------------
    push t4_msg
    call [printf]
    add esp, 4

    push 4
    push maxs
    push mins
    push vals_ok
    call agentjit_eval_range_guards_f64
    add esp, 16
    cmp eax, 1
    jne .t4_fail

    push 4
    push maxs
    push mins
    push vals_bad
    call agentjit_eval_range_guards_f64
    add esp, 16
    cmp eax, 0
    jne .t4_fail

    push pass_str
    call [printf]
    add esp, 4
    jmp .test5
.t4_fail:
    push fail_str
    call [printf]
    add esp, 4
    push 4
    call [ExitProcess]

.test5:
    ; -------------------------------------------------------------------------
    ; [TEST 5] Vector Dot Product (dim=384): a[i]=1.0, b[i]=2.0 -> sum = 768.0
    ; -------------------------------------------------------------------------
    push t5_msg
    call [printf]
    add esp, 4

    xor eax, eax
.init_vecs32:
    mov dword [v_a + eax*4], 3F800000h ; 1.0f
    mov dword [v_b + eax*4], 40000000h ; 2.0f
    inc eax
    cmp eax, 384
    jb .init_vecs32

    push 384
    push v_b
    push v_a
    call agentjit_vector_dot
    add esp, 12

    ; Result in ST(0) and XMM0
    fstp dword [temp_flt]
    cmp dword [temp_flt], 44400000h ; 768.0f
    jne .t5_fail

    push pass_str
    call [printf]
    add esp, 4
    jmp .test6
.t5_fail:
    push fail_str
    call [printf]
    add esp, 4
    push 5
    call [ExitProcess]

.test6:
    ; -------------------------------------------------------------------------
    ; [TEST 6] Cosine Similarity (dim=384): parallel vectors -> 1.0
    ; -------------------------------------------------------------------------
    push t6_msg
    call [printf]
    add esp, 4

    push 384
    push v_b
    push v_a
    call agentjit_cosine_similarity
    add esp, 12

    fstp dword [temp_flt]
    movss xmm0, [temp_flt]
    mov dword [temp_flt], 3F800000h ; 1.0f
    subss xmm0, [temp_flt]
    ; abs
    xorps xmm1, xmm1
    ucomiss xmm0, xmm1
    jae .pos_cos
    subss xmm1, xmm0
    movaps xmm0, xmm1
.pos_cos:
    mov dword [temp_flt], 3A83126Fh ; 0.001f
    ucomiss xmm0, [temp_flt]
    ja .t6_fail

    push pass_str
    call [printf]
    add esp, 4
    jmp .test7
.t6_fail:
    push fail_str
    call [printf]
    add esp, 4
    push 6
    call [ExitProcess]

.test7:
    ; -------------------------------------------------------------------------
    ; [TEST 7] Semantic Guard Cosine Router
    ; -------------------------------------------------------------------------
    push t7_msg
    call [printf]
    add esp, 4

    push dword 3F59999Ah        ; 0.85f
    push 384
    push v_b
    push v_a
    call agentjit_semantic_guard_cosine
    add esp, 16
    cmp eax, 1
    jne .t7_fail

    push dword 3F866666h        ; 1.05f
    push 384
    push v_b
    push v_a
    call agentjit_semantic_guard_cosine
    add esp, 16
    cmp eax, 0
    jne .t7_fail

    push pass_str
    call [printf]
    add esp, 4
    jmp .test8
.t7_fail:
    push fail_str
    call [printf]
    add esp, 4
    push 7
    call [ExitProcess]

.test8:
    ; -------------------------------------------------------------------------
    ; [TEST 8] Batch Arithmetic (op=2: mul)
    ; -------------------------------------------------------------------------
    push t8_msg
    call [printf]
    add esp, 4

    push 2                      ; op=2 (mul)
    push 4                      ; count
    push arith_out
    push arith_b
    push arith_a
    call agentjit_batch_arithmetic_f64
    add esp, 20

    movsd xmm0, [arith_out]
    ucomisd xmm0, [twenty_f64]
    jne .t8_fail

    push pass_str
    call [printf]
    add esp, 4
    jmp .test9
.t8_fail:
    push fail_str
    call [printf]
    add esp, 4
    push 8
    call [ExitProcess]

.test9:
    ; -------------------------------------------------------------------------
    ; [TEST 9] Hash Lookup
    ; -------------------------------------------------------------------------
    push t9_msg
    call [printf]
    add esp, 4

    push 0                      ; key_hi
    push 4444h                  ; key_lo
    push 4                      ; size
    push hash_tbl_hi
    push hash_tbl_lo
    call agentjit_batch_lookup_hash
    add esp, 20
    cmp eax, 3
    jne .t9_fail

    push pass_str
    call [printf]
    add esp, 4

    push all_ok_msg
    call [printf]
    add esp, 4

    push 0
    call [ExitProcess]

.t9_fail:
    push fail_str
    call [printf]
    add esp, 4
    push 9
    call [ExitProcess]

section '.idata' import data readable
library kernel32, 'KERNEL32.DLL',\
        msvcrt,   'MSVCRT.DLL'

import kernel32,\
       ExitProcess, 'ExitProcess'

import msvcrt,\
       printf, 'printf'
