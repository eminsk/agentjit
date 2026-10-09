; =============================================================================
; AgentJIT — High-Performance AVX2+FMA Trajectory & Guard Microkernels (x86-64)
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; Apache-2.0 License
; Windows x64 ABI Compliant
; =============================================================================

; -----------------------------------------------------------------------------
; int agentjit_version(void)
; Returns version integer (108 = v0.1.8)
; -----------------------------------------------------------------------------
align 16
agentjit_version:
    mov eax, 108
    ret


; -----------------------------------------------------------------------------
; uint64_t agentjit_fast_hash(const void* data, size_t len, uint64_t seed)
; RCX = data, RDX = len, R8 = seed
; Ultra-fast 64-bit unrolled trajectory & prompt hashing
; Returns 64-bit hash in RAX
; -----------------------------------------------------------------------------
align 16
agentjit_fast_hash:
    mov rax, r8
    test rax, rax
    jnz .h_start
    mov rax, 0CBF29CE484222325h ; 64-bit FNV offset basis

.h_start:
    mov r9, 100000001B3h        ; 64-bit FNV prime
    test rdx, rdx
    jz .h_finish

.h_loop8:
    cmp rdx, 8
    jb .h_tail

    mov r10, [rcx]
    xor rax, r10
    imul rax, r9
    rol rax, 27
    mov r11, 0c6a4a7935bd1e995h
    imul rax, r11

    add rcx, 8
    sub rdx, 8
    jmp .h_loop8

.h_tail:
    test rdx, rdx
    jz .h_finish

.h_byte_loop:
    movzx r10, byte [rcx]
    xor rax, r10
    imul rax, r9
    inc rcx
    dec rdx
    jnz .h_byte_loop

.h_finish:
    ; Murmur3 64-bit final avalanche mixer
    mov r10, rax
    shr r10, 33
    xor rax, r10
    mov r10, 0ff51afd7ed558ccdh
    imul rax, r10
    mov r10, rax
    shr r10, 33
    xor rax, r10
    mov r10, 0c4ceb9fe1a85ec53h
    imul rax, r10
    mov r10, rax
    shr r10, 33
    xor rax, r10
    ret


; -----------------------------------------------------------------------------
; int agentjit_eval_null_guards(const void* const* pointers, size_t count)
; RCX = pointers, RDX = count
; Vectorized NULL-pointer validation across all input variables
; Returns: 1 if all pointers are NON-NULL (pass), 0 if any is NULL (bailout)
; -----------------------------------------------------------------------------
align 16
agentjit_eval_null_guards:
    test rdx, rdx
    jz .null_ok

    xor rax, rax

.null_loop4:
    lea r8, [rax + 4]
    cmp r8, rdx
    ja .null_scalar

    vmovdqu ymm0, [rcx + rax*8]
    vpxor ymm1, ymm1, ymm1
    vpcmpeqq ymm2, ymm0, ymm1
    vptest ymm2, ymm2
    jnz .null_fail

    add rax, 4
    jmp .null_loop4

.null_scalar:
    cmp rax, rdx
    jae .null_ok

    cmp qword [rcx + rax*8], 0
    jz .null_fail
    inc rax
    jmp .null_scalar

.null_fail:
    vzeroupper
    xor eax, eax
    ret

.null_ok:
    vzeroupper
    mov eax, 1
    ret


; -----------------------------------------------------------------------------
; int agentjit_eval_range_guards_f64(const double* vals, const double* mins,
;                                    const double* maxs, size_t count)
; RCX = vals, RDX = mins, R8 = maxs, R9 = count
; SIMD evaluation of mins[i] <= vals[i] <= maxs[i] in parallel
; Returns: 1 if all pass, 0 if any guard violation (bailout)
; -----------------------------------------------------------------------------
align 16
agentjit_eval_range_guards_f64:
    test r9, r9
    jz .range_ok

    xor rax, rax

.range_loop4:
    lea r10, [rax + 4]
    cmp r10, r9
    ja .range_scalar

    vmovupd ymm0, [rcx + rax*8] ; vals
    vmovupd ymm1, [rdx + rax*8] ; mins
    vmovupd ymm2, [r8 + rax*8]  ; maxs

    ; ymm3 = (vals >= mins), predicate 1Dh (GE_OQ)
    vcmppd ymm3, ymm0, ymm1, 1Dh
    ; ymm4 = (vals <= maxs), predicate 12h (LE_OQ)
    vcmppd ymm4, ymm0, ymm2, 12h

    vandpd ymm3, ymm3, ymm4
    vmovmskpd r11d, ymm3
    cmp r11d, 0Fh
    jne .range_fail

    add rax, 4
    jmp .range_loop4

.range_scalar:
    cmp rax, r9
    jae .range_ok

    vmovsd xmm0, [rcx + rax*8]
    vmovsd xmm1, [rdx + rax*8]
    vmovsd xmm2, [r8 + rax*8]

    vcomisd xmm0, xmm1
    jb .range_fail
    vcomisd xmm2, xmm0
    jb .range_fail

    inc rax
    jmp .range_scalar

.range_fail:
    vzeroupper
    xor eax, eax
    ret

.range_ok:
    vzeroupper
    mov eax, 1
    ret


; -----------------------------------------------------------------------------
; float agentjit_vector_dot(const float* a, const float* b, size_t dim)
; RCX = a, RDX = b, R8 = dim
; Returns float in XMM0
; -----------------------------------------------------------------------------
align 16
agentjit_vector_dot:
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3

    test r8, r8
    jz .vdot_done

    xor rax, rax

.vdot_loop32:
    lea r10, [rax + 32]
    cmp r10, r8
    ja .vdot_tail16

    vmovups ymm4, [rcx + rax*4]
    vfmadd231ps ymm0, ymm4, [rdx + rax*4]

    vmovups ymm5, [rcx + rax*4 + 32]
    vfmadd231ps ymm1, ymm5, [rdx + rax*4 + 32]

    vmovups ymm4, [rcx + rax*4 + 64]
    vfmadd231ps ymm2, ymm4, [rdx + rax*4 + 64]

    vmovups ymm5, [rcx + rax*4 + 96]
    vfmadd231ps ymm3, ymm5, [rdx + rax*4 + 96]

    add rax, 32
    jmp .vdot_loop32

.vdot_tail16:
    vaddps ymm0, ymm0, ymm1
    vaddps ymm2, ymm2, ymm3
    vaddps ymm0, ymm0, ymm2

.vdot_loop8:
    lea r10, [rax + 8]
    cmp r10, r8
    ja .vdot_reduce

    vmovups ymm4, [rcx + rax*4]
    vfmadd231ps ymm0, ymm4, [rdx + rax*4]
    add rax, 8
    jmp .vdot_loop8

.vdot_reduce:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 04Eh
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 001h
    vaddss xmm0, xmm0, xmm1

.vdot_scalar:
    cmp rax, r8
    jae .vdot_done

    vmovss xmm4, [rcx + rax*4]
    vmulss xmm4, xmm4, [rdx + rax*4]
    vaddss xmm0, xmm0, xmm4
    inc rax
    jmp .vdot_scalar

.vdot_done:
    vzeroupper
    ret


; -----------------------------------------------------------------------------
; float agentjit_cosine_similarity(const float* a, const float* b, size_t dim)
; RCX = a, RDX = b, R8 = dim
; Returns float cosine similarity in XMM0
; -----------------------------------------------------------------------------
align 16
agentjit_cosine_similarity:
    vxorps ymm0, ymm0, ymm0     ; dot
    vxorps ymm1, ymm1, ymm1     ; norm_a
    vxorps ymm2, ymm2, ymm2     ; norm_b

    test r8, r8
    jz .cos_zero

    xor rax, rax

.cos_loop8:
    lea r10, [rax + 8]
    cmp r10, r8
    ja .cos_reduce

    vmovups ymm3, [rcx + rax*4]
    vmovups ymm4, [rdx + rax*4]

    vfmadd231ps ymm0, ymm3, ymm4
    vfmadd231ps ymm1, ymm3, ymm3
    vfmadd231ps ymm2, ymm4, ymm4

    add rax, 8
    jmp .cos_loop8

.cos_reduce:
    ; Horizontal reduction for ymm0 -> xmm0 (dot)
    vextractf128 xmm3, ymm0, 1
    vaddps xmm0, xmm0, xmm3
    vshufps xmm3, xmm0, xmm0, 04Eh
    vaddps xmm0, xmm0, xmm3
    vshufps xmm3, xmm0, xmm0, 001h
    vaddss xmm0, xmm0, xmm3

    ; Horizontal reduction for ymm1 -> xmm1 (na)
    vextractf128 xmm3, ymm1, 1
    vaddps xmm1, xmm1, xmm3
    vshufps xmm3, xmm1, xmm1, 04Eh
    vaddps xmm1, xmm1, xmm3
    vshufps xmm3, xmm1, xmm1, 001h
    vaddss xmm1, xmm1, xmm3

    ; Horizontal reduction for ymm2 -> xmm2 (nb)
    vextractf128 xmm3, ymm2, 1
    vaddps xmm2, xmm2, xmm3
    vshufps xmm3, xmm2, xmm2, 04Eh
    vaddps xmm2, xmm2, xmm3
    vshufps xmm3, xmm2, xmm2, 001h
    vaddss xmm2, xmm2, xmm3

.cos_scalar:
    cmp rax, r8
    jae .cos_calc

    vmovss xmm3, [rcx + rax*4]
    vmovss xmm4, [rdx + rax*4]
    vfmadd231ss xmm0, xmm3, xmm4
    vfmadd231ss xmm1, xmm3, xmm3
    vfmadd231ss xmm2, xmm4, xmm4
    inc rax
    jmp .cos_scalar

.cos_calc:
    vmulss xmm1, xmm1, xmm2
    vsqrtss xmm1, xmm1, xmm1

    vxorps xmm3, xmm3, xmm3
    vucomiss xmm1, xmm3
    jbe .cos_zero

    vdivss xmm0, xmm0, xmm1
    vzeroupper
    ret

.cos_zero:
    vxorps xmm0, xmm0, xmm0
    vzeroupper
    ret


; -----------------------------------------------------------------------------
; int agentjit_semantic_guard_cosine(const float* prompt_vec,
;                                    const float* traj_vec,
;                                    size_t dim, float threshold)
; RCX = prompt_vec, RDX = traj_vec, R8 = dim, XMM3 = threshold
; Evaluates if cosine_similarity >= threshold
; Returns: 1 if match (execute compiled pipeline), 0 if mismatch (bailout to LLM)
; -----------------------------------------------------------------------------
align 16
agentjit_semantic_guard_cosine:
    vmovaps xmm5, xmm3          ; preserve threshold in XMM5
    call agentjit_cosine_similarity
    vcomiss xmm0, xmm5
    jb .sem_fail
    mov eax, 1
    ret

.sem_fail:
    xor eax, eax
    ret


; -----------------------------------------------------------------------------
; int agentjit_batch_arithmetic_f64(const double* a, const double* b,
;                                   double* out, size_t count, int op)
; RCX = a, RDX = b, R8 = out, R9 = count, [rsp+28h] = op (0=add, 1=sub, 2=mul, 3=div)
; Vectorized arithmetic synthesizer for agent data-flow nodes
; -----------------------------------------------------------------------------
align 16
agentjit_batch_arithmetic_f64:
    test r9, r9
    jz .arith_done

    mov r10d, dword [rsp + 28h] ; op
    xor rax, rax

.arith_loop4:
    lea r11, [rax + 4]
    cmp r11, r9
    ja .arith_scalar

    vmovupd ymm0, [rcx + rax*8]
    vmovupd ymm1, [rdx + rax*8]

    cmp r10d, 0
    je .vop_add
    cmp r10d, 1
    je .vop_sub
    cmp r10d, 2
    je .vop_mul
    cmp r10d, 3
    je .vop_div
    jmp .vop_add

.vop_add:
    vaddpd ymm0, ymm0, ymm1
    jmp .vop_store
.vop_sub:
    vsubpd ymm0, ymm0, ymm1
    jmp .vop_store
.vop_mul:
    vmulpd ymm0, ymm0, ymm1
    jmp .vop_store
.vop_div:
    vdivpd ymm0, ymm0, ymm1

.vop_store:
    vmovupd [r8 + rax*8], ymm0
    add rax, 4
    jmp .arith_loop4

.arith_scalar:
    cmp rax, r9
    jae .arith_done

    vmovsd xmm0, [rcx + rax*8]
    vmovsd xmm1, [rdx + rax*8]

    cmp r10d, 0
    je .sop_add
    cmp r10d, 1
    je .sop_sub
    cmp r10d, 2
    je .sop_mul
    cmp r10d, 3
    je .sop_div
    jmp .sop_add

.sop_add:
    vaddsd xmm0, xmm0, xmm1
    jmp .sop_store
.sop_sub:
    vsubsd xmm0, xmm0, xmm1
    jmp .sop_store
.sop_mul:
    vmulsd xmm0, xmm0, xmm1
    jmp .sop_store
.sop_div:
    vdivsd xmm0, xmm0, xmm1

.sop_store:
    vmovsd [r8 + rax*8], xmm0
    inc rax
    jmp .arith_scalar

.arith_done:
    vzeroupper
    xor eax, eax
    ret


; -----------------------------------------------------------------------------
; int64_t agentjit_batch_lookup_hash(const uint64_t* table, size_t size, uint64_t key)
; RCX = table, RDX = size, R8 = key
; Fast lookup of trajectory 64-bit hash in cache table
; Returns: Index in table (0..size-1) or -1 if not found
; -----------------------------------------------------------------------------
align 16
agentjit_batch_lookup_hash:
    test rdx, rdx
    jz .lookup_not_found

    ; Broadcast key to ymm1
    vmovq xmm1, r8
    vpbroadcastq ymm1, xmm1

    xor rax, rax

.lookup_loop4:
    lea r10, [rax + 4]
    cmp r10, rdx
    ja .lookup_scalar

    vmovdqu ymm0, [rcx + rax*8]
    vpcmpeqq ymm2, ymm0, ymm1
    vptest ymm2, ymm2
    jz .lookup_next4

    ; Found in this 4-element block
    vmovmskpd r11d, ymm2
    test r11d, 1
    jnz .found_0
    test r11d, 2
    jnz .found_1
    test r11d, 4
    jnz .found_2
    test r11d, 8
    jnz .found_3

.found_0:
    vzeroupper
    ret
.found_1:
    lea rax, [rax + 1]
    vzeroupper
    ret
.found_2:
    lea rax, [rax + 2]
    vzeroupper
    ret
.found_3:
    lea rax, [rax + 3]
    vzeroupper
    ret

.lookup_next4:
    add rax, 4
    jmp .lookup_loop4

.lookup_scalar:
    cmp rax, rdx
    jae .lookup_not_found

    cmp [rcx + rax*8], r8
    je .found_scalar
    inc rax
    jmp .lookup_scalar

.found_scalar:
    vzeroupper
    ret

.lookup_not_found:
    vzeroupper
    mov rax, -1
    ret
