; =============================================================================
; AgentJIT — 32-bit SSE2 Trajectory & Guard Microkernels (x86 FASM)
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; Apache-2.0 License
; Standard 32-bit cdecl Calling Convention
; =============================================================================

; -----------------------------------------------------------------------------
; int agentjit_version(void)
; Returns version integer (108 = v0.1.8) in EAX
; -----------------------------------------------------------------------------
align 16
agentjit_version:
    mov eax, 108
    ret


; -----------------------------------------------------------------------------
; uint64_t agentjit_fast_hash(const void* data, size_t len, uint32_t seed_lo, uint32_t seed_hi)
; [ebp+8]=data, [ebp+12]=len, [ebp+16]=seed_lo, [ebp+20]=seed_hi
; Returns 64-bit hash in EDX:EAX
; -----------------------------------------------------------------------------
align 16
agentjit_fast_hash:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi

    mov esi, [ebp + 8]          ; data
    mov ecx, [ebp + 12]         ; len
    mov eax, [ebp + 16]         ; h_lo
    mov edx, [ebp + 20]         ; h_hi

    ; If seed == 0, use 64-bit FNV offset basis (0xCBF29CE484222325)
    mov ebx, eax
    or ebx, edx
    jnz .h32_loop
    mov eax, 84222325h
    mov edx, 0CBF29CE4h

.h32_loop:
    test ecx, ecx
    jz .h32_finish

    movzx ebx, byte [esi]
    xor eax, ebx
    inc esi
    dec ecx

    ; 64-bit multiply by FNV prime (0x100000001B3)
    ; (H_hi * 0x1B3) + (H_lo * 0x1B3 + (H_lo << 8))
    imul edx, edx, 1B3h
    mov edi, eax
    imul edi, edi, 1B3h
    add edx, edi
    rol eax, 5
    xor edx, eax

    jmp .h32_loop

.h32_finish:
    ; Avalanche
    xor eax, edx
    rol eax, 13
    xor edx, eax
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret


; -----------------------------------------------------------------------------
; int agentjit_eval_null_guards(const void* const* pointers, size_t count)
; [ebp+8]=pointers, [ebp+12]=count
; Returns: 1 if all pointers are NON-NULL, 0 if any is NULL
; -----------------------------------------------------------------------------
align 16
agentjit_eval_null_guards:
    push ebp
    mov ebp, esp
    push esi

    mov esi, [ebp + 8]          ; pointers
    mov ecx, [ebp + 12]         ; count
    test ecx, ecx
    jz .null32_ok

    xor edx, edx
.null32_loop:
    cmp dword [esi + edx*4], 0
    jz .null32_fail
    inc edx
    cmp edx, ecx
    jb .null32_loop

.null32_ok:
    mov eax, 1
    pop esi
    pop ebp
    ret

.null32_fail:
    xor eax, eax
    pop esi
    pop ebp
    ret


; -----------------------------------------------------------------------------
; int agentjit_eval_range_guards_f64(const double* vals, const double* mins,
;                                    const double* maxs, size_t count)
; [ebp+8]=vals, [ebp+12]=mins, [ebp+16]=maxs, [ebp+20]=count
; Returns: 1 if all satisfy mins[i] <= vals[i] <= maxs[i], 0 otherwise
; -----------------------------------------------------------------------------
align 16
agentjit_eval_range_guards_f64:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi

    mov esi, [ebp + 8]          ; vals
    mov edi, [ebp + 12]         ; mins
    mov ebx, [ebp + 16]         ; maxs
    mov ecx, [ebp + 20]         ; count

    test ecx, ecx
    jz .range32_ok

    xor edx, edx
.range32_loop:
    movsd xmm0, [esi + edx*8]   ; val
    movsd xmm1, [edi + edx*8]   ; min
    movsd xmm2, [ebx + edx*8]   ; max

    ucomisd xmm0, xmm1
    jb .range32_fail
    ucomisd xmm2, xmm0
    jb .range32_fail

    inc edx
    cmp edx, ecx
    jb .range32_loop

.range32_ok:
    mov eax, 1
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret

.range32_fail:
    xor eax, eax
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret


; -----------------------------------------------------------------------------
; float agentjit_vector_dot(const float* a, const float* b, size_t dim)
; [ebp+8]=a, [ebp+12]=b, [ebp+16]=dim
; Returns float in ST(0) and XMM0
; -----------------------------------------------------------------------------
align 16
agentjit_vector_dot:
    push ebp
    mov ebp, esp
    push esi
    push edi
    sub esp, 16

    mov esi, [ebp + 8]
    mov edi, [ebp + 12]
    mov ecx, [ebp + 16]

    xorps xmm0, xmm0
    xorps xmm1, xmm1
    test ecx, ecx
    jz .dot32_done

    xor eax, eax
.dot32_loop8:
    lea edx, [eax + 8]
    cmp edx, ecx
    ja .dot32_tail

    movups xmm2, [esi + eax*4]
    movups xmm3, [edi + eax*4]
    mulps  xmm2, xmm3
    addps  xmm0, xmm2

    movups xmm2, [esi + eax*4 + 16]
    movups xmm3, [edi + eax*4 + 16]
    mulps  xmm2, xmm3
    addps  xmm1, xmm2

    add eax, 8
    jmp .dot32_loop8

.dot32_tail:
    addps xmm0, xmm1

.dot32_loop4:
    lea edx, [eax + 4]
    cmp edx, ecx
    ja .dot32_reduce

    movups xmm2, [esi + eax*4]
    movups xmm3, [edi + eax*4]
    mulps  xmm2, xmm3
    addps  xmm0, xmm2
    add eax, 4
    jmp .dot32_loop4

.dot32_reduce:
    movhlps xmm1, xmm0
    addps   xmm0, xmm1
    movaps  xmm1, xmm0
    shufps  xmm1, xmm1, 001h
    addss   xmm0, xmm1

.dot32_scalar:
    cmp eax, ecx
    jae .dot32_done

    movss xmm2, [esi + eax*4]
    mulss xmm2, [edi + eax*4]
    addss xmm0, xmm2
    inc eax
    jmp .dot32_scalar

.dot32_done:
    movss [esp], xmm0
    fld dword [esp]             ; return in ST(0)
    add esp, 16
    pop edi
    pop esi
    pop ebp
    ret


; -----------------------------------------------------------------------------
; float agentjit_cosine_similarity(const float* a, const float* b, size_t dim)
; [ebp+8]=a, [ebp+12]=b, [ebp+16]=dim
; Returns float in ST(0) and XMM0
; -----------------------------------------------------------------------------
align 16
agentjit_cosine_similarity:
    push ebp
    mov ebp, esp
    push esi
    push edi
    sub esp, 16

    mov esi, [ebp + 8]
    mov edi, [ebp + 12]
    mov ecx, [ebp + 16]

    xorps xmm0, xmm0            ; dot
    xorps xmm1, xmm1            ; na
    xorps xmm2, xmm2            ; nb

    test ecx, ecx
    jz .cos32_zero

    xor eax, eax
.cos32_loop4:
    lea edx, [eax + 4]
    cmp edx, ecx
    ja .cos32_reduce

    movups xmm3, [esi + eax*4]
    movups xmm4, [edi + eax*4]

    movaps xmm5, xmm3
    mulps  xmm5, xmm4
    addps  xmm0, xmm5           ; dot += a*b

    movaps xmm5, xmm3
    mulps  xmm5, xmm3
    addps  xmm1, xmm5           ; na += a*a

    movaps xmm5, xmm4
    mulps  xmm5, xmm4
    addps  xmm2, xmm5           ; nb += b*b

    add eax, 4
    jmp .cos32_loop4

.cos32_reduce:
    ; Reduce xmm0 (dot)
    movhlps xmm3, xmm0
    addps   xmm0, xmm3
    movaps  xmm3, xmm0
    shufps  xmm3, xmm3, 001h
    addss   xmm0, xmm3

    ; Reduce xmm1 (na)
    movhlps xmm3, xmm1
    addps   xmm1, xmm3
    movaps  xmm3, xmm1
    shufps  xmm3, xmm3, 001h
    addss   xmm1, xmm3

    ; Reduce xmm2 (nb)
    movhlps xmm3, xmm2
    addps   xmm2, xmm3
    movaps  xmm3, xmm2
    shufps  xmm3, xmm3, 001h
    addss   xmm2, xmm3

.cos32_scalar:
    cmp eax, ecx
    jae .cos32_calc

    movss xmm3, [esi + eax*4]
    movss xmm4, [edi + eax*4]

    movaps xmm5, xmm3
    mulss  xmm5, xmm4
    addss  xmm0, xmm5

    movaps xmm5, xmm3
    mulss  xmm5, xmm3
    addss  xmm1, xmm5

    movaps xmm5, xmm4
    mulss  xmm5, xmm4
    addss  xmm2, xmm5

    inc eax
    jmp .cos32_scalar

.cos32_calc:
    mulss xmm1, xmm2            ; na * nb
    sqrtss xmm1, xmm1           ; denom

    xorps xmm3, xmm3
    ucomiss xmm1, xmm3
    jbe .cos32_zero

    divss xmm0, xmm1            ; cos = dot / denom
    movss [esp], xmm0
    fld dword [esp]
    add esp, 16
    pop edi
    pop esi
    pop ebp
    ret

.cos32_zero:
    xorps xmm0, xmm0
    movss [esp], xmm0
    fld dword [esp]
    add esp, 16
    pop edi
    pop esi
    pop ebp
    ret


; -----------------------------------------------------------------------------
; int agentjit_semantic_guard_cosine(const float* a, const float* b,
;                                    size_t dim, float threshold)
; [ebp+8]=a, [ebp+12]=b, [ebp+16]=dim, [ebp+20]=threshold
; Returns: 1 if cosine >= threshold, 0 if bailout
; -----------------------------------------------------------------------------
align 16
agentjit_semantic_guard_cosine:
    push ebp
    mov ebp, esp

    ; Call agentjit_cosine_similarity(a, b, dim)
    push dword [ebp + 16]       ; dim
    push dword [ebp + 12]       ; b
    push dword [ebp + 8]        ; a
    call agentjit_cosine_similarity
    add esp, 12

    ; Result in XMM0 and ST(0)
    ; Threshold at [ebp + 20]
    movss xmm1, [ebp + 20]
    ucomiss xmm0, xmm1
    jb .sem32_fail

    fstp st0                    ; pop ST(0)
    mov eax, 1
    pop ebp
    ret

.sem32_fail:
    fstp st0                    ; pop ST(0)
    xor eax, eax
    pop ebp
    ret


; -----------------------------------------------------------------------------
; int agentjit_batch_arithmetic_f64(const double* a, const double* b,
;                                   double* out, size_t count, int op)
; [ebp+8]=a, [ebp+12]=b, [ebp+16]=out, [ebp+20]=count, [ebp+24]=op
; -----------------------------------------------------------------------------
align 16
agentjit_batch_arithmetic_f64:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi

    mov esi, [ebp + 8]          ; a
    mov edi, [ebp + 12]         ; b
    mov ebx, [ebp + 16]         ; out
    mov ecx, [ebp + 20]         ; count
    mov eax, [ebp + 24]         ; op

    test ecx, ecx
    jz .arith32_done

    xor edx, edx
.arith32_loop:
    movsd xmm0, [esi + edx*8]
    movsd xmm1, [edi + edx*8]

    cmp eax, 0
    je .a32_add
    cmp eax, 1
    je .a32_sub
    cmp eax, 2
    je .a32_mul
    cmp eax, 3
    je .a32_div
    jmp .a32_add

.a32_add:
    addsd xmm0, xmm1
    jmp .a32_store
.a32_sub:
    subsd xmm0, xmm1
    jmp .a32_store
.a32_mul:
    mulsd xmm0, xmm1
    jmp .a32_store
.a32_div:
    divsd xmm0, xmm1

.a32_store:
    movsd [ebx + edx*8], xmm0
    inc edx
    cmp edx, ecx
    jb .arith32_loop

.arith32_done:
    xor eax, eax
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret


; -----------------------------------------------------------------------------
; int agentjit_batch_lookup_hash(const uint32_t* tbl_lo, const uint32_t* tbl_hi,
;                                size_t size, uint32_t key_lo, uint32_t key_hi)
; [ebp+8]=tbl_lo, [ebp+12]=tbl_hi, [ebp+16]=size, [ebp+20]=key_lo, [ebp+24]=key_hi
; Returns: index (0..size-1) or -1
; -----------------------------------------------------------------------------
align 16
agentjit_batch_lookup_hash:
    push ebp
    mov ebp, esp
    push ebx
    push esi
    push edi

    mov esi, [ebp + 8]          ; tbl_lo
    mov edi, [ebp + 12]         ; tbl_hi
    mov ecx, [ebp + 16]         ; size
    mov eax, [ebp + 20]         ; key_lo
    mov edx, [ebp + 24]         ; key_hi

    test ecx, ecx
    jz .look32_not_found

    xor ebx, ebx
.look32_loop:
    cmp [esi + ebx*4], eax
    jne .look32_next
    cmp [edi + ebx*4], edx
    je .look32_found

.look32_next:
    inc ebx
    cmp ebx, ecx
    jb .look32_loop

.look32_not_found:
    mov eax, -1
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret

.look32_found:
    mov eax, ebx
    pop edi
    pop esi
    pop ebx
    pop ebp
    ret
