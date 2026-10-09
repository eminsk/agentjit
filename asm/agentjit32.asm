; =============================================================================
; AgentJIT — High-Performance 32-bit SSE2 Engine DLL (x86 FASM)
; Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
; Apache-2.0 License
; =============================================================================

format PE GUI 4.0 DLL
entry DllEntryPoint

include 'C:\proekts\FASM\INCLUDE\WIN32A.INC'

section '.text' code readable executable

proc DllEntryPoint hinstDLL, fdwReason, lpvReserved
    mov eax, 1
    ret
endp

; -----------------------------------------------------------------------------
; const char* agentjit_simd_isa(void)
; -----------------------------------------------------------------------------
align 16
agentjit_simd_isa:
    mov eax, isa_str
    ret

; Include microkernels
include 'agentjit32_kernel.inc'

section '.data' data readable
isa_str db 'SSE2 (FASM x86 32-bit, 128-bit SIMD)', 0

section '.edata' export data readable
export 'agentjit32.dll',\
       agentjit_version,               'agentjit_version',\
       agentjit_simd_isa,              'agentjit_simd_isa',\
       agentjit_fast_hash,             'agentjit_fast_hash',\
       agentjit_eval_null_guards,      'agentjit_eval_null_guards',\
       agentjit_eval_range_guards_f64, 'agentjit_eval_range_guards_f64',\
       agentjit_vector_dot,            'agentjit_vector_dot',\
       agentjit_cosine_similarity,     'agentjit_cosine_similarity',\
       agentjit_semantic_guard_cosine, 'agentjit_semantic_guard_cosine',\
       agentjit_batch_arithmetic_f64,  'agentjit_batch_arithmetic_f64',\
       agentjit_batch_lookup_hash,     'agentjit_batch_lookup_hash'

section '.reloc' fixups data readable discardable
if $=$$
    dd 0,8
end if
