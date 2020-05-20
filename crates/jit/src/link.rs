//! Linking for JIT-compiled code.

use crate::Compilation;
use cranelift_codegen::binemit::Reloc;
use std::ptr::{read_unaligned, write_unaligned};
use wasmtime_environ::{Module, Relocation, RelocationTarget};
use wasmtime_runtime::libcalls;
use wasmtime_runtime::VMFunctionBody;

/// Links a module that has been compiled with `compiled_module` in `wasmtime-environ`.
///
/// Performs all required relocations inside the function code, provided the necessary metadata.
pub fn link_module(module: &Module, compilation: &Compilation) {
    for (i, function_relocs) in compilation.relocations.iter() {
        for r in function_relocs.iter() {
            let fatptr: *const [VMFunctionBody] = compilation.finished_functions[i];
            let body = fatptr as *const VMFunctionBody;
            apply_reloc(module, compilation, body, r);
        }
    }

    for (i, function_relocs) in compilation.trampoline_relocations.iter() {
        for r in function_relocs.iter() {
            println!("tramopline relocation");
            let body = compilation.trampolines[&i] as *const VMFunctionBody;
            apply_reloc(module, compilation, body, r);
        }
    }
}

fn apply_reloc(
    module: &Module,
    compilation: &Compilation,
    body: *const VMFunctionBody,
    r: &Relocation,
) {
    use self::libcalls::*;
    let target_func_address: usize = match r.reloc_target {
        RelocationTarget::UserFunc(index) => match module.local.defined_func_index(index) {
            Some(f) => {
                let fatptr: *const [VMFunctionBody] = compilation.finished_functions[f];
                fatptr as *const VMFunctionBody as usize
            }
            None => panic!("direct call to import"),
        },
        RelocationTarget::LibCall(libcall) => {
            use cranelift_codegen::ir::LibCall::*;
            match libcall {
                CeilF32 => wasmtime_f32_ceil as usize,
                FloorF32 => wasmtime_f32_floor as usize,
                TruncF32 => wasmtime_f32_trunc as usize,
                NearestF32 => wasmtime_f32_nearest as usize,
                CeilF64 => wasmtime_f64_ceil as usize,
                FloorF64 => wasmtime_f64_floor as usize,
                TruncF64 => wasmtime_f64_trunc as usize,
                NearestF64 => wasmtime_f64_nearest as usize,
                other => panic!("unexpected libcall: {}", other),
            }
        }
        RelocationTarget::JumpTable(func_index, jt) => {
            match module.local.defined_func_index(func_index) {
                Some(f) => {
                    let offset = *compilation
                        .jt_offsets
                        .get(f)
                        .and_then(|ofs| ofs.get(jt))
                        .expect("func jump table");
                    let fatptr: *const [VMFunctionBody] = compilation.finished_functions[f];
                    fatptr as *const VMFunctionBody as usize + offset as usize
                }
                None => panic!("func index of jump table"),
            }
        }
    };

    match r.reloc {
        #[cfg(target_pointer_width = "32")]
        Reloc::Abs4 => unsafe {
            let reloc_address = body.add(r.offset as usize) as usize;
            let reloc_addend = r.addend as isize;
            let reloc_abs = (target_func_address as u32)
                .checked_add(reloc_addend as u32)
                .unwrap();
            write_unaligned(reloc_address as *mut u32, reloc_abs);
        },
        #[cfg(target_pointer_width = "64")]
        Reloc::Abs8 => unsafe {
            let reloc_address = body.add(r.offset as usize) as usize;
            let reloc_addend = r.addend as isize;
            let reloc_abs = (target_func_address as u64)
                .checked_add(reloc_addend as u64)
                .unwrap();
            write_unaligned(reloc_address as *mut u64, reloc_abs);
        },
        #[cfg(target_pointer_width = "32")]
        Reloc::X86PCRel4 => unsafe {
            let reloc_address = body.add(r.offset as usize) as usize;
            let reloc_addend = r.addend as isize;
            let reloc_delta_u32 = (target_func_address as u32)
                .wrapping_sub(reloc_address as u32)
                .checked_add(reloc_addend as u32)
                .unwrap();
            write_unaligned(reloc_address as *mut u32, reloc_delta_u32);
        },
        #[cfg(target_pointer_width = "32")]
        Reloc::X86CallPCRel4 => unsafe {
            let reloc_address = body.add(r.offset as usize) as usize;
            let reloc_addend = r.addend as isize;
            let reloc_delta_u32 = (target_func_address as u32)
                .wrapping_sub(reloc_address as u32)
                .wrapping_add(reloc_addend as u32);
            write_unaligned(reloc_address as *mut u32, reloc_delta_u32);
        },
        Reloc::X86PCRelRodata4 => {
            // ignore
        }
        Reloc::Arm32Call => unsafe {
            let reloc_address = body.add(r.offset as usize) as usize;
            let reloc_addend = r.addend as isize;
            let reloc_delta = (target_func_address as u32).wrapping_sub(reloc_address as u32);
            let reloc_delta = reloc_delta.wrapping_add(reloc_addend as u32);
            assert!((reloc_delta as i32) < (1 << 25));
            assert!((reloc_delta as i32) >= -(1 << 25));
            let delta_bits = reloc_delta >> 1;

            let insn = read_unaligned(reloc_address as *const u32);
            assert!(insn == 0b11110_0_0000000000_11_0_1_0_00000000000);

            let imm11 = delta_bits & 0x7ff;
            let imm10 = (delta_bits >> 11) & 0x3ff;
            let s = delta_bits >> 23;
            let i1 = delta_bits >> 22;
            let i2 = delta_bits >> 21;
            let j1 = (i1 ^ s) ^ 0x1;
            let j2 = (i2 ^ s) ^ 0x1;

            let new_insn = insn | imm11 | (imm10 << 16) | (s << 26) | (j1 << 13) | (j2 << 11);
            write_unaligned(reloc_address as *mut u32, new_insn);
        },
        Reloc::Arm64Call => unsafe {
            let reloc_address = body.add(r.offset as usize) as usize;
            let reloc_addend = r.addend as isize;
            let reloc_delta = (target_func_address as u64).wrapping_sub(reloc_address as u64);
            // TODO: come up with a PLT-like solution for longer calls. We can't extend the
            // code segment at this point, but we could conservatively allocate space at the
            // end of the function during codegen, a fixed amount per call, to allow for
            // potential branch islands.
            assert!((reloc_delta as i64) < (1 << 27));
            assert!((reloc_delta as i64) >= -(1 << 27));
            let reloc_delta = reloc_delta as u32;
            let reloc_delta = reloc_delta.wrapping_add(reloc_addend as u32);
            let delta_bits = reloc_delta >> 2;
            let insn = read_unaligned(reloc_address as *const u32);
            let new_insn = (insn & 0xfc00_0000) | (delta_bits & 0x03ff_ffff);
            write_unaligned(reloc_address as *mut u32, new_insn);
        },
        _ => panic!("unsupported reloc kind"),
    }
}
