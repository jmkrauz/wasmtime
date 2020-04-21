//! This module defines 32-bit ARM specific machine instruction types.

// Some variants are not constructed, but we still want them as options in the future.
#![allow(dead_code)]

use crate::binemit::CodeOffset;
use crate::ir::Type;
use crate::machinst::*;

use regalloc::Map as RegallocMap;
use regalloc::{RealReg, RealRegUniverse, Reg, RegClass, SpillSlot, VirtualReg, Writable};
use regalloc::RegUsageCollector;

use alloc::vec::Vec;
use std::string::{String, ToString};

mod args;
pub use self::args::*;
mod emit;
pub use self::emit::*;
mod regs;
pub use self::regs::*;

/// Instruction formats.
#[derive(Clone, Debug)]
pub enum Inst {
    /// A no-op of zero size.
    Nop0,
}

//=============================================================================
// Instructions: misc functions and external interface

#[allow(unused)]
impl MachInst for Inst {
    fn get_regs(&self, collector: &mut RegUsageCollector) {
        unimplemented!()
    }

    fn map_regs(
        &mut self,
        pre_map: &RegallocMap<VirtualReg, RealReg>,
        post_map: &RegallocMap<VirtualReg, RealReg>,
    ) {
        unimplemented!()
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        unimplemented!()
    }

    fn is_epilogue_placeholder(&self) -> bool {
        unimplemented!()
    }

    fn is_term<'a>(&'a self) -> MachTerminator<'a> {
        unimplemented!()
    }

    fn gen_move(to_reg: Writable<Reg>, from_reg: Reg, ty: Type) -> Inst {
        unimplemented!()
    }

    fn gen_zero_len_nop() -> Inst {
        Inst::Nop0
    }

    fn gen_nop(preferred_size: usize) -> Inst {
        unimplemented!()
    }

    fn maybe_direct_reload(&self, _reg: VirtualReg, _slot: SpillSlot) -> Option<Inst> {
        unimplemented!()
    }

    fn rc_for_type(ty: Type) -> RegClass {
        unimplemented!()
    }

    fn gen_jump(blockindex: BlockIndex) -> Inst {
        unimplemented!()
    }

    fn with_block_rewrites(&mut self, block_target_map: &[BlockIndex]) {
        unimplemented!()
    }

    fn with_fallthrough_block(&mut self, fallthrough: Option<BlockIndex>) {
        unimplemented!()
    }

    fn with_block_offsets(&mut self, my_offset: CodeOffset, targets: &[CodeOffset]) {
        unimplemented!()
    }

    fn reg_universe() -> RealRegUniverse {
        create_reg_universe()
    }
}

//=============================================================================
// Pretty-printing of instructions.

#[allow(unused)]
fn mem_finalize_for_show(mem: &MemArg, mb_rru: Option<&RealRegUniverse>) -> (String, MemArg) {
    let (mem_insts, mem) = mem_finalize(0, mem);
    let mut mem_str = mem_insts
        .into_iter()
        .map(|inst| inst.show_rru(mb_rru))
        .collect::<Vec<_>>()
        .join(" ; ");
    if !mem_str.is_empty() {
        mem_str += " ; ";
    }

    (mem_str, mem)
}

#[allow(unused)]
impl ShowWithRRU for Inst {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            &Inst::Nop0 => "nop-zero-len".to_string(),
        }
    }
}
