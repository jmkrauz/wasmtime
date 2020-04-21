//! 32-bit ARM ISA: binary code emission.

use crate::binemit::CodeOffset;
use crate::isa::arm32::inst::*;

/// Memory addressing mode finalization: convert "special" modes (e.g.,
/// generic arbitrary stack offset) into real addressing modes, possibly by
/// emitting some helper instructions that come immediately before the use
/// of this amode.
#[allow(unused)]
pub fn mem_finalize(insn_off: CodeOffset, mem: &MemArg) -> (Vec<Inst>, MemArg) {
    unimplemented!()
}

#[allow(unused)]
impl<O: MachSectionOutput> MachInstEmit<O> for Inst {
    fn emit(&self, sink: &mut O) {
        match self {
            &Inst::Nop0 => {}
        }
    }
}
