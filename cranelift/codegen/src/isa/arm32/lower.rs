//! Lowering rules for 32-bit ARM.

use crate::ir::Inst as IRInst;
use crate::machinst::lower::*;
use crate::machinst::*;

use crate::isa::arm32::inst::*;
use crate::isa::arm32::ArmBackend;

//=============================================================================
// Lowering-backend trait implementation.

#[allow(unused)]
impl LowerBackend for ArmBackend {
    type MInst = Inst;

    fn lower<C: LowerCtx<I = Inst>>(&self, ctx: &mut C, ir_inst: IRInst) {
        unimplemented!()
    }

    fn lower_branch_group<C: LowerCtx<I = Inst>>(
        &self,
        ctx: &mut C,
        branches: &[IRInst],
        targets: &[BlockIndex],
        fallthrough: Option<BlockIndex>,
    ) {
        unimplemented!()
    }
}
