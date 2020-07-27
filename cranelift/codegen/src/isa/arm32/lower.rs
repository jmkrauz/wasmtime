//! Lowering rules for 32-bit ARM.

use crate::ir::condcodes::IntCC;
use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{InstructionData, Opcode, TrapCode, Type};
use crate::machinst::lower::*;
use crate::machinst::*;
use crate::CodegenResult;

use crate::isa::arm32::inst::*;
use crate::isa::arm32::Arm32Backend;

use super::lower_inst;

use regalloc::{Reg, RegClass, Writable};

//============================================================================
// Instruction input and output "slots".
//
// We use these types to refer to operand numbers, and result numbers, together
// with the associated instruction, in a type-safe way.

/// Identifier for a particular output of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InsnOutput {
    pub(crate) insn: IRInst,
    pub(crate) output: usize,
}

/// Identifier for a particular input of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InsnInput {
    pub(crate) insn: IRInst,
    pub(crate) input: usize,
}

//============================================================================
// Lowering: convert instruction outputs to result types.

/// Lower an instruction input to a 32-bit constant, if possible.
/*pub(crate) fn input_to_const<C: LowerCtx<I = Inst>>(ctx: &mut C, input: InsnInput) -> Option<u64> {
    let input = ctx.get_input(input.insn, input.input);
    input.constant
}*/

/// Lower an instruction output to a 32-bit constant, if possible.
pub(crate) fn output_to_const<C: LowerCtx<I = Inst>>(ctx: &mut C, out: InsnOutput) -> Option<u64> {
    if out.output > 0 {
        None
    } else {
        let inst_data = ctx.data(out.insn);
        if inst_data.opcode() == Opcode::Null {
            Some(0)
        } else {
            match inst_data {
                &InstructionData::UnaryImm { opcode: _, imm } => {
                    // Only has Into for i64; we use u64 elsewhere, so we cast.
                    let imm: i64 = imm.into();
                    Some(imm as u64)
                }
                &InstructionData::UnaryBool { opcode: _, imm } => Some(u64::from(imm)),
                &InstructionData::UnaryIeee32 { .. } | &InstructionData::UnaryIeee64 { .. } => {
                    unimplemented!()
                }
                _ => None,
            }
        }
    }
}

/// How to handle narrow values loaded into registers; see note on `narrow_mode`
/// parameter to `input_to_*` below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NarrowValueMode {
    None,
    /// Zero-extend to 32 bits if original is < 32 bits.
    ZeroExtend,
    /// Sign-extend to 32 bits if original is < 32 bits.
    SignExtend,
}

/// Lower an instruction output to a reg.
pub(crate) fn output_to_reg<C: LowerCtx<I = Inst>>(ctx: &mut C, out: InsnOutput) -> Writable<Reg> {
    ctx.get_output(out.insn, out.output)
}

/// Lower an instruction input to a reg.
///
/// The given register will be extended appropriately, according to `narrow_mode`.
pub(crate) fn input_to_reg<C: LowerCtx<I = Inst>>(
    ctx: &mut C,
    input: InsnInput,
    narrow_mode: NarrowValueMode,
) -> Reg {
    let ty = ctx.input_ty(input.insn, input.input);
    let from_bits = ty.bits() as u8;
    let inputs = ctx.get_input(input.insn, input.input);
    let in_reg = if let Some(c) = inputs.constant {
        // Generate constants fresh at each use to minimize long-range register pressure.
        let to_reg = ctx.alloc_tmp(Inst::rc_for_type(ty).unwrap(), ty);
        for inst in Inst::gen_constant(to_reg, c, ty, |reg_class, ty| ctx.alloc_tmp(reg_class, ty))
            .into_iter()
        {
            ctx.emit(inst);
        }
        to_reg.to_reg()
    } else {
        ctx.use_input_reg(inputs);
        inputs.reg
    };

    match (narrow_mode, from_bits) {
        (NarrowValueMode::None, _) => in_reg,
        (NarrowValueMode::ZeroExtend, 1) => {
            let tmp = ctx.alloc_tmp(RegClass::I32, I32);
            ctx.emit(Inst::AluRRImm8 {
                alu_op: ALUOp::And,
                rd: tmp,
                rn: in_reg,
                imm8: 0x1,
            });
            tmp.to_reg()
        }
        (NarrowValueMode::ZeroExtend, n) if n < 32 => {
            let tmp = ctx.alloc_tmp(RegClass::I32, I32);
            let from_bytes = ByteAmt::from_bits(n).unwrap();
            ctx.emit(Inst::Extend {
                rd: tmp,
                rm: in_reg,
                signed: false,
                from_bytes,
            });
            tmp.to_reg()
        }
        (NarrowValueMode::SignExtend, n) if n < 32 => {
            let tmp = ctx.alloc_tmp(RegClass::I32, I32);
            let from_bytes = ByteAmt::from_bits(n).unwrap();
            ctx.emit(Inst::Extend {
                rd: tmp,
                rm: in_reg,
                signed: true,
                from_bytes,
            });
            tmp.to_reg()
        }
        (NarrowValueMode::ZeroExtend, 32) | (NarrowValueMode::SignExtend, 32) => in_reg,
        _ => panic!(
            "Unsupported input width: input ty {} bits {} mode {:?}",
            ty, from_bits, narrow_mode
        ),
    }
}

/// Lower the address of a load or store.
pub(crate) fn lower_address<C: LowerCtx<I = Inst>>(
    ctx: &mut C,
    elem_ty: Type,
    addends: &[InsnInput],
    offset: i32,
) -> MemArg {
    // Handle one reg and offset that fits in immediate, if possible.
    if addends.len() == 1 {
        let reg = input_to_reg(ctx, addends[0], NarrowValueMode::ZeroExtend);
        if let Some(memarg) = MemArg::reg_maybe_offset(reg, offset) {
            return memarg;
        } else {
            let tmp = ctx.alloc_tmp(RegClass::I32, elem_ty);
            lower_constant(ctx, tmp, offset as u64);
            return MemArg::reg_plus_reg(reg, tmp.to_reg(), 0);
        }
    }

    // Handle two regs and a zero offset, if possible.
    if addends.len() == 2 && offset == 0 {
        let ra = input_to_reg(ctx, addends[0], NarrowValueMode::ZeroExtend);
        let rb = input_to_reg(ctx, addends[1], NarrowValueMode::ZeroExtend);
        return MemArg::reg_plus_reg(ra, rb, 0);
    }

    unimplemented!()
}

pub(crate) fn lower_constant<C: LowerCtx<I = Inst>>(ctx: &mut C, rd: Writable<Reg>, value: u64) {
    assert!((value >> 32) == 0x0 || (value >> 32) == (1 << 32) - 1);

    for inst in Inst::load_constant(rd, (value & ((1 << 32) - 1)) as u32) {
        ctx.emit(inst);
    }
}

pub(crate) fn emit_cmp<C: LowerCtx<I = Inst>>(ctx: &mut C, insn: IRInst) {
    let inputs = [InsnInput { insn, input: 0 }, InsnInput { insn, input: 1 }];

    // TODO Try to commute the operands (and invert the condition) if one is an immediate.
    let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
    let rm = input_to_reg(ctx, inputs[1], NarrowValueMode::None);
    ctx.emit(Inst::Cmp { rn, rm });
}

pub(crate) fn lower_condcode(cc: IntCC) -> Cond {
    match cc {
        IntCC::Equal => Cond::Eq,
        IntCC::NotEqual => Cond::Ne,
        IntCC::SignedGreaterThanOrEqual => Cond::Ge,
        IntCC::SignedGreaterThan => Cond::Gt,
        IntCC::SignedLessThanOrEqual => Cond::Le,
        IntCC::SignedLessThan => Cond::Lt,
        IntCC::UnsignedGreaterThanOrEqual => Cond::Hs,
        IntCC::UnsignedGreaterThan => Cond::Hi,
        IntCC::UnsignedLessThanOrEqual => Cond::Ls,
        IntCC::UnsignedLessThan => Cond::Lo,
        IntCC::Overflow => Cond::Vs,
        IntCC::NotOverflow => Cond::Vc,
    }
}

/// Determines whether this condcode interprets inputs as signed or
/// unsigned.  See the documentation for the `icmp` instruction in
/// cranelift-codegen/meta/src/shared/instructions.rs for further insights
/// into this.
pub(crate) fn condcode_is_signed(cc: IntCC) -> bool {
    match cc {
        IntCC::Equal => false,
        IntCC::NotEqual => false,
        IntCC::SignedGreaterThanOrEqual => true,
        IntCC::SignedGreaterThan => true,
        IntCC::SignedLessThanOrEqual => true,
        IntCC::SignedLessThan => true,
        IntCC::UnsignedGreaterThanOrEqual => false,
        IntCC::UnsignedGreaterThan => false,
        IntCC::UnsignedLessThanOrEqual => false,
        IntCC::UnsignedLessThan => false,
        IntCC::Overflow => true,
        IntCC::NotOverflow => true,
    }
}

//=============================================================================
// Helpers for instruction lowering.

pub(crate) fn ty_is_int(ty: Type) -> bool {
    match ty {
        B1 | B8 | I8 | B16 | I16 | B32 | I32 => true,
        F32 | F64 | B128 | I128 => false,
        IFLAGS | FFLAGS => panic!("Unexpected flags type"),
        _ => panic!("ty_is_int() on unknown type: {:?}", ty),
    }
}

pub(crate) fn ldst_offset(data: &InstructionData) -> Option<i32> {
    match data {
        &InstructionData::Load { offset, .. }
        | &InstructionData::StackLoad { offset, .. }
        | &InstructionData::LoadComplex { offset, .. }
        | &InstructionData::Store { offset, .. }
        | &InstructionData::StackStore { offset, .. }
        | &InstructionData::StoreComplex { offset, .. } => Some(offset.into()),
        _ => None,
    }
}

pub(crate) fn inst_condcode(data: &InstructionData) -> Option<IntCC> {
    match data {
        &InstructionData::IntCond { cond, .. }
        | &InstructionData::BranchIcmp { cond, .. }
        | &InstructionData::IntCompare { cond, .. }
        | &InstructionData::IntCondTrap { cond, .. }
        | &InstructionData::BranchInt { cond, .. }
        | &InstructionData::IntSelect { cond, .. }
        | &InstructionData::IntCompareImm { cond, .. } => Some(cond),
        _ => None,
    }
}

pub(crate) fn inst_trapcode(data: &InstructionData) -> Option<TrapCode> {
    match data {
        &InstructionData::Trap { code, .. }
        | &InstructionData::CondTrap { code, .. }
        | &InstructionData::IntCondTrap { code, .. } => Some(code),
        &InstructionData::FloatCondTrap { code, .. } => {
            panic!("Unexpected float cond trap {:?}", code)
        }
        _ => None,
    }
}

//=============================================================================
// Lowering-backend trait implementation.

impl LowerBackend for Arm32Backend {
    type MInst = Inst;

    fn lower<C: LowerCtx<I = Inst>>(&self, ctx: &mut C, ir_inst: IRInst) -> CodegenResult<()> {
        lower_inst::lower_insn_to_regs(ctx, ir_inst)
    }

    fn lower_branch_group<C: LowerCtx<I = Inst>>(
        &self,
        ctx: &mut C,
        branches: &[IRInst],
        targets: &[MachLabel],
        fallthrough: Option<MachLabel>,
    ) -> CodegenResult<()> {
        lower_inst::lower_branch(ctx, branches, targets, fallthrough)
    }

    fn maybe_pinned_reg(&self) -> Option<Reg> {
        None
    }
}
