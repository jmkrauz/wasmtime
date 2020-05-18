//! Lowering rules for 32-bit ARM.

use crate::ir::condcodes::{FloatCC, IntCC};
use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{InstructionData, Opcode, TrapCode, Type};
use crate::machinst::lower::*;
use crate::machinst::*;

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

/// Producer of a value: either a previous instruction's output, or a register that will be
/// codegen'd separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InsnInputSource {
    Output(InsnOutput),
    Reg(Reg),
}

impl InsnInputSource {
    fn as_output(self) -> Option<InsnOutput> {
        match self {
            InsnInputSource::Output(o) => Some(o),
            _ => None,
        }
    }
}

fn get_input<C: LowerCtx<I = Inst>>(ctx: &mut C, output: InsnOutput, num: usize) -> InsnInput {
    assert!(num <= ctx.num_inputs(output.insn));
    InsnInput {
        insn: output.insn,
        input: num,
    }
}

/// Convert an instruction input to a producing instruction's output if possible (in same BB), or a
/// register otherwise.
fn input_source<C: LowerCtx<I = Inst>>(ctx: &mut C, input: InsnInput) -> InsnInputSource {
    if let Some((input_inst, result_num)) = ctx.input_inst(input.insn, input.input) {
        let out = InsnOutput {
            insn: input_inst,
            output: result_num,
        };
        InsnInputSource::Output(out)
    } else {
        let reg = ctx.input(input.insn, input.input);
        InsnInputSource::Reg(reg)
    }
}

//============================================================================
// Lowering: convert instruction outputs to result types.

/// Lower an instruction output to a 64-bit constant, if possible.
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
                &InstructionData::UnaryIeee32 { opcode: _, imm } => Some(u64::from(imm.bits())),
                &InstructionData::UnaryIeee64 { opcode: _, imm } => Some(imm.bits()),
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
    ctx.output(out.insn, out.output)
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
    let from_bits = ty_bits(ty) as u8;
    let in_reg = ctx.input(input.insn, input.input);
    match (narrow_mode, from_bits) {
        (NarrowValueMode::None, _) => in_reg,
        (NarrowValueMode::ZeroExtend, n) if n < 32 => {
            let tmp = ctx.tmp(RegClass::I32, I32);
            ctx.emit(Inst::Extend {
                rd: tmp,
                rm: in_reg,
                signed: false,
                from_bits,
            });
            tmp.to_reg()
        }
        (NarrowValueMode::SignExtend, n) if n < 32 => {
            let tmp = ctx.tmp(RegClass::I32, I32);
            ctx.emit(Inst::Extend {
                rd: tmp,
                rm: in_reg,
                signed: true,
                from_bits,
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

//============================================================================
// Lowering: addressing mode support. Takes instruction directly, rather
// than an `InsnInput`, to do more introspection.

/// Lower the address of a load or store.
pub(crate) fn lower_address<C: LowerCtx<I = Inst>>(
    ctx: &mut C,
    _elem_ty: Type,
    addends: &[InsnInput],
    offset: i32,
) -> MemArg {
    // TODO: support base_reg + scale * index_reg. For this, we would need to pattern-match shl or
    // mul instructions (Load/StoreComplex don't include scale factors).

    // Handle one reg and offset that fits in immediate, if possible.
    if addends.len() == 1 {
        let reg = input_to_reg(ctx, addends[0], NarrowValueMode::ZeroExtend);
        if let Some(memarg) = MemArg::reg_maybe_offset(reg, offset) {
            return memarg;
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

pub(crate) fn lower_constant_int<C: LowerCtx<I = Inst>>(
    ctx: &mut C,
    rd: Writable<Reg>,
    value: u64,
) {
    assert!((value >> 32) == 0x0 || (value >> 32) == (1 << 32) - 1);

    for inst in Inst::load_constant(rd, (value & ((1 << 32) - 1)) as u32) {
        ctx.emit(inst);
    }
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
pub fn condcode_is_signed(cc: IntCC) -> bool {
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
pub fn ty_bits(ty: Type) -> usize {
    match ty {
        B1 => 1,
        B8 | I8 => 8,
        B16 | I16 => 16,
        B32 | I32 | F32 => 32,
        B64 | I64 | F64 => 64,
        B128 | I128 => 128,
        IFLAGS | FFLAGS | _ => panic!("ty_bits() on unknown type: {:?}", ty),
    }
}

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

pub(crate) fn inst_fp_condcode(data: &InstructionData) -> Option<FloatCC> {
    match data {
        &InstructionData::BranchFloat { cond, .. }
        | &InstructionData::FloatCompare { cond, .. }
        | &InstructionData::FloatCond { cond, .. }
        | &InstructionData::FloatCondTrap { cond, .. } => Some(cond),
        _ => None,
    }
}

pub(crate) fn inst_trapcode(data: &InstructionData) -> Option<TrapCode> {
    match data {
        &InstructionData::Trap { code, .. }
        | &InstructionData::CondTrap { code, .. }
        | &InstructionData::IntCondTrap { code, .. }
        | &InstructionData::FloatCondTrap { code, .. } => Some(code),
        _ => None,
    }
}

/// Checks for an instance of `op` feeding the given input. Marks as merged (decrementing refcount) if so.
pub(crate) fn maybe_input_insn<C: LowerCtx<I = Inst>>(
    c: &mut C,
    input: InsnInput,
    op: Opcode,
) -> Option<IRInst> {
    if let InsnInputSource::Output(out) = input_source(c, input) {
        let data = c.data(out.insn);
        if data.opcode() == op {
            c.merged(out.insn);
            return Some(out.insn);
        }
    }
    None
}

/// Checks for an instance of `op` feeding the given input, possibly via a conversion `conv` (e.g.,
/// Bint or a bitcast). Marks one or both as merged if so, as appropriate.
pub(crate) fn maybe_input_insn_via_conv<C: LowerCtx<I = Inst>>(
    c: &mut C,
    input: InsnInput,
    op: Opcode,
    conv: Opcode,
) -> Option<IRInst> {
    if let Some(ret) = maybe_input_insn(c, input, op) {
        return Some(ret);
    }

    if let InsnInputSource::Output(out) = input_source(c, input) {
        let data = c.data(out.insn);
        if data.opcode() == conv {
            let conv_insn = out.insn;
            let conv_input = InsnInput {
                insn: conv_insn,
                input: 0,
            };
            if let Some(inner) = maybe_input_insn(c, conv_input, op) {
                c.merged(conv_insn);
                return Some(inner);
            }
        }
    }
    None
}

//=============================================================================
// Lowering-backend trait implementation.

#[allow(unused)]
impl LowerBackend for Arm32Backend {
    type MInst = Inst;

    fn lower<C: LowerCtx<I = Inst>>(&self, ctx: &mut C, ir_inst: IRInst) {
        lower_inst::lower_insn_to_regs(ctx, ir_inst);
    }

    fn lower_branch_group<C: LowerCtx<I = Inst>>(
        &self,
        ctx: &mut C,
        branches: &[IRInst],
        targets: &[BlockIndex],
        fallthrough: Option<BlockIndex>,
    ) {
        lower_inst::lower_branch(ctx, branches, targets, fallthrough)
    }
}
