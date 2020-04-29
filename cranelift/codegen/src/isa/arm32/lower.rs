//! Lowering rules for 32-bit ARM.

use crate::ir::condcodes::{FloatCC, IntCC};
use crate::ir::Inst as IRInst;
use crate::ir::{InstructionData, Opcode, TrapCode, Type};
use crate::machinst::lower::*;
use crate::machinst::*;

use crate::isa::arm32::inst::*;
use crate::isa::arm32::Arm32Backend;

use regalloc::{Reg, RegClass, Writable};
use smallvec::SmallVec;

//============================================================================
// Instruction input and output "slots".
//
// We use these types to refer to operand numbers, and result numbers, together
// with the associated instruction, in a type-safe way.

/// Identifier for a particular output of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InsnOutput {
    insn: IRInst,
    output: usize,
}

/// Identifier for a particular input of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InsnInput {
    insn: IRInst,
    input: usize,
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

fn lower_condcode(cc: IntCC) -> Cond {
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
// Top-level instruction lowering entry point, for one instruction.

/// Actually codegen an instruction's results into registers.
fn lower_insn_to_regs<C: LowerCtx<I = Inst>>(ctx: &mut C, insn: IRInst) {
    let op = ctx.data(insn).opcode();
    let inputs: SmallVec<[InsnInput; 4]> = (0..ctx.num_inputs(insn))
        .map(|i| InsnInput { insn, input: i })
        .collect();
    let outputs: SmallVec<[InsnOutput; 2]> = (0..ctx.num_outputs(insn))
        .map(|i| InsnOutput { insn, output: i })
        .collect();
    let ty = if outputs.len() > 0 {
        Some(ctx.output_ty(insn, 0))
    } else {
        None
    };

    match op {
        _ => unimplemented!(),
    }
}

fn ldst_offset(data: &InstructionData) -> Option<i32> {
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

fn inst_condcode(data: &InstructionData) -> Option<IntCC> {
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

fn inst_fp_condcode(data: &InstructionData) -> Option<FloatCC> {
    match data {
        &InstructionData::BranchFloat { cond, .. }
        | &InstructionData::FloatCompare { cond, .. }
        | &InstructionData::FloatCond { cond, .. }
        | &InstructionData::FloatCondTrap { cond, .. } => Some(cond),
        _ => None,
    }
}

fn inst_trapcode(data: &InstructionData) -> Option<TrapCode> {
    match data {
        &InstructionData::Trap { code, .. }
        | &InstructionData::CondTrap { code, .. }
        | &InstructionData::IntCondTrap { code, .. }
        | &InstructionData::FloatCondTrap { code, .. } => Some(code),
        _ => None,
    }
}

/// Checks for an instance of `op` feeding the given input. Marks as merged (decrementing refcount) if so.
fn maybe_input_insn<C: LowerCtx<I = Inst>>(
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
fn maybe_input_insn_via_conv<C: LowerCtx<I = Inst>>(
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

fn lower_icmp_or_ifcmp_to_flags<C: LowerCtx<I = Inst>>(ctx: &mut C, insn: IRInst, is_signed: bool) {
    unimplemented!()
}

//=============================================================================
// Lowering-backend trait implementation.

#[allow(unused)]
impl LowerBackend for Arm32Backend {
    type MInst = Inst;

    fn lower<C: LowerCtx<I = Inst>>(&self, ctx: &mut C, ir_inst: IRInst) {
        lower_insn_to_regs(ctx, ir_inst);
    }

    fn lower_branch_group<C: LowerCtx<I = Inst>>(
        &self,
        ctx: &mut C,
        branches: &[IRInst],
        targets: &[BlockIndex],
        fallthrough: Option<BlockIndex>,
    ) {
        // A block should end with at most two branches. The first may be a
        // conditional branch; a conditional branch can be followed only by an
        // unconditional branch or fallthrough. Otherwise, if only one branch,
        // it may be an unconditional branch, a fallthrough, a return, or a
        // trap. These conditions are verified by `is_ebb_basic()` during the
        // verifier pass.
        assert!(branches.len() <= 2);

        if branches.len() == 2 {
            // Must be a conditional branch followed by an unconditional branch.
            let op0 = ctx.data(branches[0]).opcode();
            let op1 = ctx.data(branches[1]).opcode();

            assert!(op1 == Opcode::Jump || op1 == Opcode::Fallthrough);
            let taken = BranchTarget::Block(targets[0]);
            let not_taken = match op1 {
                Opcode::Jump => BranchTarget::Block(targets[1]),
                Opcode::Fallthrough => BranchTarget::Block(fallthrough.unwrap()),
                _ => unreachable!(), // assert above.
            };
            match op0 {
                Opcode::Brz | Opcode::Brnz => {
                    let flag_input = InsnInput {
                        insn: branches[0],
                        input: 0,
                    };
                    if let Some(icmp_insn) =
                        maybe_input_insn_via_conv(ctx, flag_input, Opcode::Icmp, Opcode::Bint)
                    {
                        let condcode = inst_condcode(ctx.data(icmp_insn)).unwrap();
                        let cond = lower_condcode(condcode);
                        let is_signed = condcode_is_signed(condcode);
                        let negated = op0 == Opcode::Brz;
                        let cond = if negated { cond.invert() } else { cond };

                        lower_icmp_or_ifcmp_to_flags(ctx, icmp_insn, is_signed);
                        ctx.emit(Inst::CondBr {
                            taken,
                            not_taken,
                            kind: CondBrKind::Cond(cond),
                        });
                    } else if let Some(fcmp_insn) =
                        maybe_input_insn_via_conv(ctx, flag_input, Opcode::Fcmp, Opcode::Bint)
                    {
                        unimplemented!()
                    } else {
                        let rt = ctx.input(branches[0], 0); // Probably need to change this one
                        let kind = match op0 {
                            Opcode::Brz => CondBrKind::Zero(rt),
                            Opcode::Brnz => CondBrKind::NotZero(rt),
                            _ => unreachable!(),
                        };
                        ctx.emit(Inst::CondBr {
                            taken,
                            not_taken,
                            kind,
                        });
                    }
                }
                Opcode::BrIcmp => unimplemented!(),
                Opcode::Brif => unimplemented!(),
                Opcode::Brff => unimplemented!(),
                _ => unimplemented!(),
            }
        } else {
            // Must be an unconditional branch or an indirect branch.
            let op = ctx.data(branches[0]).opcode();
            match op {
                Opcode::Jump | Opcode::Fallthrough => {
                    assert!(branches.len() == 1);
                    // In the Fallthrough case, the machine-independent driver
                    // fills in `targets[0]` with our fallthrough block, so this
                    // is valid for both Jump and Fallthrough.
                    ctx.emit(Inst::Jump {
                        dest: BranchTarget::Block(targets[0]),
                    });
                }
                Opcode::BrTable => unimplemented!(),
                _ => panic!("Unknown branch type!"),
            }
        }
    }
}
