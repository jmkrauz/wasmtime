//! Lowering rules for 32-bit ARM.

use crate::ir::condcodes::{FloatCC, IntCC};
use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{InstructionData, Opcode, TrapCode, Type};
use crate::machinst::lower::*;
use crate::machinst::*;

use crate::isa::arm32::abi::*;
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

//============================================================================
// Lowering: convert instruction outputs to result types.

/// Lower an instruction output to a 64-bit constant, if possible.
fn output_to_const<C: LowerCtx<I = Inst>>(ctx: &mut C, out: InsnOutput) -> Option<u64> {
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
enum NarrowValueMode {
    None,
    /// Zero-extend to 32 bits if original is < 32 bits.
    ZeroExtend,
    /// Sign-extend to 32 bits if original is < 32 bits.
    SignExtend,
}

/// Lower an instruction output to a reg.
fn output_to_reg<C: LowerCtx<I = Inst>>(ctx: &mut C, out: InsnOutput) -> Writable<Reg> {
    ctx.output(out.insn, out.output)
}

/// Lower an instruction input to a reg.
///
/// The given register will be extended appropriately, according to `narrow_mode`.
fn input_to_reg<C: LowerCtx<I = Inst>>(
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
fn lower_address<C: LowerCtx<I = Inst>>(
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

fn lower_constant_int<C: LowerCtx<I = Inst>>(ctx: &mut C, rd: Writable<Reg>, value: u64) {
    assert!((value >> 32) == 0x0 || (value >> 32) == (1 << 32) - 1);

    for inst in Inst::load_constant(rd, (value & ((1 << 32) - 1)) as u32) {
        ctx.emit(inst);
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
        Opcode::Iconst | Opcode::Bconst | Opcode::Null => {
            let value = output_to_const(ctx, outputs[0]).unwrap();
            let rd = output_to_reg(ctx, outputs[0]);
            lower_constant_int(ctx, rd, value);
        }
        Opcode::Iadd
        | Opcode::Isub
        | Opcode::Band
        | Opcode::Bor
        | Opcode::Bxor
        | Opcode::BandNot
        | Opcode::BorNot => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let rm = input_to_reg(ctx, inputs[1], NarrowValueMode::None);
            let alu_op = match op {
                Opcode::Iadd => ALUOp::Add,
                Opcode::Isub => ALUOp::Sub,
                Opcode::Band => ALUOp::And,
                Opcode::Bor => ALUOp::Orr,
                Opcode::Bxor => ALUOp::Eor,
                Opcode::BandNot => ALUOp::Bic,
                Opcode::BorNot => ALUOp::Orn,
                _ => unreachable!(),
            };
            ctx.emit(Inst::AluRRRShift {
                alu_op,
                rd,
                rn,
                rm,
                shift: None,
            });
        }
        Opcode::SaddSat
        | Opcode::SsubSat
        | Opcode::Imul
        | Opcode::Udiv
        | Opcode::Sdiv
        | Opcode::Ishl
        | Opcode::Ushr
        | Opcode::Sshr
        | Opcode::Rotr => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let rm = input_to_reg(ctx, inputs[1], NarrowValueMode::None);
            let alu_op = match op {
                Opcode::SaddSat => ALUOp::Qadd,
                Opcode::SsubSat => ALUOp::Qsub,
                Opcode::Imul => ALUOp::Mul,
                Opcode::Udiv => ALUOp::Udiv,
                Opcode::Sdiv => ALUOp::Sdiv,
                Opcode::Ishl => ALUOp::Lsl,
                Opcode::Ushr => ALUOp::Lsr,
                Opcode::Sshr => ALUOp::Asr,
                Opcode::Rotr => ALUOp::Ror,
                _ => unreachable!(),
            };
            ctx.emit(Inst::AluRRR { alu_op, rd, rn, rm });
        }
        Opcode::Icmp => {
            let condcode = inst_condcode(ctx.data(insn)).unwrap();
            let cond = lower_condcode(condcode);
            let is_signed = condcode_is_signed(condcode);
            let narrow_mode = if is_signed {
                NarrowValueMode::SignExtend
            } else {
                NarrowValueMode::ZeroExtend
            };
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], narrow_mode);
            let rm = input_to_reg(ctx, inputs[1], narrow_mode);
            ctx.emit(Inst::Cmp { rn, rm });
            ctx.emit(Inst::It {
                cond,
                te1: Some(true),
                te2: Some(false),
                te3: None,
            });
            ctx.emit(Inst::MovImm16 { rd, imm16: 0x1 });
            ctx.emit(Inst::MovImm16 { rd, imm16: 0x0 });
        }
        Opcode::Store
        | Opcode::Istore8
        | Opcode::Istore16
        | Opcode::Istore32
        | Opcode::StoreComplex
        | Opcode::Istore8Complex
        | Opcode::Istore16Complex
        | Opcode::Istore32Complex => {
            let off = ldst_offset(ctx.data(insn)).unwrap();
            let elem_ty = match op {
                Opcode::Istore8 | Opcode::Istore8Complex => I8,
                Opcode::Istore16 | Opcode::Istore16Complex => I16,
                Opcode::Istore32 | Opcode::Istore32Complex => I32,
                Opcode::Store | Opcode::StoreComplex => ctx.input_ty(insn, 0),
                _ => unreachable!(),
            };
            if !ty_is_int(elem_ty) {
                unimplemented!()
            }

            let mem = lower_address(ctx, elem_ty, &inputs[1..], off);
            let rt = input_to_reg(ctx, inputs[0], NarrowValueMode::None);

            let memflags = ctx.memflags(insn).expect("memory flags");
            let srcloc = if !memflags.notrap() {
                Some(ctx.srcloc(insn))
            } else {
                None
            };
            let bits = elem_ty.bits() as u8;

            ctx.emit(Inst::Store {
                rt,
                mem,
                srcloc,
                bits,
            });
        }
        Opcode::Load
        | Opcode::Uload8
        | Opcode::Sload8
        | Opcode::Uload16
        | Opcode::Sload16
        | Opcode::Uload32
        | Opcode::Sload32
        | Opcode::LoadComplex
        | Opcode::Uload8Complex
        | Opcode::Sload8Complex
        | Opcode::Uload16Complex
        | Opcode::Sload16Complex
        | Opcode::Uload32Complex
        | Opcode::Sload32Complex => {
            let off = ldst_offset(ctx.data(insn)).unwrap();
            let elem_ty = match op {
                Opcode::Sload8 | Opcode::Uload8 => I8,
                Opcode::Sload16
                | Opcode::Uload16
                | Opcode::Sload16Complex
                | Opcode::Uload16Complex => I16,
                Opcode::Sload32
                | Opcode::Uload32
                | Opcode::Sload32Complex
                | Opcode::Uload32Complex => I32,
                Opcode::Load | Opcode::LoadComplex => ctx.output_ty(insn, 0),
                _ => unreachable!(),
            };
            let sign_extend = match op {
                Opcode::Sload8
                | Opcode::Sload8Complex
                | Opcode::Sload16
                | Opcode::Sload16Complex
                | Opcode::Sload32
                | Opcode::Sload32Complex => true,
                _ => false,
            };

            if !ty_is_int(elem_ty) {
                unimplemented!()
            }

            let mem = lower_address(ctx, elem_ty, &inputs[..], off);
            let rt = output_to_reg(ctx, outputs[0]);

            let memflags = ctx.memflags(insn).expect("memory flags");
            let srcloc = if !memflags.notrap() {
                Some(ctx.srcloc(insn))
            } else {
                None
            };
            let bits = elem_ty.bits() as u8;

            ctx.emit(Inst::Load {
                rt,
                mem,
                srcloc,
                bits,
                sign_extend,
            });
        }
        Opcode::Uextend | Opcode::Sextend => {
            let output_ty = ty.unwrap();
            let input_ty = ctx.input_ty(insn, 0);
            let from_bits = ty_bits(input_ty) as u8;
            let to_bits = ty_bits(output_ty) as u8;

            if to_bits != 32 {
                unimplemented!()
            }

            if from_bits < to_bits {
                let signed = op == Opcode::Sextend;
                // If we reach this point, we weren't able to incorporate the extend as
                // a register-mode on another instruction, so we have a 'None'
                // narrow-value/extend mode here, and we emit the explicit instruction.
                let rm = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
                let rd = output_to_reg(ctx, outputs[0]);
                ctx.emit(Inst::Extend {
                    rd,
                    rm,
                    from_bits,
                    signed,
                });
            }
        }
        Opcode::Debugtrap => {
            ctx.emit(Inst::Bkpt);
        }
        Opcode::Trap => {
            let trap_info = (ctx.srcloc(insn), inst_trapcode(ctx.data(insn)).unwrap());
            ctx.emit(Inst::Udf { trap_info })
        }
        Opcode::Return => {
            ctx.emit(Inst::Ret);
        }
        Opcode::Call | Opcode::CallIndirect => {
            let loc = ctx.srcloc(insn);
            let (abi, inputs) = match op {
                Opcode::Call => {
                    let extname = ctx.call_target(insn).unwrap();
                    let extname = extname.clone();
                    let sig = ctx.call_sig(insn).unwrap();
                    assert!(inputs.len() == sig.params.len());
                    assert!(outputs.len() == sig.returns.len());
                    (Arm32ABICall::from_func(sig, &extname, loc), &inputs[..])
                }
                Opcode::CallIndirect => {
                    let ptr = input_to_reg(ctx, inputs[0], NarrowValueMode::ZeroExtend);
                    let sig = ctx.call_sig(insn).unwrap();
                    assert!(inputs.len() - 1 == sig.params.len());
                    assert!(outputs.len() == sig.returns.len());
                    (Arm32ABICall::from_ptr(sig, ptr, loc, op), &inputs[1..])
                }
                _ => unreachable!(),
            };

            for inst in abi.gen_stack_pre_adjust().into_iter() {
                ctx.emit(inst);
            }
            assert!(inputs.len() == abi.num_args());
            for (i, input) in inputs.iter().enumerate() {
                let arg_reg = input_to_reg(ctx, *input, NarrowValueMode::None);
                ctx.emit(abi.gen_copy_reg_to_arg(i, arg_reg));
            }
            for inst in abi.gen_call().into_iter() {
                ctx.emit(inst);
            }
            for (i, output) in outputs.iter().enumerate() {
                let retval_reg = output_to_reg(ctx, *output);
                ctx.emit(abi.gen_copy_retval_to_reg(i, retval_reg));
            }
            for inst in abi.gen_stack_post_adjust().into_iter() {
                ctx.emit(inst);
            }
        }
        _ => panic!("Lowering {} unimplemented!", op),
    }
}

//=============================================================================
// Helpers for instruction lowering.
fn ty_bits(ty: Type) -> usize {
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

fn ty_is_int(ty: Type) -> bool {
    match ty {
        B1 | B8 | I8 | B16 | I16 | B32 | I32 => true,
        F32 | F64 | B128 | I128 => false,
        IFLAGS | FFLAGS => panic!("Unexpected flags type"),
        _ => panic!("ty_is_int() on unknown type: {:?}", ty),
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
                        unimplemented!()
                    } else if let Some(fcmp_insn) =
                        maybe_input_insn_via_conv(ctx, flag_input, Opcode::Fcmp, Opcode::Bint)
                    {
                        unimplemented!()
                    } else {
                        let rt = input_to_reg(
                            ctx,
                            InsnInput {
                                insn: branches[0],
                                input: 0,
                            },
                            NarrowValueMode::ZeroExtend,
                        );
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
