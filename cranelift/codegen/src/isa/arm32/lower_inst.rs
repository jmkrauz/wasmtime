//! Lower a single Cranelift instruction into vcode.

use crate::ir::types::*;
use crate::ir::Inst as IRInst;
use crate::ir::{InstructionData, Opcode};
use crate::machinst::lower::*;
use crate::machinst::*;
use crate::CodegenResult;

use crate::isa::arm32::abi::*;
use crate::isa::arm32::inst::*;

use regalloc::RegClass;

use core::convert::TryFrom;
use smallvec::SmallVec;

use super::lower::*;

/// Actually codegen an instruction's results into registers.
pub(crate) fn lower_insn_to_regs<C: LowerCtx<I = Inst>>(
    ctx: &mut C,
    insn: IRInst,
) -> CodegenResult<()> {
    let op = ctx.data(insn).opcode();
    let inputs: SmallVec<[InsnInput; 4]> = (0..ctx.num_inputs(insn))
        .map(|i| InsnInput { insn, input: i })
        .collect();
    let outputs: SmallVec<[InsnOutput; 2]> = (0..ctx.num_outputs(insn))
        .map(|i| InsnOutput { insn, output: i })
        .collect();
    let ty = if outputs.len() > 0 {
        let ty = ctx.output_ty(insn, 0);
        if ty.bits() > 32 {
            panic!("Cannot lower inst with type {}!", ty);
        }
        Some(ty)
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
        | Opcode::IaddIfcin
        | Opcode::IaddIfcout
        | Opcode::IaddIfcarry
        | Opcode::Isub
        | Opcode::IsubIfbin
        | Opcode::IsubIfbout
        | Opcode::IsubIfborrow
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
                Opcode::IaddIfcin => ALUOp::Adc,
                Opcode::IaddIfcout => ALUOp::Adds,
                Opcode::IaddIfcarry => ALUOp::Adcs,
                Opcode::Isub => ALUOp::Sub,
                Opcode::IsubIfbin => ALUOp::Sbc,
                Opcode::IsubIfbout => ALUOp::Subs,
                Opcode::IsubIfborrow => ALUOp::Sbcs,
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
            match ty {
                Some(I32) | Some(B32) => {}
                _ => unimplemented!(),
            }

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
        Opcode::Smulhi | Opcode::Umulhi => {
            let ty = ty.unwrap();
            let is_signed = op == Opcode::Smulhi;
            match ty {
                I32 => {
                    let rd_hi = output_to_reg(ctx, outputs[0]);
                    let rd_lo = ctx.alloc_tmp(RegClass::I32, ty);
                    let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
                    let rm = input_to_reg(ctx, inputs[1], NarrowValueMode::None);
                    let alu_op = if is_signed {
                        ALUOp::Smull
                    } else {
                        ALUOp::Umull
                    };
                    ctx.emit(Inst::AluRRRR {
                        alu_op,
                        rd_hi,
                        rd_lo,
                        rn,
                        rm,
                    });
                }
                I16 | I8 => {
                    let narrow_mode = if is_signed {
                        NarrowValueMode::SignExtend
                    } else {
                        NarrowValueMode::ZeroExtend
                    };
                    let rd = output_to_reg(ctx, outputs[0]);
                    let rn = input_to_reg(ctx, inputs[0], narrow_mode);
                    let rm = input_to_reg(ctx, inputs[1], narrow_mode);
                    ctx.emit(Inst::AluRRR {
                        alu_op: ALUOp::Mul,
                        rd,
                        rn,
                        rm,
                    });
                    let shift_amt = if ty == I16 { 16 } else { 8 };
                    let alu_op = if is_signed { ALUOp::Asr } else { ALUOp::Lsr };
                    ctx.emit(Inst::AluRRImm8 {
                        alu_op,
                        rd,
                        rn: rd.to_reg(),
                        imm8: shift_amt,
                    });
                }
                _ => panic!("Unexpected type {} in lower {}!", ty, op),
            }
        }
        Opcode::Bnot => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rm = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            ctx.emit(Inst::AluRR {
                alu_op: ALUOp::Mvn,
                rd,
                rm,
            });
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
            let signed = op == Opcode::Sextend;
            let rm = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let rd = output_to_reg(ctx, outputs[0]);

            if to_bits != 32 {
                unimplemented!()
            }

            if from_bits == 1 {
                assert!(!signed);
                ctx.emit(Inst::AluRRImm8 {
                    alu_op: ALUOp::And,
                    rd,
                    rn: rd.to_reg(),
                    imm8: 0x1,
                });
            } else if from_bits < to_bits {
                ctx.emit(Inst::Extend {
                    rd,
                    rm,
                    from_bits,
                    signed,
                });
            }
        }
        Opcode::Bint | Opcode::Breduce | Opcode::Bextend | Opcode::Ireduce => {
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::ZeroExtend);
            let rd = output_to_reg(ctx, outputs[0]);
            let ty = ctx.input_ty(insn, 0);
            ctx.emit(Inst::gen_move(rd, rn, ty));
        }
        Opcode::Copy => {
            let rd = output_to_reg(ctx, outputs[0]);
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let ty = ctx.input_ty(insn, 0);
            ctx.emit(Inst::gen_move(rd, rn, ty));
        }
        Opcode::StackAddr | Opcode::StackLoad => {
            let (stack_slot, offset) = match *ctx.data(insn) {
                InstructionData::StackLoad {
                    opcode: Opcode::StackAddr,
                    stack_slot,
                    offset,
                } => (stack_slot, offset),
                _ => unreachable!(),
            };
            let rd = output_to_reg(ctx, outputs[0]);
            let offset: i32 = offset.into();
            let offset = u32::try_from(offset).unwrap();
            let inst = if op == Opcode::StackAddr {
                ctx.abi().stackslot_addr(stack_slot, offset, rd)
            } else {
                ctx.abi()
                    .load_stackslot(stack_slot, offset, ty.unwrap(), rd)
            };
            ctx.emit(inst);
        }
        Opcode::StackStore => {
            let (stack_slot, offset) = match *ctx.data(insn) {
                InstructionData::StackStore {
                    opcode: Opcode::StackStore,
                    stack_slot,
                    offset,
                    ..
                } => (stack_slot, offset),
                _ => unreachable!(),
            };
            let rn = input_to_reg(ctx, inputs[0], NarrowValueMode::None);
            let ty = ctx.input_ty(insn, 0);
            let offset: i32 = offset.into();
            let inst =
                ctx.abi()
                    .store_stackslot(stack_slot, u32::try_from(offset).unwrap(), ty, rn);
            ctx.emit(inst);
        }
        Opcode::Debugtrap => {
            ctx.emit(Inst::Bkpt);
        }
        Opcode::Trap => {
            let trap_info = (ctx.srcloc(insn), inst_trapcode(ctx.data(insn)).unwrap());
            ctx.emit(Inst::Udf { trap_info })
        }
        Opcode::FallthroughReturn | Opcode::Return => {
            for (i, input) in inputs.iter().enumerate() {
                let reg = input_to_reg(ctx, *input, NarrowValueMode::ZeroExtend);
                let retval_reg = ctx.retval(i);
                let ty = ctx.input_ty(insn, i);
                ctx.emit(Inst::gen_move(retval_reg, reg, ty));
            }
        }
        Opcode::Call | Opcode::CallIndirect => {
            let loc = ctx.srcloc(insn);
            let (mut abi, inputs) = match op {
                Opcode::Call => {
                    let (extname, dist) = ctx.call_target(insn).unwrap();
                    let extname = extname.clone();
                    let sig = ctx.call_sig(insn).unwrap();
                    assert!(inputs.len() == sig.params.len());
                    assert!(outputs.len() == sig.returns.len());
                    (
                        Arm32ABICall::from_func(sig, &extname, dist, loc)?,
                        &inputs[..],
                    )
                }
                Opcode::CallIndirect => {
                    let ptr = input_to_reg(ctx, inputs[0], NarrowValueMode::ZeroExtend);
                    let sig = ctx.call_sig(insn).unwrap();
                    assert!(inputs.len() - 1 == sig.params.len());
                    assert!(outputs.len() == sig.returns.len());
                    (Arm32ABICall::from_ptr(sig, ptr, loc, op)?, &inputs[1..])
                }
                _ => unreachable!(),
            };
            assert!(inputs.len() == abi.num_args());
            for (i, input) in inputs.iter().enumerate() {
                // ugly
                if i <= 3 {
                    let arg_reg = input_to_reg(ctx, *input, NarrowValueMode::None);
                    abi.emit_copy_reg_to_arg(ctx, i, arg_reg);
                }
            }
            abi.emit_call(ctx);
            for (i, output) in outputs.iter().enumerate() {
                let retval_reg = output_to_reg(ctx, *output);
                abi.emit_copy_retval_to_reg(ctx, i, retval_reg);
            }
        }
        _ => panic!("Lowering {} unimplemented!", op),
    }

    Ok(())
}

pub(crate) fn lower_branch<C: LowerCtx<I = Inst>>(
    ctx: &mut C,
    branches: &[IRInst],
    targets: &[MachLabel],
    fallthrough: Option<MachLabel>,
) -> CodegenResult<()> {
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
        let taken = BranchTarget::Label(targets[0]);
        let not_taken = match op1 {
            Opcode::Jump => BranchTarget::Label(targets[1]),
            Opcode::Fallthrough => BranchTarget::Label(fallthrough.unwrap()),
            _ => unreachable!(), // assert above.
        };
        match op0 {
            Opcode::Brz | Opcode::Brnz => {
                let rn = input_to_reg(
                    ctx,
                    InsnInput {
                        insn: branches[0],
                        input: 0,
                    },
                    NarrowValueMode::ZeroExtend,
                );
                let kind = match op0 {
                    Opcode::Brz => CondBrKind::Cond(Cond::Eq),
                    Opcode::Brnz => CondBrKind::Cond(Cond::Ne),
                    _ => unreachable!(),
                };

                ctx.emit(Inst::CmpImm8 { rn, imm8: 0 });
                ctx.emit(Inst::CondBr {
                    taken,
                    not_taken,
                    kind,
                });
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
                    dest: BranchTarget::Label(targets[0]),
                });
            }
            Opcode::BrTable => unimplemented!(),
            _ => panic!("Unknown branch type!"),
        }
    }

    Ok(())
}
