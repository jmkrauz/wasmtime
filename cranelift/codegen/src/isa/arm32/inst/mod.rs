//! This module defines 32-bit ARM specific machine instruction types.

// Some variants are not constructed, but we still want them as options in the future.
#![allow(dead_code)]

use crate::binemit::CodeOffset;
#[allow(unused)]
use crate::ir::types::{B1, B16, B32, B64, B8, F32, F64, FFLAGS, I16, I32, I64, I8, IFLAGS};
use crate::ir::{ExternalName, Opcode, SourceLoc, TrapCode, Type};
use crate::machinst::*;
use crate::{settings, CodegenError, CodegenResult};

use regalloc::Map as RegallocMap;
use regalloc::{RealReg, RealRegUniverse, Reg, RegClass, SpillSlot, VirtualReg, Writable};
use regalloc::{RegUsageCollector, RegUsageMapper, Set};

use alloc::vec::Vec;
use smallvec::{smallvec, SmallVec};
use std::string::{String, ToString};

mod args;
pub use self::args::*;
mod emit;
pub use self::emit::*;
mod regs;
pub use self::regs::*;

//=============================================================================
// Instructions (top level): definition

/// An ALU operation. This can be paired with several instruction formats
/// below (see `Inst`) in any combination.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ALUOp {
    Add,
    Adc,
    Qadd,
    Sub,
    Sbc,
    Rsb,
    Qsub,
    Mul,
    Udiv,
    Sdiv,
    And,
    Orr,
    Orn,
    Eor,
    Mvn,
    Bic,
    Lsl,
    Lsr,
    Asr,
    Ror,
}

/// Instruction formats.
#[derive(Clone, Debug)]
pub enum Inst {
    /// A no-op of zero size.
    Nop0,

    /// An ALU operation with two register sources and one register destination.
    AluRRR {
        alu_op: ALUOp,
        rd: Writable<Reg>,
        rn: Reg,
        rm: Reg,
    },

    AluRRRShift {
        alu_op: ALUOp,
        rd: Writable<Reg>,
        rn: Reg,
        rm: Reg,
        shift: Option<ShiftOpAndAmt>,
    },

    /// An ALU operation with two register sources, one of which is also a destination register.
    AluRR {
        alu_op: ALUOp,
        rd: Writable<Reg>,
        rm: Reg,
    },

    /// An ALU operation with a register source, an immediate-5 source and destination register.
    AluRRImm5 {
        alu_op: ALUOp,
        rd: Writable<Reg>,
        rm: Reg,
        imm5: u8,
    },

    /// An ALU operation with a register source, which is also a destination register
    /// and an immediate-8 source.
    AluRImm8 {
        alu_op: ALUOp,
        rd: Writable<Reg>,
        imm8: u8,
    },

    /// Move with one register source and one register destination.
    Mov {
        rd: Writable<Reg>,
        rm: Reg,
    },

    /// Move with an immediate-8 source and one register destination.
    MovImm8 {
        rd: Writable<Reg>,
        imm8: u8,
    },

    MovImm16 {
        rd: Writable<Reg>,
        imm16: u16,
    },

    Movt {
        rd: Writable<Reg>,
        imm16: u16,
    },

    Cmp {
        rn: Reg,
        rm: Reg,
    },

    Store {
        rt: Reg,
        mem: MemArg,
        srcloc: Option<SourceLoc>,
        bits: u8,
    },

    Load {
        rt: Writable<Reg>,
        mem: MemArg,
        srcloc: Option<SourceLoc>,
        bits: u8,
        sign_extend: bool,
    },

    /// A sign- or zero-extend operation.
    Extend {
        rd: Writable<Reg>,
        rm: Reg,
        from_bits: u8,
        signed: bool,
    },

    // An If-Then instruction, which makes up to four following instructions conditinal.
    It {
        cond: Cond,
        te1: Option<bool>,
        te2: Option<bool>,
        te3: Option<bool>,
    },

    /// A "breakpoint" instruction, used for e.g. traps and debug breakpoints.
    Bkpt,

    /// An instruction guaranteed to always be undefined and to trigger an illegal instruction at
    /// runtime.
    Udf {
        trap_info: (SourceLoc, TrapCode),
    },

    /// A machine call instruction.
    Call {
        dest: ExternalName,
        uses: Set<Reg>,
        defs: Set<Writable<Reg>>,
        loc: SourceLoc,
        opcode: Opcode,
    },

    /// A machine indirect-call instruction, encoded as `blx`.
    CallInd {
        rm: Reg,
        uses: Set<Reg>,
        defs: Set<Writable<Reg>>,
        loc: SourceLoc,
        opcode: Opcode,
    },

    /// A return instruction is encoded as `mov pc, lr`, however it is more convenient if it is
    /// kept as separate `Inst` entitiy.
    Ret,

    /// An unconditional branch.
    Jump {
        dest: BranchTarget,
    },

    /// A conditional branch.
    CondBr {
        taken: BranchTarget,
        not_taken: BranchTarget,
        kind: CondBrKind,
    },

    /// Lowered conditional branch: contains the original branch kind (or the
    /// inverse), but only one BranchTarget is retained. The other is
    /// implicitly the next instruction, given the final basic-block layout.
    CondBrLowered {
        target: BranchTarget,
        kind: CondBrKind,
    },

    /// As for `CondBrLowered`, but represents a condbr/uncond-br sequence (two
    /// actual machine instructions). Needed when the final block layout implies
    /// that neither arm of a conditional branch targets the fallthrough block.
    CondBrLoweredCompound {
        taken: BranchTarget,
        not_taken: BranchTarget,
        kind: CondBrKind,
    },

    /// A placeholder instruction, generating no code, meaning that a function epilogue must be
    /// inserted there.
    EpiloguePlaceholder,
}

impl Inst {
    /// Create a move instruction.
    pub fn mov(to_reg: Writable<Reg>, from_reg: Reg) -> Inst {
        Inst::Mov {
            rd: to_reg,
            rm: from_reg,
        }
    }

    /// Create an instruction that loads a constant.
    pub fn load_constant(rd: Writable<Reg>, value: u32) -> SmallVec<[Inst; 7]> {
        let imm16 = (value & 0xffff) as u16;
        let mut insts = smallvec![Inst::MovImm16 { rd, imm16 }];

        let imm16 = (value >> 16) as u16;
        if imm16 != 0 {
            insts.push(Inst::Movt { rd, imm16 });
        }
        insts
    }
}

//=============================================================================
// Instructions: get_regs

fn memarg_regs(memarg: &MemArg, collector: &mut RegUsageCollector) {
    match memarg {
        &MemArg::RegReg(rn, rm, ..) => {
            collector.add_use(rn);
            collector.add_use(rm);
        }
        &MemArg::Offset12(rn, ..) => {
            collector.add_use(rn);
        }
    }
}

fn arm32_get_regs(inst: &Inst, collector: &mut RegUsageCollector) {
    match inst {
        &Inst::Nop0
        | &Inst::It { .. }
        | &Inst::Bkpt
        | &Inst::Udf { .. }
        | &Inst::Ret
        | &Inst::Call { .. }
        | &Inst::EpiloguePlaceholder
        | &Inst::Jump { .. } => {}
        &Inst::AluRRR { rd, rn, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rn);
            collector.add_use(rm);
        }
        &Inst::AluRRRShift { rd, rn, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rn);
            collector.add_use(rm);
        }
        &Inst::AluRR { rd, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rm);
        }
        &Inst::AluRRImm5 { rd, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rm);
        }
        &Inst::AluRImm8 { rd, .. } => {
            collector.add_def(rd);
        }
        &Inst::Mov { rd, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rm);
        }
        &Inst::MovImm8 { rd, .. } => {
            collector.add_def(rd);
        }
        &Inst::MovImm16 { rd, .. } => {
            collector.add_def(rd);
        }
        &Inst::Movt { rd, .. } => {
            collector.add_def(rd);
        }
        &Inst::Cmp { rn, rm } => {
            collector.add_use(rn);
            collector.add_use(rm);
        }
        &Inst::Store { rt, ref mem, .. } => {
            collector.add_use(rt);
            memarg_regs(mem, collector);
        }
        &Inst::Load { rt, ref mem, .. } => {
            collector.add_def(rt);
            memarg_regs(mem, collector);
        }
        &Inst::Extend { rd, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rm);
        }
        &Inst::CallInd { rm, .. } => {
            collector.add_use(rm);
        }
        &Inst::CondBr { ref kind, .. }
        | &Inst::CondBrLowered { ref kind, .. }
        | &Inst::CondBrLoweredCompound { ref kind, .. } => match kind {
            CondBrKind::Zero(rt) | CondBrKind::NotZero(rt) => {
                collector.add_use(*rt);
            }
            CondBrKind::Cond(_) => {}
        },
    }
}

//=============================================================================
// Instructions: map_regs

fn arm32_map_regs(
    inst: &mut Inst,
    mapper: &RegUsageMapper
) {
    fn map(m: &RegallocMap<VirtualReg, RealReg>, r: &mut Reg) {
        if r.is_virtual() {
            let new = m.get(&r.to_virtual_reg()).cloned().unwrap().to_reg();
            *r = new;
        }
    }

    fn map_use(m: &RegUsageMapper, r: &mut Reg) {
        if r.is_virtual() {
            let new = m.get_use(r.to_virtual_reg()).unwrap().to_reg();
            *r = new;
        }
    }

    fn map_def(m: &RegUsageMapper, r: &mut Writable<Reg>) {
        if r.to_reg().is_virtual() {
            let new = m.get_def(r.to_reg().to_virtual_reg()).unwrap().to_reg();
            *r = Writable::from_reg(new);
        }
    }

    fn map_mem(m: &RegUsageMapper, mem: &mut MemArg) {
        match mem {
            &mut MemArg::RegReg(ref mut rn, ref mut rm, ..) => {
                map_use(m, rn);
                map_use(m, rm);
            }
            &mut MemArg::Offset12(ref mut rn, ..) => map_use(m, rn),
        };
    }

    fn map_br(m: &RegUsageMapper, br: &mut CondBrKind) {
        match br {
            &mut CondBrKind::Zero(ref mut reg) => map_use(m, reg),
            &mut CondBrKind::NotZero(ref mut reg) => map_use(m, reg),
            &mut CondBrKind::Cond(..) => {}
        };
    }

    match inst {
        &mut Inst::Nop0
        | &mut Inst::It { .. }
        | &mut Inst::Bkpt
        | &mut Inst::Udf { .. }
        | &mut Inst::Call { .. }
        | &mut Inst::Ret
        | &mut Inst::EpiloguePlaceholder
        | &mut Inst::Jump { .. } => {}
        &mut Inst::AluRRR {
            ref mut rd,
            ref mut rn,
            ref mut rm,
            ..
        } => {
            map_def(mapper, rd);
            map_use(mapper, rn);
            map_use(mapper, rm);
        }
        &mut Inst::AluRRRShift {
            ref mut rd,
            ref mut rn,
            ref mut rm,
            ..
        } => {
            map_def(mapper, rd);
            map_use(mapper, rn);
            map_use(mapper, rm);
        }
        &mut Inst::AluRR {
            ref mut rd,
            ref mut rm,
            ..
        } => {
            map_def(mapper, rd);
            map_use(mapper, rm);
        }
        &mut Inst::AluRRImm5 {
            ref mut rd,
            ref mut rm,
            ..
        } => {
            map_def(mapper, rd);
            map_use(mapper, rm);
        }
        &mut Inst::AluRImm8 { ref mut rd, .. } => {
            map_def(mapper, rd);
        }
        &mut Inst::Mov {
            ref mut rd,
            ref mut rm,
            ..
        } => {
            map_def(mapper, rd);
            map_use(mapper, rm);
        }
        &mut Inst::MovImm8 { ref mut rd, .. } => {
            map_def(mapper, rd);
        }
        &mut Inst::MovImm16 { ref mut rd, .. } => {
            map_def(mapper, rd);
        }
        &mut Inst::Movt { ref mut rd, .. } => {
            map_def(mapper, rd);
        }
        &mut Inst::Cmp {
            ref mut rn,
            ref mut rm,
        } => {
            map_use(mapper, rn);
            map_use(mapper, rm);
        }
        &mut Inst::Store {
            ref mut rt,
            ref mut mem,
            ..
        } => {
            map_use(mapper, rt);
            map_mem(mapper, mem);
        }
        &mut Inst::Load {
            ref mut rt,
            ref mut mem,
            ..
        } => {
            map_def(mapper, rt);
            map_mem(mapper, mem);
        }
        &mut Inst::Extend {
            ref mut rd,
            ref mut rm,
            ..
        } => {
            map_def(mapper, rd);
            map_use(mapper, rm);
        }
        &mut Inst::CallInd { ref mut rm, .. } => {
            map_use(mapper, rm);
        }
        &mut Inst::CondBr { ref mut kind, .. }
        | &mut Inst::CondBrLowered { ref mut kind, .. }
        | &mut Inst::CondBrLoweredCompound { ref mut kind, .. } => {
            map_br(mapper, kind);
        }
    }
}

//=============================================================================
// Instructions: misc functions and external interface

#[allow(unused)]
impl MachInst for Inst {
    fn get_regs(&self, collector: &mut RegUsageCollector) {
        arm32_get_regs(self, collector)
    }

    fn map_regs(&mut self, mapper: &RegUsageMapper) {
        arm32_map_regs(self, mapper);
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        match self {
            &Inst::Mov { rd, rm } => Some((rd, rm)),
            _ => None,
        }
    }

    fn is_epilogue_placeholder(&self) -> bool {
        if let Inst::EpiloguePlaceholder = self {
            true
        } else {
            false
        }
    }

    fn is_term<'a>(&'a self) -> MachTerminator<'a> {
        match self {
            &Inst::Ret | &Inst::EpiloguePlaceholder => MachTerminator::Ret,
            &Inst::Jump { dest } => MachTerminator::Uncond(dest.as_block_index().unwrap()),
            &Inst::CondBr {
                taken, not_taken, ..
            } => MachTerminator::Cond(
                taken.as_block_index().unwrap(),
                not_taken.as_block_index().unwrap(),
            ),
            &Inst::CondBrLowered { .. } => {
                // When this is used prior to branch finalization for branches
                // within an open-coded sequence, i.e. with ResolvedOffsets,
                // do not consider it a terminator. From the point of view of CFG analysis,
                // it is part of a black-box single-in single-out region, hence is not
                // denoted a terminator.
                MachTerminator::None
            }
            &Inst::CondBrLoweredCompound { .. } => {
                panic!("is_term() called after lowering branches");
            }
            _ => MachTerminator::None,
        }
    }

    fn gen_move(to_reg: Writable<Reg>, from_reg: Reg, ty: Type) -> Inst {
        assert!(ty.bits() <= 32);
        assert!(to_reg.to_reg().get_class() == from_reg.get_class());

        if from_reg.get_class() == RegClass::I32 {
            Inst::mov(to_reg, from_reg)
        } else {
            unimplemented!()
        }
    }

    fn gen_zero_len_nop() -> Inst {
        Inst::Nop0
    }

    fn gen_nop(preferred_size: usize) -> Inst {
        unimplemented!()
    }

    fn maybe_direct_reload(&self, _reg: VirtualReg, _slot: SpillSlot) -> Option<Inst> {
        None
    }

    fn rc_for_type(ty: Type) -> CodegenResult<RegClass> {
        match ty {
            I8 | I16 | I32 | B1 | B8 | B16 | B32 => Ok(RegClass::I32),
            _ => Err(CodegenError::Unsupported(format!(
                "Unexpected SSA-value type: {}",
                ty
            ))),
        }
    }

    fn gen_jump(blockindex: BlockIndex) -> Inst {
        Inst::Jump {
            dest: BranchTarget::Block(blockindex),
        }
    }

    fn with_block_rewrites(&mut self, block_target_map: &[BlockIndex]) {
        match self {
            &mut Inst::Jump { ref mut dest } => {
                dest.map(block_target_map);
            }
            &mut Inst::CondBr {
                ref mut taken,
                ref mut not_taken,
                ..
            } => {
                taken.map(block_target_map);
                not_taken.map(block_target_map);
            }
            &mut Inst::CondBrLowered { .. } => {
                // See note in `is_term()`: this is used in open-coded sequences
                // within blocks and should be left alone.
            }
            &mut Inst::CondBrLoweredCompound { .. } => {
                panic!("with_block_rewrites called after branch lowering!");
            }
            _ => {}
        }
    }

    fn with_fallthrough_block(&mut self, fallthrough: Option<BlockIndex>) {
        match self {
            &mut Inst::CondBr {
                taken,
                not_taken,
                kind,
            } => {
                if taken.as_block_index() == fallthrough
                    && not_taken.as_block_index() == fallthrough
                {
                    *self = Inst::Nop0;
                } else if taken.as_block_index() == fallthrough {
                    *self = Inst::CondBrLowered {
                        target: not_taken,
                        kind: kind.invert(),
                    };
                } else if not_taken.as_block_index() == fallthrough {
                    *self = Inst::CondBrLowered {
                        target: taken,
                        kind,
                    };
                } else {
                    // We need a compound sequence (condbr / uncond-br).
                    *self = Inst::CondBrLoweredCompound {
                        taken,
                        not_taken,
                        kind,
                    };
                }
            }
            &mut Inst::Jump { dest } => {
                if dest.as_block_index() == fallthrough {
                    *self = Inst::Nop0;
                }
            }
            _ => {}
        }
    }

    fn with_block_offsets(&mut self, my_offset: CodeOffset, targets: &[CodeOffset]) {
        match self {
            &mut Inst::CondBrLowered { ref mut target, .. } => {
                target.lower(targets, my_offset);
            }
            &mut Inst::CondBrLoweredCompound {
                ref mut taken,
                ref mut not_taken,
                ..
            } => {
                taken.lower(targets, my_offset);
                not_taken.lower(targets, my_offset + 2);
            }
            &mut Inst::Jump { ref mut dest } => {
                dest.lower(targets, my_offset);
            }
            _ => {}
        }
    }
}

//=============================================================================
// Pretty-printing of instructions.
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

impl ShowWithRRU for Inst {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        fn op_name(alu_op: ALUOp) -> &'static str {
            match alu_op {
                ALUOp::Add => "add",
                ALUOp::Adc => "adc",
                ALUOp::Qadd => "qadd",
                ALUOp::Sub => "sub",
                ALUOp::Sbc => "sbc",
                ALUOp::Rsb => "rsb",
                ALUOp::Qsub => "qsub",
                ALUOp::Mul => "mul",
                ALUOp::Udiv => "udiv",
                ALUOp::Sdiv => "sdiv",
                ALUOp::And => "and",
                ALUOp::Orr => "orr",
                ALUOp::Orn => "orn",
                ALUOp::Eor => "eor",
                ALUOp::Mvn => "mvn",
                ALUOp::Bic => "bic",
                ALUOp::Lsl => "lsl",
                ALUOp::Lsr => "lsr",
                ALUOp::Asr => "asr",
                ALUOp::Ror => "ror",
            }
        }

        fn reg_shift_str(
            shift: &Option<ShiftOpAndAmt>,
            mb_rru: Option<&RealRegUniverse>,
        ) -> String {
            if let Some(ref shift) = shift {
                format!(", {}", shift.show_rru(mb_rru))
            } else {
                "".to_string()
            }
        }

        match self {
            &Inst::Nop0 => "nop-zero-len".to_string(),
            &Inst::AluRRR { alu_op, rd, rn, rm } => {
                let op = op_name(alu_op);
                let rd = rd.show_rru(mb_rru);
                let rn = rn.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                format!("{} {}, {}, {}", op, rd, rn, rm)
            }
            &Inst::AluRRRShift {
                alu_op,
                rd,
                rn,
                rm,
                ref shift,
            } => {
                let op = op_name(alu_op);
                let rd = rd.show_rru(mb_rru);
                let rn = rn.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                let shift = reg_shift_str(shift, mb_rru);
                format!("{} {}, {}, {}{}", op, rd, rn, rm, shift)
            }
            &Inst::AluRR { alu_op, rd, rm } => {
                let op = op_name(alu_op);
                let rd = rd.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                if alu_op == ALUOp::Rsb {
                    format!("{} {}, {}, #0", op, rd, rm)
                } else {
                    format!("{} {}, {}", op, rd, rm)
                }
            }
            &Inst::AluRRImm5 {
                alu_op,
                rd,
                rm,
                imm5,
            } => {
                let op = op_name(alu_op);
                let rd = rd.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                format!("{} {}, {}, #{}", op, rd, rm, imm5)
            }
            &Inst::AluRImm8 { alu_op, rd, imm8 } => {
                let op = op_name(alu_op);
                let rd = rd.show_rru(mb_rru);
                format!("{} {}, #{}", op, rd, imm8)
            }
            &Inst::Mov { rd, rm } => {
                let rd = rd.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                format!("mov {}, {}", rd, rm)
            }
            &Inst::MovImm8 { rd, imm8 } => {
                let rd = rd.show_rru(mb_rru);
                format!("mov {}, #{}", rd, imm8)
            }
            &Inst::MovImm16 { rd, imm16 } => {
                let rd = rd.show_rru(mb_rru);
                format!("movw {}, #{}", rd, imm16)
            }
            &Inst::Movt { rd, imm16 } => {
                let rd = rd.show_rru(mb_rru);
                format!("movt {}, #{}", rd, imm16)
            }
            &Inst::Cmp { rn, rm } => {
                let rn = rn.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                format!("cmp {}, {}", rn, rm)
            }
            &Inst::Store {
                rt, ref mem, bits, ..
            } => {
                let op = match bits {
                    32 => "str",
                    16 => "strh",
                    8 => "strb",
                    _ => panic!("Unsupported Store case: {:?}", self),
                };
                let rt = rt.show_rru(mb_rru);
                let mem = mem.show_rru(mb_rru);
                format!("{} {}, {}", op, rt, mem)
            }
            &Inst::Load {
                rt,
                ref mem,
                bits,
                sign_extend,
                ..
            } => {
                let op = match (bits, sign_extend) {
                    (32, _) => "ldr",
                    (16, true) => "ldrsh",
                    (16, false) => "ldrh",
                    (8, true) => "ldrsb",
                    (8, false) => "ldrb",
                    _ => panic!("Unsupported Load case: {:?}", self),
                };
                let rt = rt.show_rru(mb_rru);
                let mem = mem.show_rru(mb_rru);
                format!("{} {}, {}", op, rt, mem)
            }
            &Inst::Extend {
                rd,
                rm,
                from_bits,
                signed,
            } => {
                let op = match (from_bits, signed) {
                    (16, true) => "sxth",
                    (16, false) => "uxth",
                    (8, true) => "sxtb",
                    (8, false) => "uxtb",
                    _ => panic!("Unsupported Extend case: {:?}", self),
                };
                let rd = rd.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                format!("{} {}, {}", op, rd, rm)
            }
            &Inst::It {
                cond,
                te1,
                te2,
                te3,
            } => {
                fn show_te(te: Option<bool>) -> &'static str {
                    match te {
                        None => "",
                        Some(true) => "t",
                        Some(false) => "e",
                    }
                }

                match (te1, te2, te3) {
                    (None, None, None)
                    | (Some(_), None, None)
                    | (Some(_), Some(_), None)
                    | (Some(_), Some(_), Some(_)) => {}
                    _ => panic!(
                        "Invalid condition combination {:?} {:?} {:?} in it instruction",
                        te1, te2, te3
                    ),
                }

                let cond = cond.show_rru(mb_rru);
                let te1 = show_te(te1);
                let te2 = show_te(te2);
                let te3 = show_te(te3);
                format!("it{}{}{} {}", te1, te2, te3, cond)
            }
            &Inst::Bkpt => "bkpt #0".to_string(),
            &Inst::Udf { .. } => "udf".to_string(),
            &Inst::Call { dest: _, .. } => format!("bl 0"),
            &Inst::CallInd { rm, .. } => {
                let rm = rm.show_rru(mb_rru);
                format!("blx {}", rm)
            }
            &Inst::Ret => "mov pc, lr".to_string(),
            &Inst::EpiloguePlaceholder => "epilogue placeholder".to_string(),
            &Inst::Jump { ref dest } => {
                let dest = dest.show_rru(mb_rru);
                format!("b {}", dest)
            }
            &Inst::CondBr {
                ref taken,
                ref not_taken,
                ref kind,
            } => {
                let taken = taken.show_rru(mb_rru);
                let not_taken = not_taken.show_rru(mb_rru);
                match kind {
                    &CondBrKind::Zero(reg) => {
                        let reg = reg.show_rru(mb_rru);
                        format!("cbz {}, {} ; b {}", reg, taken, not_taken)
                    }
                    &CondBrKind::NotZero(reg) => {
                        let reg = reg.show_rru(mb_rru);
                        format!("cbnz {}, {} ; b {}", reg, taken, not_taken)
                    }
                    &CondBrKind::Cond(c) => {
                        let c = c.show_rru(mb_rru);
                        format!("b.{} {} ; b {}", c, taken, not_taken)
                    }
                }
            }
            &Inst::CondBrLowered {
                ref target,
                ref kind,
            } => {
                let target = target.show_rru(mb_rru);
                match &kind {
                    &CondBrKind::Zero(reg) => {
                        let reg = reg.show_rru(mb_rru);
                        format!("cbz {}, {}", reg, target)
                    }
                    &CondBrKind::NotZero(reg) => {
                        let reg = reg.show_rru(mb_rru);
                        format!("cbnz {}, {}", reg, target)
                    }
                    &CondBrKind::Cond(c) => {
                        let c = c.show_rru(mb_rru);
                        format!("b.{} {}", c, target)
                    }
                }
            }
            &Inst::CondBrLoweredCompound {
                ref taken,
                ref not_taken,
                ref kind,
            } => {
                let first = Inst::CondBrLowered {
                    target: taken.clone(),
                    kind: kind.clone(),
                };
                let second = Inst::Jump {
                    dest: not_taken.clone(),
                };
                first.show_rru(mb_rru) + " ; " + &second.show_rru(mb_rru)
            }
        }
    }
}
