//! This module defines 32-bit ARM specific machine instruction types.

// Some variants are not constructed, but we still want them as options in the future.
#![allow(dead_code)]

use crate::binemit::CodeOffset;
#[allow(unused)]
use crate::ir::types::{B1, B16, B32, B64, B8, F32, F64, FFLAGS, I16, I32, I64, I8, IFLAGS};
use crate::ir::{ExternalName, Opcode, SourceLoc, TrapCode, Type};
use crate::machinst::*;
use crate::{settings, CodegenError, CodegenResult};

use regalloc::{RealRegUniverse, Reg, RegClass, SpillSlot, VirtualReg, Writable};
use regalloc::{RegUsageCollector, RegUsageMapper};

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::convert::TryInto;
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
    Adds,
    Adc,
    Adcs,
    Qadd,
    Sub,
    Subs,
    Sbc,
    Sbcs,
    Rsb,
    Qsub,
    Mul,
    Smull,
    Umull,
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

    AluRRRR {
        alu_op: ALUOp,
        rd_hi: Writable<Reg>,
        rd_lo: Writable<Reg>,
        rn: Reg,
        rm: Reg,
    },

    AluRRImm12 {
        alu_op: ALUOp,
        rd: Writable<Reg>,
        rn: Reg,
        imm12: u16,
    },

    // https://static.docs.arm.com/ddi0406/c/DDI0406C_C_arm_architecture_reference_manual.pdf#G10.4954509
    AluRRImm8 {
        alu_op: ALUOp,
        rd: Writable<Reg>,
        rn: Reg,
        imm8: u8,
    },

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

    CmpImm8 {
        rn: Reg,
        imm8: u8,
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

    Push {
        reg_list: SmallVec<[Reg; 16]>,
    },

    Pop {
        reg_list: SmallVec<[Writable<Reg>; 16]>,
    },

    /// A machine call instruction.
    Call {
        dest: Box<ExternalName>,
        uses: Box<Vec<Reg>>,
        defs: Box<Vec<Writable<Reg>>>,
        loc: SourceLoc,
        opcode: Opcode,
    },
    /// A machine indirect-call instruction.
    CallInd {
        rm: Reg,
        uses: Box<Vec<Reg>>,
        defs: Box<Vec<Writable<Reg>>>,
        loc: SourceLoc,
        opcode: Opcode,
    },

    /// Load an inline symbol reference.
    LoadExtName {
        rt: Writable<Reg>,
        name: ExternalName,
        srcloc: SourceLoc,
        offset: i32,
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

    /// A one-way conditional branch, invisible to the CFG processing; used *only* as part of
    /// straight-line sequences in code to be emitted.
    OneWayCondBr {
        target: BranchTarget,
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
    pub fn load_constant(rd: Writable<Reg>, value: u32) -> SmallVec<[Inst; 4]> {
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
        &Inst::AluRRRR {
            rd_hi,
            rd_lo,
            rn,
            rm,
            ..
        } => {
            collector.add_def(rd_hi);
            collector.add_def(rd_lo);
            collector.add_use(rn);
            collector.add_use(rm);
        }
        &Inst::AluRRImm12 { rd, rn, .. } => {
            collector.add_def(rd);
            collector.add_use(rn);
        }
        &Inst::AluRRImm8 { rd, rn, .. } => {
            collector.add_def(rd);
            collector.add_use(rn);
        }
        &Inst::AluRImm8 { rd, .. } => {
            collector.add_def(rd);
        }
        &Inst::Mov { rd, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rm);
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
        &Inst::CmpImm8 { rn, .. } => {
            collector.add_use(rn);
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
        &Inst::Push { ref reg_list } => {
            for reg in reg_list {
                collector.add_use(*reg);
            }
        }
        &Inst::Pop { ref reg_list } => {
            for reg in reg_list {
                collector.add_def(*reg);
            }
        }
        &Inst::CallInd { rm, .. } => {
            collector.add_use(rm);
        }
        &Inst::LoadExtName { rt, .. } => {
            collector.add_def(rt);
        }
        &Inst::CondBr { ref kind, .. } => match kind {
            CondBrKind::Zero(rt) | CondBrKind::NotZero(rt) => {
                collector.add_use(*rt);
            }
            CondBrKind::Cond(_) => {}
        },
        &Inst::OneWayCondBr { ref kind, .. } => match kind {
            CondBrKind::Zero(rt) | CondBrKind::NotZero(rt) => {
                collector.add_use(*rt);
            }
            CondBrKind::Cond(_) => {}
        },
    }
}

//=============================================================================
// Instructions: map_regs

fn arm32_map_regs<RUM: RegUsageMapper>(inst: &mut Inst, mapper: &RUM) {
    fn map_use<RUM: RegUsageMapper>(m: &RUM, r: &mut Reg) {
        if r.is_virtual() {
            let new = m.get_use(r.to_virtual_reg()).unwrap().to_reg();
            *r = new;
        }
    }

    fn map_def<RUM: RegUsageMapper>(m: &RUM, r: &mut Writable<Reg>) {
        if r.to_reg().is_virtual() {
            let new = m.get_def(r.to_reg().to_virtual_reg()).unwrap().to_reg();
            *r = Writable::from_reg(new);
        }
    }

    fn map_mem<RUM: RegUsageMapper>(m: &RUM, mem: &mut MemArg) {
        match mem {
            &mut MemArg::RegReg(ref mut rn, ref mut rm, ..) => {
                map_use(m, rn);
                map_use(m, rm);
            }
            &mut MemArg::Offset12(ref mut rn, ..) => map_use(m, rn),
        };
    }

    fn map_br<RUM: RegUsageMapper>(m: &RUM, br: &mut CondBrKind) {
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
        &mut Inst::AluRRRR {
            ref mut rd_hi,
            ref mut rd_lo,
            ref mut rn,
            ref mut rm,
            ..
        } => {
            map_def(mapper, rd_hi);
            map_def(mapper, rd_lo);
            map_use(mapper, rn);
            map_use(mapper, rm);
        }
        &mut Inst::AluRRImm12 {
            ref mut rd,
            ref mut rn,
            ..
        } => {
            map_def(mapper, rd);
            map_use(mapper, rn);
        }
        &mut Inst::AluRRImm8 {
            ref mut rd,
            ref mut rn,
            ..
        } => {
            map_def(mapper, rd);
            map_use(mapper, rn);
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
        &mut Inst::CmpImm8 { ref mut rn, .. } => {
            map_use(mapper, rn);
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
        &mut Inst::Push { ref mut reg_list } => {
            for reg in reg_list {
                map_use(mapper, reg);
            }
        }
        &mut Inst::Pop { ref mut reg_list } => {
            for reg in reg_list {
                map_def(mapper, reg);
            }
        }
        &mut Inst::CallInd { ref mut rm, .. } => {
            map_use(mapper, rm);
        }
        &mut Inst::LoadExtName { ref mut rt, .. } => {
            map_def(mapper, rt);
        }
        &mut Inst::CondBr { ref mut kind, .. } => {
            map_br(mapper, kind);
        }
        &mut Inst::OneWayCondBr { ref mut kind, .. } => {
            map_br(mapper, kind);
        }
    }
}

//=============================================================================
// Instructions: misc functions and external interface

#[allow(unused)]
impl MachInst for Inst {
    type LabelUse = LabelUse;

    fn get_regs(&self, collector: &mut RegUsageCollector) {
        arm32_get_regs(self, collector)
    }

    fn map_regs<RUM: RegUsageMapper>(&mut self, mapper: &RUM) {
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
            &Inst::Jump { dest } => MachTerminator::Uncond(dest.as_label().unwrap()),
            &Inst::CondBr {
                taken, not_taken, ..
            } => MachTerminator::Cond(taken.as_label().unwrap(), not_taken.as_label().unwrap()),
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

    fn gen_constant(to_reg: Writable<Reg>, value: u64, ty: Type) -> SmallVec<[Inst; 4]> {
        match ty {
            B1 | I8 | B8 | I16 | B16 | I32 | B32 => {
                let value: i64 = value as i64;
                if value >= (1 << 32) || value < -(1 << 32) {
                    panic!("Cannot load constant value {}", value)
                }
                let value: i32 = value.try_into().unwrap();
                Inst::load_constant(to_reg, value as u32)
            }
            _ => unimplemented!(),
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
            IFLAGS | FFLAGS => Ok(RegClass::I32),
            _ => Err(CodegenError::Unsupported(format!(
                "Unexpected SSA-value type: {}",
                ty
            ))),
        }
    }

    fn gen_jump(target: MachLabel) -> Inst {
        Inst::Jump {
            dest: BranchTarget::Label(target),
        }
    }

    fn reg_universe(_flags: &settings::Flags) -> RealRegUniverse {
        create_reg_universe()
    }

    fn worst_case_size() -> CodeOffset {
        // LoadExtName
        12
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
                ALUOp::Adds => "adds",
                ALUOp::Adc => "adc",
                ALUOp::Adcs => "adcs",
                ALUOp::Qadd => "qadd",
                ALUOp::Sub => "sub",
                ALUOp::Subs => "subs",
                ALUOp::Sbc => "sbc",
                ALUOp::Sbcs => "sbcs",
                ALUOp::Rsb => "rsb",
                ALUOp::Qsub => "qsub",
                ALUOp::Mul => "mul",
                ALUOp::Smull => "smull",
                ALUOp::Umull => "umull",
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
                format!("{} {}, {}", op, rd, rm)
            }
            &Inst::AluRRRR {
                alu_op,
                rd_hi,
                rd_lo,
                rn,
                rm,
            } => {
                let op = op_name(alu_op);
                let rd_hi = rd_hi.show_rru(mb_rru);
                let rd_lo = rd_lo.show_rru(mb_rru);
                let rn = rn.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                format!("{} {}, {}, {}, {}", op, rd_lo, rd_hi, rn, rm)
            }
            &Inst::AluRRImm12 {
                alu_op,
                rd,
                rn,
                imm12,
            } => {
                let op = op_name(alu_op);
                let rd = rd.show_rru(mb_rru);
                let rn = rn.show_rru(mb_rru);
                format!("{} {}, {}, #{}", op, rd, rn, imm12)
            }
            &Inst::AluRRImm8 {
                alu_op,
                rd,
                rn,
                imm8,
            } => {
                let op = op_name(alu_op);
                let rd = rd.show_rru(mb_rru);
                let rn = rn.show_rru(mb_rru);
                format!("{} {}, {}, #{}", op, rd, rn, imm8)
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
            &Inst::CmpImm8 { rn, imm8 } => {
                let rn = rn.show_rru(mb_rru);
                format!("cmp {}, #{}", rn, imm8)
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
            &Inst::Push { ref reg_list } => {
                assert!(!reg_list.is_empty());
                let first_reg = reg_list[0].show_rru(mb_rru);
                let regs: String = reg_list
                    .iter()
                    .skip(1)
                    .map(|r| [",", &r.show_rru(mb_rru)].join(" "))
                    .collect();
                format!("push {{{}{}}}", first_reg, regs)
            }
            &Inst::Pop { ref reg_list } => {
                assert!(!reg_list.is_empty());
                let first_reg = reg_list[0].show_rru(mb_rru);
                let regs: String = reg_list
                    .iter()
                    .skip(1)
                    .map(|r| [",", &r.show_rru(mb_rru)].join(" "))
                    .collect();
                format!("pop {{{}{}}}", first_reg, regs)
            }
            &Inst::Call { dest: _, .. } => format!("bl 0"),
            &Inst::CallInd { rm, .. } => {
                let rm = rm.show_rru(mb_rru);
                format!("blx {}", rm)
            }
            &Inst::LoadExtName {
                rt,
                ref name,
                offset,
                srcloc: _srcloc,
            } => {
                let rt = rt.show_rru(mb_rru);
                format!("ldr {} [pc, #4] ; b 4 ; data {:?} + {}", rt, name, offset)
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
            &Inst::OneWayCondBr {
                ref target,
                ref kind,
            } => {
                let target = target.show_rru(mb_rru);
                match kind {
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
        }
    }
}

//=============================================================================
// Label fixups and jump veneers.

/// Different forms of label references for different instruction formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelUse {
    /// 6-bit branch offset used by 16-bit cbz and cbnz instructions.
    Branch6,

    /// 20-bit branch offset used by 32-bit conditional jumps.
    Branch20,

    /// 24-bit branch offset used by 32-bit uncoditional jump instruction.
    Branch24,
}

impl MachInstLabelUse for LabelUse {
    /// Alignment for veneer code. Every instruction must be 4-byte-aligned.
    const ALIGN: CodeOffset = 2;

    /// Maximum PC-relative range (positive), inclusive.
    fn max_pos_range(self) -> CodeOffset {
        match self {
            LabelUse::Branch6 => (1 << 7) + 2,
            LabelUse::Branch20 => (1 << 20) + 2,
            LabelUse::Branch24 => (1 << 24) + 2,
        }
    }

    /// Maximum PC-relative range (negative).
    fn max_neg_range(self) -> CodeOffset {
        match self {
            LabelUse::Branch6 => 0,
            LabelUse::Branch20 => (1 << 20) - 4,
            LabelUse::Branch24 => (1 << 24) - 4,
        }
    }

    /// Size of window into code needed to do the patch.
    fn patch_size(self) -> CodeOffset {
        if self == LabelUse::Branch6 {
            2
        } else {
            4
        }
    }

    /// Perform the patch.
    fn patch(self, buffer: &mut [u8], use_offset: CodeOffset, label_offset: CodeOffset) {
        let off = (label_offset as i64) - (use_offset as i64);
        debug_assert!(off <= self.max_pos_range() as i64);
        debug_assert!(off >= -(self.max_neg_range() as i64));
        let off = off - 4;
        match self {
            LabelUse::Branch6 => {
                let off = off as u16 >> 1;
                let off_inserted = ((off & 0x1f) << 3) | ((off & 0x20) << 4);
                let insn = u16::from_le_bytes([buffer[0], buffer[1]]);
                let insn = (insn & !0x02f8) | off_inserted;
                buffer[0..2].clone_from_slice(&u16::to_le_bytes(insn));
            }
            LabelUse::Branch20 => {
                let off = off as u32 >> 1;
                let imm11 = (off & 0x7ff) as u16;
                let imm6 = ((off >> 11) & 0x3f) as u16;
                let j1 = ((off >> 17) & 0x1) as u16;
                let j2 = ((off >> 18) & 0x1) as u16;
                let s = ((off >> 19) & 0x1) as u16;
                let insn_fst = u16::from_le_bytes([buffer[0], buffer[1]]);
                let insn_fst = (insn_fst & !0x43f) | imm6 | (s << 10);
                let insn_snd = u16::from_le_bytes([buffer[2], buffer[3]]);
                let insn_snd = (insn_snd & !0x2fff) | imm11 | (j2 << 11) | (j1 << 13);
                buffer[0..2].clone_from_slice(&u16::to_le_bytes(insn_fst));
                buffer[2..4].clone_from_slice(&u16::to_le_bytes(insn_snd));
            }
            LabelUse::Branch24 => {
                let off = off as u32 >> 1;
                let imm11 = (off & 0x7ff) as u16;
                let imm10 = ((off >> 11) & 0x3ff) as u16;
                let s = ((off >> 23) & 0x1) as u16;
                let j1 = (((off >> 22) & 0x1) as u16 ^ s) ^ 0x1;
                let j2 = (((off >> 21) & 0x1) as u16 ^ s) ^ 0x1;
                let insn_fst = u16::from_le_bytes([buffer[0], buffer[1]]);
                let insn_fst = (insn_fst & !0x07ff) | imm10 | (s << 10);
                let insn_snd = u16::from_le_bytes([buffer[2], buffer[3]]);
                let insn_snd = (insn_snd & !0x2fff) | imm11 | (j2 << 11) | (j1 << 13);
                buffer[0..2].clone_from_slice(&u16::to_le_bytes(insn_fst));
                buffer[2..4].clone_from_slice(&u16::to_le_bytes(insn_snd));
            }
        }
    }

    fn supports_veneer(self) -> bool {
        false
    }

    /// How large is the veneer, if supported?
    fn veneer_size(self) -> CodeOffset {
        0
    }

    /// Generate a veneer into the buffer, given that this veneer is at `veneer_offset`, and return
    /// an offset and label-use for the veneer's use of the original label.
    fn generate_veneer(
        self,
        _buffer: &mut [u8],
        _veneer_offset: CodeOffset,
    ) -> (CodeOffset, LabelUse) {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_branch6() {
        let label_use = LabelUse::Branch6;
        let mut buffer = 0xb100_u16.to_le_bytes(); // cbz r0
        let use_offset: CodeOffset = 0;
        let label_offset: CodeOffset = label_use.max_pos_range();
        label_use.patch(&mut buffer, use_offset, label_offset);
        assert_eq!(u16::from_le_bytes(buffer), 0xb3f8);
    }

    #[test]
    fn patch_branch20() {
        let label_use = LabelUse::Branch20;
        let mut buffer = 0x8000_f000_u32.to_le_bytes(); // beq
        let use_offset: CodeOffset = 0;
        let label_offset: CodeOffset = label_use.max_pos_range();
        label_use.patch(&mut buffer, use_offset, label_offset);
        assert_eq!(u16::from_le_bytes([buffer[0], buffer[1]]), 0xf03f);
        assert_eq!(u16::from_le_bytes([buffer[2], buffer[3]]), 0xafff);

        let mut buffer = 0x8000_f000_u32.to_le_bytes(); // beq
        let use_offset = label_use.max_neg_range();
        let label_offset: CodeOffset = 0;
        label_use.patch(&mut buffer, use_offset, label_offset);
        assert_eq!(u16::from_le_bytes([buffer[0], buffer[1]]), 0xf400);
        assert_eq!(u16::from_le_bytes([buffer[2], buffer[3]]), 0x8000);
    }

    #[test]
    fn patch_branch24() {
        let label_use = LabelUse::Branch24;
        let mut buffer = 0x9000_f000_u32.to_le_bytes();
        let use_offset: CodeOffset = 0;
        let label_offset: CodeOffset = label_use.max_pos_range();
        label_use.patch(&mut buffer, use_offset, label_offset);
        assert_eq!(u16::from_le_bytes([buffer[0], buffer[1]]), 0xf3ff);
        assert_eq!(u16::from_le_bytes([buffer[2], buffer[3]]), 0x97ff);

        let mut buffer = 0x9000_f000_u32.to_le_bytes();
        let use_offset = label_use.max_neg_range();
        let label_offset: CodeOffset = 0;
        label_use.patch(&mut buffer, use_offset, label_offset);
        assert_eq!(u16::from_le_bytes([buffer[0], buffer[1]]), 0xf400);
        assert_eq!(u16::from_le_bytes([buffer[2], buffer[3]]), 0x9000);
    }
}