//! This module defines 32-bit ARM specific machine instruction types.

// Some variants are not constructed, but we still want them as options in the future.
#![allow(dead_code)]

use crate::binemit::CodeOffset;
#[allow(unused)]
use crate::ir::types::{B1, B16, B32, B64, B8, F32, F64, FFLAGS, I16, I32, I64, I8, IFLAGS};
use crate::ir::{Opcode, SourceLoc, TrapCode, Type};
use crate::machinst::*;

use regalloc::Map as RegallocMap;
use regalloc::{RealReg, RealRegUniverse, Reg, RegClass, SpillSlot, VirtualReg, Writable};
use regalloc::{RegUsageCollector, Set};

use alloc::vec::Vec;
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
    Sub,
    Sbc,
    Rsb,
    Mul,
    And,
    Orr,
    Eor,
    Mvn,
    Bic,
    Lsl,
    Lsr,
    Asr,
    Ror,
    Cmp,
    Cmn,
    Tst,
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

    /// An ALU operation with two register sources, one of which is also a destination register.
    AluRR {
        alu_op: ALUOp,
        rd: Writable<Reg>,
        rm: Reg,
    },

    /// An ALU operation with two register sources.
    AluRRNoResult { alu_op: ALUOp, rn: Reg, rm: Reg },

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

    /// An ALU operation with a register source and an immediate-8 source.
    AluRImm8NoResult { alu_op: ALUOp, rn: Reg, imm8: u8 },

    /// Move with one register source and one register destination.
    MovRR { rd: Writable<Reg>, rm: Reg },

    /// Move with an immediate-8 source and one register destination.
    MovRImm8 { rd: Writable<Reg>, imm8: u8 },

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
        sign_extend: Option<bool>,
    },

    /// A sign- or zero-extend operation.
    Extend {
        rd: Writable<Reg>,
        rm: Reg,
        from_bits: u8,
        signed: bool,
    },

    /// A "breakpoint" instruction, used for e.g. traps and debug breakpoints.
    Bkpt,

    /// An instruction guaranteed to always be undefined and to trigger an illegal instruction at
    /// runtime.
    Udf { trap_info: (SourceLoc, TrapCode) },

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
    Jump { dest: BranchTarget },

    /// A conditional branch.
    CondBr {
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
        assert!(to_reg.to_reg().get_class() == from_reg.get_class());
        if from_reg.get_class() == RegClass::I32 {
            Inst::MovRR {
                rd: to_reg,
                rm: from_reg,
            }
        } else {
            unimplemented!()
        }
    }
}

//=============================================================================
// Instructions: get_regs

fn memarg_regs(memarg: &MemArg, collector: &mut RegUsageCollector) {
    match memarg {
        &MemArg::RegReg(rn, rm) => {
            collector.add_use(rn);
            collector.add_use(rm);
        }
        &MemArg::Offset5(rn, ..) => {
            collector.add_use(rn);
        }
        &MemArg::SPOffset(..) => {
            collector.add_use(sp_reg());
        }
    }
}

fn arm32_get_regs(inst: &Inst, collector: &mut RegUsageCollector) {
    match inst {
        &Inst::Nop0
        | &Inst::Bkpt
        | &Inst::Udf { .. }
        | &Inst::Ret
        | &Inst::EpiloguePlaceholder
        | &Inst::Jump { .. }
        | &Inst::CondBr { .. } => {}
        &Inst::AluRRR { rd, rn, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rn);
            collector.add_use(rm);
        }
        &Inst::AluRR { rd, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rm);
        }
        &Inst::AluRRNoResult { rn, rm, .. } => {
            collector.add_use(rn);
            collector.add_use(rm);
        }
        &Inst::AluRRImm5 { rd, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rm);
        }
        &Inst::AluRImm8 { rd, .. } => {
            collector.add_def(rd);
        }
        &Inst::AluRImm8NoResult { rn, .. } => {
            collector.add_use(rn);
        }
        &Inst::MovRR { rd, rm, .. } => {
            collector.add_def(rd);
            collector.add_use(rm);
        }
        &Inst::MovRImm8 { rd, .. } => {
            collector.add_def(rd);
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
    }
}

//=============================================================================
// Instructions: map_regs

fn arm32_map_regs(
    inst: &mut Inst,
    pre_map: &RegallocMap<VirtualReg, RealReg>,
    post_map: &RegallocMap<VirtualReg, RealReg>,
) {
    fn map(m: &RegallocMap<VirtualReg, RealReg>, r: &mut Reg) {
        if r.is_virtual() {
            let new = m.get(&r.to_virtual_reg()).cloned().unwrap().to_reg();
            *r = new;
        }
    }

    fn map_wr(m: &RegallocMap<VirtualReg, RealReg>, r: &mut Writable<Reg>) {
        let mut reg = r.to_reg();
        map(m, &mut reg);
        *r = Writable::from_reg(reg);
    }

    fn map_mem(u: &RegallocMap<VirtualReg, RealReg>, mem: &mut MemArg) {
        match mem {
            &mut MemArg::RegReg(ref mut rn, ref mut rm) => {
                map(u, rn);
                map(u, rm);
            }
            &mut MemArg::Offset5(ref mut rn, ..) => map(u, rn),
            &mut MemArg::SPOffset(..) => {}
        };
    }

    let u = pre_map; // For brevity below.
    let d = post_map;

    match inst {
        &mut Inst::Nop0
        | &mut Inst::Bkpt
        | &mut Inst::Udf { .. }
        | &mut Inst::Ret
        | &mut Inst::EpiloguePlaceholder
        | &mut Inst::Jump { .. }
        | &mut Inst::CondBr { .. } => {}
        &mut Inst::AluRRR {
            ref mut rd,
            ref mut rn,
            ref mut rm,
            ..
        } => {
            map_wr(d, rd);
            map(u, rn);
            map(u, rm);
        }
        &mut Inst::AluRR {
            ref mut rd,
            ref mut rm,
            ..
        } => {
            map_wr(d, rd);
            map(u, rm);
        }
        &mut Inst::AluRRNoResult {
            ref mut rn,
            ref mut rm,
            ..
        } => {
            map(u, rn);
            map(u, rm);
        }
        &mut Inst::AluRRImm5 {
            ref mut rd,
            ref mut rm,
            ..
        } => {
            map_wr(d, rd);
            map(u, rm);
        }
        &mut Inst::AluRImm8 { ref mut rd, .. } => {
            map_wr(d, rd);
        }
        &mut Inst::AluRImm8NoResult { ref mut rn, .. } => {
            map(u, rn);
        }
        &mut Inst::MovRR {
            ref mut rd,
            ref mut rm,
            ..
        } => {
            map_wr(d, rd);
            map(u, rm);
        }
        &mut Inst::MovRImm8 { ref mut rd, .. } => {
            map_wr(d, rd);
        }
        &mut Inst::Store {
            ref mut rt,
            ref mut mem,
            ..
        } => {
            map(u, rt);
            map_mem(u, mem);
        }
        &mut Inst::Load {
            ref mut rt,
            ref mut mem,
            ..
        } => {
            map_wr(d, rt);
            map_mem(u, mem);
        }
        &mut Inst::Extend {
            ref mut rd,
            ref mut rm,
            ..
        } => {
            map_wr(d, rd);
            map(u, rm);
        }
        &mut Inst::CallInd { ref mut rm, .. } => {
            map(u, rm);
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

    fn map_regs(
        &mut self,
        pre_map: &RegallocMap<VirtualReg, RealReg>,
        post_map: &RegallocMap<VirtualReg, RealReg>,
    ) {
        arm32_map_regs(self, pre_map, post_map);
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        match self {
            &Inst::MovRR { rd, rm } => Some((rd, rm)),
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
            _ => MachTerminator::None,
        }
    }

    fn gen_move(to_reg: Writable<Reg>, from_reg: Reg, ty: Type) -> Inst {
        if ty.bits() <= 32 {
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

    fn rc_for_type(ty: Type) -> RegClass {
        match ty {
            I8 | I16 | I32 | B1 | B8 | B16 | B32 => RegClass::I32,
            _ => panic!("Unexpected SSA-value type: {}", ty),
        }
    }

    fn gen_jump(blockindex: BlockIndex) -> Inst {
        unimplemented!()
    }

    fn with_block_rewrites(&mut self, block_target_map: &[BlockIndex]) {
        match self {
            _ => {}
        }
    }

    fn with_fallthrough_block(&mut self, fallthrough: Option<BlockIndex>) {
        match self {
            _ => {}
        }
    }

    fn with_block_offsets(&mut self, my_offset: CodeOffset, targets: &[CodeOffset]) {
        match self {
            _ => {}
        }
    }

    fn reg_universe() -> RealRegUniverse {
        create_reg_universe()
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
                ALUOp::Sub => "sub",
                ALUOp::Sbc => "sbc",
                ALUOp::Rsb => "rsb",
                ALUOp::Mul => "mul",
                ALUOp::And => "and",
                ALUOp::Orr => "orr",
                ALUOp::Eor => "eor",
                ALUOp::Mvn => "mvn",
                ALUOp::Bic => "bic",
                ALUOp::Lsl => "lsl",
                ALUOp::Lsr => "lsr",
                ALUOp::Asr => "asr",
                ALUOp::Ror => "ror",
                ALUOp::Cmp => "cmp",
                ALUOp::Cmn => "cmn",
                ALUOp::Tst => "tst",
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
            &Inst::AluRR { alu_op, rd, rm } => {
                let op = op_name(alu_op);
                let rd = rd.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                if (alu_op == ALUOp::Rsb) {
                    format!("{} {}, {}, #0", op, rd, rm)
                } else {
                    format!("{} {}, {}", op, rd, rm)
                }
            }
            &Inst::AluRRNoResult { alu_op, rn, rm } => {
                let op = op_name(alu_op);
                let rn = rn.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                format!("{} {}, {}", op, rn, rm)
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
            &Inst::AluRImm8NoResult { alu_op, rn, imm8 } => {
                let op = op_name(alu_op);
                let rn = rn.show_rru(mb_rru);
                format!("{} {}, #{}", op, rn, imm8)
            }
            &Inst::MovRR { rd, rm } => {
                let rd = rd.show_rru(mb_rru);
                let rm = rm.show_rru(mb_rru);
                format!("mov {}, {}", rd, rm)
            }
            &Inst::MovRImm8 { rd, imm8 } => {
                let rd = rd.show_rru(mb_rru);
                format!("mov {}, #{}", rd, imm8)
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
                    (32, None) => "ldr",
                    (16, Some(true)) => "ldrsh",
                    (16, Some(false)) => "ldrh",
                    (8, Some(true)) => "ldrsb",
                    (8, Some(false)) => "ldrb",
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
            &Inst::Bkpt => "bkpt #0".to_string(),
            &Inst::Udf { .. } => "udf".to_string(),
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
        }
    }
}
