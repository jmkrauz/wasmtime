//! 32-bit ARM ISA: binary code emission.

use crate::binemit::CodeOffset;
use crate::isa::arm32::inst::*;

use core::convert::TryFrom;

/// Memory addressing mode finalization: convert "special" modes (e.g.,
/// generic arbitrary stack offset) into real addressing modes, possibly by
/// emitting some helper instructions that come immediately before the use
/// of this amode.
#[allow(unused)]
pub fn mem_finalize(insn_off: CodeOffset, mem: &MemArg) -> (Vec<Inst>, MemArg) {
    unimplemented!()
}

//=============================================================================
// Instructions and subcomponents: emission

fn machreg_to_gpr(m: Reg) -> u16 {
    assert!(m.get_class() == RegClass::I32);
    u16::try_from(m.to_real_reg().get_hw_encoding()).unwrap()
}

fn machreg_to_gpr_lo(m: Reg) -> u16 {
    let gpr_lo = machreg_to_gpr(m);
    assert!(gpr_lo < 8);
    gpr_lo
}

fn enc_rrr_lo(bits_15_9: u16, rd: Reg, rn: Reg, rm: Reg) -> u16 {
    (bits_15_9 << 9)
        | machreg_to_gpr_lo(rd)
        | (machreg_to_gpr_lo(rn) << 3)
        | (machreg_to_gpr_lo(rm) << 6)
}

fn enc_rr_lo(bits_15_6: u16, rd: Reg, rm: Reg) -> u16 {
    (bits_15_6 << 6) | machreg_to_gpr_lo(rd) | (machreg_to_gpr_lo(rm) << 3)
}

fn enc_rr(bits_15_8: u16, rd: Reg, rm: Reg) -> u16 {
    let rd = machreg_to_gpr(rd);
    (bits_15_8 << 8) | (rd & 0x7) | ((rd & 0x8) << 7) | (machreg_to_gpr(rm) << 3)
}

fn enc_rr_imm5_lo(bits_15_11: u16, rd: Reg, rm: Reg, imm5: u8) -> u16 {
    assert!(imm5 < 32);
    (bits_15_11 << 11) | machreg_to_gpr_lo(rd) | (machreg_to_gpr_lo(rm) << 3) | u16::from(imm5) << 6
}

fn enc_r_imm8_lo(bits_15_11: u16, rd: Reg, imm8: u8) -> u16 {
    (bits_15_11 << 11) | u16::from(imm8) | (machreg_to_gpr_lo(rd) << 8)
}

fn enc_mov(rd: Writable<Reg>, rm: Reg) -> u16 {
    enc_rr(0b01000110, rd.to_reg(), rm)
}

impl<O: MachSectionOutput> MachInstEmit<O> for Inst {
    fn emit(&self, sink: &mut O) {
        match self {
            &Inst::Nop0 | &Inst::EpiloguePlaceholder => {}
            &Inst::AluRRR { alu_op, rd, rn, rm } => {
                let bits_15_9 = match alu_op {
                    ALUOp::Add => 0b0001100,
                    ALUOp::Sub => 0b0001101,
                    _ => panic!("Invalid ALUOp {:?} in RRR form!", alu_op),
                };
                sink.put2(enc_rrr_lo(bits_15_9, rd.to_reg(), rn, rm));
            }
            &Inst::AluRR { alu_op, rd, rm } => {
                if (alu_op == ALUOp::Add) {
                    sink.put2(enc_rr(0b01000100, rd.to_reg(), rm));
                    return;
                }
                let bits_15_6 = match alu_op {
                    ALUOp::Adc => 0b01000001_01,
                    ALUOp::Sbc => 0b01000001_10,
                    ALUOp::Rsb => 0b01000010_01,
                    ALUOp::Mul => 0b01000011_01,
                    ALUOp::And => 0b01000000_00,
                    ALUOp::Orr => 0b01000011_00,
                    ALUOp::Eor => 0b01000000_01,
                    ALUOp::Mvn => 0b01000011_11,
                    ALUOp::Bic => 0b01000011_10,
                    ALUOp::Lsl => 0b01000000_10,
                    ALUOp::Lsr => 0b01000000_11,
                    ALUOp::Asr => 0b01000001_00,
                    ALUOp::Ror => 0b01000001_11,
                    _ => panic!("Invalid ALUOp {:?} in RR form!", alu_op),
                };
                sink.put2(enc_rr_lo(bits_15_6, rd.to_reg(), rm));
            }
            &Inst::AluRRNoResult { alu_op, rn, rm } => {
                let bits_15_6 = match alu_op {
                    ALUOp::Cmp => 0b01000010_10,
                    ALUOp::Cmn => 0b01000010_11,
                    ALUOp::Tst => 0b01000010_00,
                    _ => panic!("Invalid ALUOp {:?} in RR no result form!", alu_op),
                };
                sink.put2(enc_rr_lo(bits_15_6, rn, rm));
            }
            &Inst::AluRRImm5 {
                alu_op,
                rd,
                rm,
                imm5,
            } => {
                let bits_15_11 = match alu_op {
                    ALUOp::Lsl => 0b00000,
                    ALUOp::Lsr => 0b00001,
                    ALUOp::Asr => 0b00010,
                    _ => panic!("Invalid ALUOp {:?} in RRImm5 form!", alu_op),
                };
                sink.put2(enc_rr_imm5_lo(bits_15_11, rd.to_reg(), rm, imm5));
            }
            &Inst::AluRImm8 { alu_op, rd, imm8 } => {
                let bits_15_11 = match alu_op {
                    ALUOp::Add => 0b00110,
                    ALUOp::Sub => 0b00111,
                    _ => panic!("Invalid ALUOp {:?} in RImm8 form!", alu_op),
                };
                sink.put2(enc_r_imm8_lo(bits_15_11, rd.to_reg(), imm8));
            }
            &Inst::AluRImm8NoResult { alu_op, rn, imm8 } => {
                let bits_15_11 = match alu_op {
                    ALUOp::Cmp => 0b00101,
                    _ => panic!("Invalid ALUOp {:?} in RImm8 no result form!", alu_op),
                };
                sink.put2(enc_r_imm8_lo(bits_15_11, rn, imm8));
            }
            &Inst::MovRR { rd, rm } => {
                sink.put2(enc_mov(rd, rm));
            }
            &Inst::MovRImm8 { rd, imm8 } => {
                sink.put2(enc_r_imm8_lo(0b00100, rd.to_reg(), imm8));
            }
            &Inst::Store {
                rt,
                ref mem,
                srcloc,
                bits,
            } => {
                if let Some(srcloc) = srcloc {
                    // Register the offset at which the store instruction starts.
                    sink.add_trap(srcloc, TrapCode::OutOfBounds);
                }
                match mem {
                    &MemArg::RegReg(rn, rm) => {
                        let bits_15_9 = match bits {
                            32 => 0b0101000,
                            16 => 0b0101001,
                            8 => 0b0101010,
                            _ => panic!("Unsupported Store case: {:?}", self),
                        };
                        sink.put2(enc_rrr_lo(bits_15_9, rt, rn, rm));
                    }
                    &MemArg::Offset5(rn, imm5) => {
                        let bits_15_11 = match bits {
                            32 => 0b01100,
                            16 => 0b10000,
                            8 => 0b01110,
                            _ => panic!("Unsupported Store case: {:?}", self),
                        };
                        sink.put2(enc_rr_imm5_lo(bits_15_11, rt, rn, imm5));
                    }
                    &MemArg::SPOffset(imm8) => {
                        let bits_15_11 = match bits {
                            32 => 0b10010,
                            _ => panic!("Unsupported Store case: {:?}", self),
                        };
                        sink.put2(enc_r_imm8_lo(bits_15_11, rt, imm8));
                    }
                }
            }
            &Inst::Load {
                rt,
                ref mem,
                srcloc,
                bits,
                sign_extend,
            } => {
                if let Some(srcloc) = srcloc {
                    // Register the offset at which the load instruction starts.
                    sink.add_trap(srcloc, TrapCode::OutOfBounds);
                }
                match mem {
                    &MemArg::RegReg(rn, rm) => {
                        let bits_15_9 = match (bits, sign_extend) {
                            (32, None) => 0b0101100,
                            (16, Some(true)) => 0b0101111,
                            (16, Some(false)) => 0b0101101,
                            (8, Some(true)) => 0b0101011,
                            (8, Some(false)) => 0b0101110,
                            _ => panic!("Unsupported Load case: {:?}", self),
                        };
                        sink.put2(enc_rrr_lo(bits_15_9, rt.to_reg(), rn, rm));
                    }
                    &MemArg::Offset5(rn, imm5) => {
                        let bits_15_11 = match (bits, sign_extend) {
                            (32, None) => 0b01101,
                            (16, Some(false)) => 0b10001,
                            (8, Some(false)) => 0b01111,
                            _ => panic!("Unsupported Load case: {:?}", self),
                        };
                        sink.put2(enc_rr_imm5_lo(bits_15_11, rt.to_reg(), rn, imm5));
                    }
                    &MemArg::SPOffset(imm8) => {
                        let bits_15_11 = match (bits, sign_extend) {
                            (32, None) => 0b10011,
                            _ => panic!("Unsupported Load case: {:?}", self),
                        };
                        sink.put2(enc_r_imm8_lo(bits_15_11, rt.to_reg(), imm8));
                    }
                }
            }
            &Inst::Extend {
                rd,
                rm,
                from_bits,
                signed,
            } => {
                let bits_15_6 = match (from_bits, signed) {
                    (16, true) => 0b1011001000,
                    (16, false) => 0b1011001010,
                    (8, true) => 0b1011001001,
                    (8, false) => 0b1011001011,
                    _ => panic!("Unsupported Extend case: {:?}", self),
                };
                sink.put2(enc_rr_lo(bits_15_6, rd.to_reg(), rm));
            }
            &Inst::Bkpt => {
                sink.put2(0b10111110_00000000);
            }
            &Inst::Udf { trap_info } => {
                let (srcloc, code) = trap_info;
                sink.add_trap(srcloc, code);
                sink.put2(0b11011110_00000000);
            }
            &Inst::CallInd {
                rm, loc, opcode, ..
            } => {
                sink.put2(0b01000111_1_0000_000 | (machreg_to_gpr(rm) << 3));
                if opcode.is_call() {
                    sink.add_call_site(loc, opcode);
                }
            }
            &Inst::Ret => {
                sink.put2(enc_mov(writable_pc_reg(), lr_reg()));
            }
            &Inst::Jump { ref dest } => {
                let off11 = if let Some(off) = dest.as_off11() {
                    off
                } else {
                    unimplemented!()
                };
                sink.put2(0b11100_00000000000 | off11);
            }
            &Inst::CondBr { .. } => unimplemented!(),
        }
    }
}
