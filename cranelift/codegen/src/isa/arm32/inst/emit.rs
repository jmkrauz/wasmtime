//! 32-bit ARM ISA: binary code emission.

use crate::binemit::{CodeOffset, Reloc};
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

fn machreg_is_lo(m: Reg) -> bool {
    return machreg_to_gpr(m) < 8;
}

fn enc_16_rrr(bits_15_9: u16, rd: Reg, rn: Reg, rm: Reg) -> u16 {
    (bits_15_9 << 9)
        | machreg_to_gpr_lo(rd)
        | (machreg_to_gpr_lo(rn) << 3)
        | (machreg_to_gpr_lo(rm) << 6)
}

fn enc_16_rr(bits_15_6: u16, rd: Reg, rm: Reg) -> u16 {
    (bits_15_6 << 6) | machreg_to_gpr_lo(rd) | (machreg_to_gpr_lo(rm) << 3)
}

fn enc_16_rr_any(bits_15_8: u16, rd: Reg, rm: Reg) -> u16 {
    let rd = machreg_to_gpr(rd);
    (bits_15_8 << 8) | (rd & 0x7) | ((rd >> 3) << 7) | (machreg_to_gpr(rm) << 3)
}

fn enc_16_rr_imm5(bits_15_11: u16, rd: Reg, rm: Reg, imm5: u8) -> u16 {
    assert!(imm5 < 32);
    (bits_15_11 << 11) | machreg_to_gpr_lo(rd) | (machreg_to_gpr_lo(rm) << 3) | u16::from(imm5) << 6
}

fn enc_16_r_imm8(bits_15_11: u16, rd: Reg, imm8: u8) -> u16 {
    (bits_15_11 << 11) | u16::from(imm8) | (machreg_to_gpr_lo(rd) << 8)
}

fn enc_16_mov(rd: Writable<Reg>, rm: Reg) -> u16 {
    enc_16_rr_any(0b01000110, rd.to_reg(), rm)
}

fn enc_16_jump11(off11: u16) -> u16 {
    0b11100_00000000000 | off11
}

fn enc_16_cond_branch8(cond: Cond, off8: u16) -> u16 {
    0b1101_0000_00000000 | (cond.bits() << 8) | off8
}

fn enc_16_comp_branch6(rn: Reg, zero: bool, off6: u16) -> u16 {
    let op = if zero { 0b0 } else { 0b1 };

    0b1011_0_0_0_1_00000_000
        | machreg_to_gpr_lo(rn)
        | ((off6 & 0x1f) << 3)
        | ((off6 >> 5) << 8)
        | (op << 11)
}

fn enc_16_it(cond: Cond, te1: Option<bool>, te2: Option<bool>, te3: Option<bool>) -> u16 {
    fn te_bit(cond: u16, te: Option<bool>) -> u16 {
        match te {
            None => 0,
            Some(true) => cond & 0x1,
            Some(false) => (cond & 0x1) ^ 0x1,
        }
    }

    let cond = cond.bits();
    let last_one = match (te1, te2, te3) {
        (None, None, None) => 0b1000,
        (Some(_), None, None) => 0b0100,
        (Some(_), Some(_), None) => 0b0010,
        (Some(_), Some(_), Some(_)) => 0b0001,
        _ => panic!(
            "Invalid condition combination {:?} {:?} {:?} in it instruction",
            te1, te2, te3
        ),
    };

    let mask =
        last_one | (te_bit(cond, te1) << 3) | (te_bit(cond, te2) << 2) | (te_bit(cond, te1) << 1);
    0b1011_1111_0000_0000 | (cond << 4) | mask
}

fn enc_32_regs(
    mut inst: u32,
    reg_0: Option<Reg>,
    reg_8: Option<Reg>,
    reg_12: Option<Reg>,
    reg_16: Option<Reg>,
) -> u32 {
    if let Some(reg_0) = reg_0 {
        inst |= u32::from(machreg_to_gpr(reg_0));
    }
    if let Some(reg_8) = reg_8 {
        inst |= u32::from(machreg_to_gpr(reg_8)) << 8;
    }
    if let Some(reg_12) = reg_12 {
        inst |= u32::from(machreg_to_gpr(reg_12)) << 12;
    }
    if let Some(reg_16) = reg_16 {
        inst |= u32::from(machreg_to_gpr(reg_16)) << 16;
    }
    inst
}

fn enc_32_reg_shift(inst: u32, shift: &Option<ShiftOpAndAmt>) -> u32 {
    match shift {
        Some(shift) => {
            let op = u32::from(shift.op().bits());
            let amt = u32::from(shift.amt().value());
            let imm2 = amt & 0x3;
            let imm3 = (amt >> 2) & 0x7;

            inst | (op << 4) | (imm2 << 6) | (imm3 << 12)
        }
        None => inst,
    }
}

fn enc_32_r_imm16(bits_31_20: u32, rd: Reg, imm16: u16) -> u32 {
    let imm16 = u32::from(imm16);
    let imm8 = imm16 & 0xff;
    let imm3 = (imm16 >> 8) & 0x7;
    let i = (imm16 >> 11) & 0x1;
    let imm4 = (imm16 >> 12) & 0xf;

    let inst = ((bits_31_20 << 20) & !(1 << 26)) | imm8 | (imm3 << 12) | (imm4 << 16) | (i << 26);
    enc_32_regs(inst, None, Some(rd), None, None)
}

fn enc_32_rrr(bits_31_20: u32, bits_15_12: u32, bits_7_4: u32, rd: Reg, rm: Reg, rn: Reg) -> u32 {
    let inst = (bits_31_20 << 20) | (bits_15_12 << 12) | (bits_7_4 << 4);
    enc_32_regs(inst, Some(rm), Some(rd), None, Some(rn))
}

fn enc_32_imm12(inst: u32, imm12: u16) -> u32 {
    let imm12 = u32::from(imm12);
    let imm8 = imm12 & 0xff;
    let imm3 = (imm12 >> 8) & 0x7;
    let i = (imm12 >> 11) & 0x1;
    inst | imm8 | (imm3 << 12) | (i << 26)
}

fn enc_32_mem_r(bits_24_20: u32, rt: Reg, rn: Reg, rm: Reg, imm2: u32) -> u32 {
    let inst = (imm2 << 4) | (bits_24_20 << 20) | (0b11111 << 27);
    enc_32_regs(inst, Some(rm), None, Some(rt), Some(rn))
}

fn enc_32_mem_off12(bits_24_20: u32, rt: Reg, rn: Reg, off12: u32) -> u32 {
    let inst = off12 | (bits_24_20 << 20) | (0b11111 << 27);
    enc_32_regs(inst, None, None, Some(rt), Some(rn))
}

fn enc_32_jump24(off24: u32) -> u32 {
    let imm11 = off24 & 0x7ff;
    let imm10 = (off24 >> 11) & 0x3ff;
    let i2 = (off24 >> 21) & 0x1;
    let i1 = (off24 >> 22) & 0x1;
    let s = (off24 >> 23) & 0x1;
    let j1 = (i1 ^ s) ^ 1;
    let j2 = (i2 ^ s) ^ 1;

    0b11110_0_0000000000_10_0_1_0_00000000000
        | imm11
        | (j2 << 11)
        | (j1 << 13)
        | (imm10 << 16)
        | (s << 26)
}

fn enc_32_cond_branch20(cond: Cond, off20: u32) -> u32 {
    let cond = u32::from(cond.bits());
    let imm11 = off20 & 0x7ff;
    let imm6 = (off20 >> 11) & 0x3f;
    let j1 = (off20 >> 17) & 0x1;
    let j2 = (off20 >> 18) & 0x1;
    let s = (off20 >> 19) & 0x1;

    0b11110_0_0000_000000_10_0_0_0_00000000000
        | imm11
        | (j2 << 11)
        | (j1 << 13)
        | (imm6 << 16)
        | (cond << 22)
        | (s << 26)
}

fn enc_vfp_reg(v: Reg, precision: Precision) -> (u32, u32) {
    assert!(v.get_class() == RegClass::F64);
    let reg: u32 = v.to_real_reg().get_hw_encoding() as u32;

    match precision {
        Precision::Single => {
            let reg = reg * 2;
            ((reg >> 1) & 0x7, reg & 0x1)
        }
        Precision::Double => (reg & 0x7, (reg >> 3) & 0x1),
    }
}

fn enc_vfp_regs(
    mut inst: u32,
    vn: Option<Reg>,
    vd: Option<Reg>,
    vm: Option<Reg>,
    precision: Precision,
) -> u32 {
    if precision == Precision::Double {
        inst |= 1 << 8;
    }
    if let Some(vn) = vn {
        let (vn, n) = enc_vfp_reg(vn, precision);
        inst |= vn << 16;
        inst |= n << 7;
    }
    if let Some(vd) = vd {
        let (vd, d) = enc_vfp_reg(vd, precision);
        inst |= vd << 12;
        inst |= d << 22;
    }
    if let Some(vm) = vm {
        let (vm, m) = enc_vfp_reg(vm, precision);
        inst |= vm;
        inst |= m << 5;
    }
    inst
}

fn u32_swap_halfwords(x: u32) -> u32 {
    (x >> 16) | (x << 16)
}

fn emit_32(inst: u32, sink: &mut MachBuffer<Inst>) {
    let inst_hi = (inst >> 16) as u16;
    let inst_lo = (inst & 0xffff) as u16;
    sink.put2(inst_hi);
    sink.put2(inst_lo);
}

/// State carried between emissions of a sequence of instructions.
#[derive(Default, Clone, Debug)]
pub struct EmitState {
    virtual_sp_offset: i64,
}

impl MachInstEmit for Inst {
    type State = EmitState;

    fn emit(&self, sink: &mut MachBuffer<Inst>, _flags: &settings::Flags, _state: &mut EmitState) {
        let start_off = sink.cur_offset();

        match self {
            &Inst::Nop0 | &Inst::EpiloguePlaceholder => {}
            &Inst::AluRRR { alu_op, rd, rn, rm } => {
                let (bits_31_20, bits_15_12, bits_7_4) = match alu_op {
                    ALUOp::Lsl => (0b111110100000, 0b1111, 0b0000),
                    ALUOp::Lsr => (0b111110100010, 0b1111, 0b0000),
                    ALUOp::Asr => (0b111110100100, 0b1111, 0b0000),
                    ALUOp::Ror => (0b111110100110, 0b1111, 0b0000),
                    ALUOp::Qadd => (0b111110101000, 0b1111, 0b1000),
                    ALUOp::Qsub => (0b111110101000, 0b1111, 0b1010),
                    ALUOp::Mul => (0b111110110000, 0b1111, 0b0000),
                    ALUOp::Udiv => (0b111110111011, 0b1111, 0b1111),
                    ALUOp::Sdiv => (0b11111011101, 0b1111, 0b1111),
                    _ => panic!("Invalid ALUOp {:?} in RRR form!", alu_op),
                };
                emit_32(
                    enc_32_rrr(bits_31_20, bits_15_12, bits_7_4, rd.to_reg(), rm, rn),
                    sink,
                );
            }
            &Inst::AluRRRShift {
                alu_op,
                rd,
                rn,
                rm,
                ref shift,
            } => {
                let bits_31_24 = 0b111_0101;
                let bits_24_20 = match alu_op {
                    ALUOp::And => 0b00000,
                    ALUOp::Bic => 0b00010,
                    ALUOp::Orr => 0b00100,
                    ALUOp::Orn => 0b00110,
                    ALUOp::Eor => 0b01000,
                    ALUOp::Add => 0b10000,
                    ALUOp::Adds => 0b10001,
                    ALUOp::Adc => 0b10100,
                    ALUOp::Adcs => 0b10101,
                    ALUOp::Sbc => 0b10110,
                    ALUOp::Sbcs => 0b10111,
                    ALUOp::Sub => 0b11010,
                    ALUOp::Subs => 0b11011,
                    ALUOp::Rsb => 0b11100,
                    _ => panic!("Invalid ALUOp {:?} in RRRShift form!", alu_op),
                };
                let bits_31_20 = (bits_31_24 << 5) | bits_24_20;
                let inst = enc_32_rrr(bits_31_20, 0, 0, rd.to_reg(), rm, rn);
                let inst = enc_32_reg_shift(inst, shift);
                emit_32(inst, sink);
            }
            &Inst::AluRR { alu_op, rd, rm } => match alu_op {
                ALUOp::Mvn => {
                    let inst = 0b11101010011011110000_0000_0000_0000;
                    let inst = enc_32_regs(inst, Some(rm), Some(rd.to_reg()), None, None);
                    emit_32(inst, sink);
                }
                _ => panic!("Invalid ALUOp {:?} in RR form!", alu_op),
            },
            &Inst::AluRRRR {
                alu_op,
                rd_hi,
                rd_lo,
                rn,
                rm,
            } => {
                let (bits_22_20, bits_7_4) = match alu_op {
                    ALUOp::Smull => (0b000, 0b0000),
                    ALUOp::Umull => (0b010, 0b0000),
                    _ => panic!("Invalid ALUOp {:?} in RRRR form!", alu_op),
                };
                let inst = (0b111110111 << 23) | (bits_22_20 << 20) | (bits_7_4 << 4);
                let inst = enc_32_regs(
                    inst,
                    Some(rm),
                    Some(rd_hi.to_reg()),
                    Some(rd_lo.to_reg()),
                    Some(rn),
                );
                emit_32(inst, sink);
            }
            &Inst::AluRRImm12 {
                alu_op,
                rd,
                rn,
                imm12,
            } => {
                let bits_24_20 = match alu_op {
                    ALUOp::Add => 0b00000,
                    ALUOp::Sub => 0b01010,
                    _ => panic!("Invalid ALUOp {:?} in RRImm12 form!", alu_op),
                };
                let inst = (0b11110_0_1 << 25) | (bits_24_20 << 20);
                let inst = enc_32_regs(inst, None, Some(rd.to_reg()), None, Some(rn));
                let inst = enc_32_imm12(inst, imm12);
                emit_32(inst, sink);
            }
            &Inst::AluRRImm8 {
                alu_op,
                rd,
                rn,
                imm8,
            } => {
                let bits_24_20 = match alu_op {
                    ALUOp::And => 0b00000,
                    ALUOp::Bic => 0b00010,
                    ALUOp::Orr => 0b00100,
                    ALUOp::Orn => 0b00110,
                    ALUOp::Eor => 0b01000,
                    ALUOp::Add => 0b10000,
                    ALUOp::Adc => 0b10100,
                    ALUOp::Sbc => 0b10110,
                    ALUOp::Sub => 0b11010,
                    ALUOp::Rsb => 0b11100,
                    _ => panic!("Invalid ALUOp {:?} in RRImm8 form!", alu_op),
                };
                let imm8 = u32::from(imm8);
                let inst = 0b11110_0_0_00000_0000_0_000_0000_00000000 | imm8 | (bits_24_20 << 20);
                let inst = enc_32_regs(inst, None, Some(rd.to_reg()), None, Some(rn));
                emit_32(inst, sink);
            }
            &Inst::AluRImm8 { alu_op, rd, imm8 } => {
                let bits_24_20 = match alu_op {
                    ALUOp::Mvn => 0b00110,
                    _ => panic!("Invalid ALUOp {:?} in RImm8 form!", alu_op),
                };
                let imm8 = u32::from(imm8);
                let inst = 0b11110_0_0_00000_1111_0_000_0000_00000000 | imm8 | (bits_24_20 << 20);
                let inst = enc_32_regs(inst, None, Some(rd.to_reg()), None, None);
                emit_32(inst, sink);
            }
            &Inst::FpuRR {
                fpu_op,
                vd,
                vm,
                precision,
            } => {
                let (bits_20_16, bits_7_6) = match fpu_op {
                    FPUOp1::Vabs => (0b0000, 0b11),
                    FPUOp1::Vneg => (0b0001, 0b01),
                    FPUOp1::Vsqrt => (0b0001, 0b11),
                };
                let inst = 0b111011101_0_11_0000_0000_101_0_00_0_0_0000
                    | (bits_20_16 << 16)
                    | (bits_7_6 << 6);
                let inst = enc_vfp_regs(inst, None, Some(vd.to_reg()), Some(vm), precision);
                emit_32(inst, sink);
            }
            &Inst::FpuRRR {
                fpu_op,
                vd,
                vn,
                vm,
                precision,
            } => {
                let (bits_21_20, bit_6) = match fpu_op {
                    FPUOp2::Vadd => (0b11, 0b0),
                    FPUOp2::Vsub => (0b11, 0b1),
                    FPUOp2::Vmul => (0b10, 0b0),
                    FPUOp2::Vdiv => (0b00, 0b0),
                };
                let inst = 0b111011100_0_00_0000_0000_101_0_0_0_0_0_0000
                    | (bits_21_20 << 20)
                    | (bit_6 << 6);
                let inst = enc_vfp_regs(inst, Some(vn), Some(vd.to_reg()), Some(vm), precision);
                emit_32(inst, sink);
            }
            &Inst::FpuRRROp3 {
                fpu_op,
                vd,
                vn,
                vm,
                precision,
            } => {
                let bit_6 = match fpu_op {
                    FPUOp3::Vfma => 0b0,
                    FPUOp3::Vfms => 0b1,
                };
                let inst = 0b111011101_0_10_0000_0000_101_0_0_0_0_0_0000 | (bit_6 << 6);
                let inst = enc_vfp_regs(inst, Some(vn), Some(vd.to_reg()), Some(vm), precision);
                emit_32(inst, sink);
            }
            &Inst::BitOpRR { bit_op, rd, rm } => {
                let (bits_22_20, bits_7_4) = match bit_op {
                    BitOp::Rbit => (0b001, 0b1010),
                    BitOp::Rev => (0b001, 0b1000),
                    BitOp::Clz => (0b011, 0b1000),
                };
                let inst = 0b111110101_000_0000_1111_0000_0000_0000 | (bits_22_20 << 20) | (bits_7_4 << 4);
                let inst = enc_32_regs(inst, Some(rm), Some(rd.to_reg()), None, Some(rm));
                emit_32(inst, sink);
            }
            &Inst::Mov { rd, rm } => {
                sink.put2(enc_16_mov(rd, rm));
            }
            &Inst::MovImm16 { rd, imm16 } => {
                emit_32(enc_32_r_imm16(0b11110_0_100100, rd.to_reg(), imm16), sink);
            }
            &Inst::Movt { rd, imm16 } => {
                emit_32(enc_32_r_imm16(0b11110_0_101100, rd.to_reg(), imm16), sink);
            }
            &Inst::Vmov { vd, vm, precision } => {
                let inst = 0b111011101_0_11_0000_0000_101_0_01_0_0_0000;
                let inst = enc_vfp_regs(inst, None, Some(vd.to_reg()), Some(vm), precision);
                emit_32(inst, sink);
            }
            &Inst::MoveGprToFpu { vn, rt, lo } => {
                let inst = 0b11101110000_0_0000_0000_1010_0_0010000;
                let inst = enc_32_regs(inst, None, None, Some(rt), None);
                let mut inst = enc_vfp_regs(inst, Some(vn.to_reg()), None, None, Precision::Single);
                if !lo {
                    inst |= 1 << 7;
                }
                emit_32(inst, sink);
            }
            &Inst::MoveFpuToGpr { rt, vn, lo } => {
                let inst = 0b11101110000_1_0000_0000_1010_0_0010000;
                let inst = enc_32_regs(inst, None, None, Some(rt.to_reg()), None);
                let mut inst = enc_vfp_regs(inst, Some(vn), None, None, Precision::Single);
                if !lo {
                    inst |= 1 << 7;
                }
                emit_32(inst, sink);
            }
            &Inst::Cmp { rn, rm } => {
                if machreg_is_lo(rn) && machreg_is_lo(rm) {
                    sink.put2(enc_16_rr(0b0100001010, rn, rm));
                } else {
                    sink.put2(enc_16_rr_any(0b01000101, rn, rm));
                }
            }
            &Inst::CmpImm8 { rn, imm8 } => {
                let inst = 0b11110_0_011011_0000_0_000_1111_00000000 | u32::from(imm8);
                let inst = enc_32_regs(inst, None, None, None, Some(rn));
                emit_32(inst, sink);
            }
            &Inst::Vcmp { vd, vm, precision } => {
                let inst = 0b111011101_0_110100_0000_101_0_0_1_0_0_0000;
                let inst = enc_vfp_regs(inst, None, Some(vd), Some(vm), precision);
                emit_32(inst, sink);
            }
            &Inst::VcmpZero { vd, precision } => {
                let inst = 0b111011101_0_110101_0000_101_0_0_1_000000;
                let inst = enc_vfp_regs(inst, None, Some(vd), None, precision);
                emit_32(inst, sink);
            }
            &Inst::Store {
                rt,
                ref mem,
                srcloc,
                bits,
            } => {
                if let Some(srcloc) = srcloc {
                    // Register the offset at which the store instruction starts.
                    sink.add_trap(srcloc, TrapCode::HeapOutOfBounds);
                }
                match mem {
                    &MemArg::RegReg(rn, rm, imm2) => {
                        let bits_24_20 = match bits {
                            32 => 0b00100,
                            16 => 0b00010,
                            8 => 0b00000,
                            _ => panic!("Unsupported Store case {:?}", self),
                        };
                        emit_32(enc_32_mem_r(bits_24_20, rt, rn, rm, imm2), sink);
                    }
                    &MemArg::Offset12(rn, off12) => {
                        let bits_24_20 = match bits {
                            32 => 0b01100,
                            16 => 0b01010,
                            8 => 0b01000,
                            _ => panic!("Unsupported Store case {:?}", self),
                        };
                        emit_32(enc_32_mem_off12(bits_24_20, rt, rn, off12), sink);
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
                    sink.add_trap(srcloc, TrapCode::HeapOutOfBounds);
                }
                match mem {
                    &MemArg::RegReg(rn, rm, imm2) => {
                        let bits_24_20 = match (bits, sign_extend) {
                            (32, _) => 0b00101,
                            (16, true) => 0b10011,
                            (16, false) => 0b00011,
                            (8, true) => 0b10001,
                            (8, false) => 0b00001,
                            _ => panic!("Unsupported Load case: {:?}", self),
                        };
                        emit_32(enc_32_mem_r(bits_24_20, rt.to_reg(), rn, rm, imm2), sink);
                    }
                    &MemArg::Offset12(rn, off12) => {
                        let bits_24_20 = match (bits, sign_extend) {
                            (32, _) => 0b01101,
                            (16, true) => 0b11011,
                            (16, false) => 0b01011,
                            (8, true) => 0b11001,
                            (8, false) => 0b01001,
                            _ => panic!("Unsupported Load case: {:?}", self),
                        };
                        emit_32(enc_32_mem_off12(bits_24_20, rt.to_reg(), rn, off12), sink);
                    }
                }
            }
            &Inst::Extend {
                rd,
                rm,
                from_bits,
                signed,
            } => {
                let rd = rd.to_reg();
                if machreg_is_lo(rd) && machreg_is_lo(rm) {
                    let bits_15_9 = match (from_bits, signed) {
                        (16, true) => 0b1011001000,
                        (16, false) => 0b1011001010,
                        (8, true) => 0b1011001001,
                        (8, false) => 0b1011001011,
                        _ => panic!("Unsupported Extend case: {:?}", self),
                    };
                    sink.put2(enc_16_rr(bits_15_9, rd, rm));
                } else {
                    let bits_22_20 = match (from_bits, signed) {
                        (16, true) => 0b000,
                        (16, false) => 0b001,
                        (8, true) => 0b100,
                        (8, false) => 0b101,
                        _ => panic!("Unsupported Extend case: {:?}", self),
                    };
                    let inst = 0b111110100_000_11111111_0000_1000_0000 | (bits_22_20 << 20);
                    emit_32(enc_32_regs(inst, Some(rm), Some(rd), None, None), sink);
                }
            }
            &Inst::It {
                cond,
                te1,
                te2,
                te3,
            } => sink.put2(enc_16_it(cond, te1, te2, te3)),
            &Inst::Bkpt => {
                sink.put2(0b10111110_00000000);
            }
            &Inst::Udf { trap_info } => {
                let (srcloc, code) = trap_info;
                sink.add_trap(srcloc, code);
                sink.put2(0b11011110_00000000);
            }
            &Inst::Push { ref reg_list } => match reg_list.len() {
                0 => panic!("Unsupported Push case: {:?}", self),
                1 => {
                    let reg = u32::from(machreg_to_gpr(reg_list[0]));
                    let inst: u32 = 0b1111100001001101_0000_110100000100 | (reg << 12);
                    emit_32(inst, sink);
                }
                _ => {
                    let mut inst: u32 = 0b1110100100101101 << 16;
                    for reg in reg_list {
                        inst |= 1 << machreg_to_gpr(*reg);
                    }
                    if inst & ((1 << 13) | (1 << 15)) != 0 {
                        panic!("Unsupported Push case: {:?}", self);
                    }
                    emit_32(inst, sink);
                }
            },
            &Inst::Pop { ref reg_list } => match reg_list.len() {
                0 => panic!("Unsupported Pop case: {:?}", self),
                1 => {
                    let reg = u32::from(machreg_to_gpr(reg_list[0].to_reg()));
                    let inst: u32 = 0b1111100001011101_0000_101100000100 | (reg << 12);
                    emit_32(inst, sink);
                }
                _ => {
                    let mut inst: u32 = 0b1110100010111101 << 16;
                    for reg in reg_list {
                        inst |= 1 << machreg_to_gpr(reg.to_reg());
                    }
                    if (inst & (1 << 14) != 0) && (inst & (1 << 15) != 0) {
                        panic!("Unsupported Pop case: {:?}", self);
                    }
                    emit_32(inst, sink);
                }
            },
            &Inst::Call {
                ref dest,
                loc,
                opcode,
                ..
            } => {
                sink.add_reloc(loc, Reloc::Arm32Call, dest, 0);
                sink.put4(0b11110_0_0000000000_11_0_1_0_00000000000);
                if opcode.is_call() {
                    sink.add_call_site(loc, opcode);
                }
            }
            &Inst::CallInd {
                rm, loc, opcode, ..
            } => {
                sink.put2(0b01000111_1_0000_000 | (machreg_to_gpr(rm) << 3));
                if opcode.is_call() {
                    sink.add_call_site(loc, opcode);
                }
            }
            &Inst::LoadExtName {
                rt,
                ref name,
                offset,
                srcloc,
            } => {
                let inst: u32 = (0b11111000_1_1011111 << 16) | 0x4;
                emit_32(enc_32_regs(inst, None, None, Some(rt.to_reg()), None), sink);
                let padding = if start_off & 0x3 == 0 { 2 } else { 0 };
                // This jump should take 2 bytes.
                sink.put2(enc_16_jump11((padding + 2) >> 1));
                sink.align_to(4);
                sink.add_reloc(srcloc, Reloc::Abs4, name, (offset + 1).into());
                sink.put4(0);
            }
            &Inst::Ret => {
                sink.put2(0b010001110_1110_000); // bx lr
            }
            &Inst::Jump { ref dest } => {
                let off = sink.cur_offset();
                // Indicate that the jump uses a label, if so, so that a fixup can occur later.
                if let Some(l) = dest.as_label() {
                    sink.use_label_at_offset(off, l, LabelUse::Branch24);
                    sink.add_uncond_branch(off, off + 4, l);
                }
                if let Some(off24) = dest.as_off24() {
                    emit_32(enc_32_jump24(off24), sink);
                } else {
                    unimplemented!()
                }
            }
            &Inst::CondBr {
                taken,
                not_taken,
                kind,
            } => {
                // Conditional part first.
                let cond_off = sink.cur_offset();
                if let Some(l) = taken.as_label() {
                    let label_use = if let CondBrKind::Cond(_) = kind {
                        LabelUse::Branch20
                    } else {
                        LabelUse::Branch6
                    };
                    sink.use_label_at_offset(cond_off, l, label_use);
                    match kind {
                        CondBrKind::Zero(reg) => {
                            let inverted =
                                enc_16_comp_branch6(reg, false, taken.as_off6().unwrap())
                                    .to_le_bytes();
                            sink.add_cond_branch(cond_off, cond_off + 2, l, &inverted[..]);
                        }
                        CondBrKind::NotZero(reg) => {
                            let inverted = enc_16_comp_branch6(reg, true, taken.as_off6().unwrap())
                                .to_le_bytes();
                            sink.add_cond_branch(cond_off, cond_off + 2, l, &inverted[..]);
                        }
                        CondBrKind::Cond(c) => {
                            let inverted =
                                enc_32_cond_branch20(c.invert(), taken.as_off20().unwrap());
                            let inverted = u32_swap_halfwords(inverted).to_le_bytes();
                            sink.add_cond_branch(cond_off, cond_off + 4, l, &inverted[..]);
                        }
                    }
                }
                let maybe_off6 = taken.as_off6();
                let maybe_off20 = taken.as_off20();

                match (kind, maybe_off6, maybe_off20) {
                    (CondBrKind::Zero(reg), Some(off), _) => {
                        sink.put2(enc_16_comp_branch6(reg, true, off));
                    }
                    (CondBrKind::NotZero(reg), Some(off), _) => {
                        sink.put2(enc_16_comp_branch6(reg, false, off));
                    }
                    (CondBrKind::Cond(c), _, Some(off)) => {
                        emit_32(enc_32_cond_branch20(c, off), sink);
                    }
                    _ => unimplemented!(),
                }

                // Unconditional part.
                let uncond_off = sink.cur_offset();
                if let Some(l) = not_taken.as_label() {
                    sink.use_label_at_offset(uncond_off, l, LabelUse::Branch24);
                    sink.add_uncond_branch(uncond_off, uncond_off + 4, l);
                }
                emit_32(enc_32_jump24(not_taken.as_off24().unwrap()), sink);
            }
            &Inst::OneWayCondBr { target, kind } => {
                if let Some(_) = target.as_label() {
                    unimplemented!()
                }
                match kind {
                    CondBrKind::Zero(reg) => {
                        sink.put2(enc_16_comp_branch6(reg, true, target.as_off6().unwrap()));
                    }
                    CondBrKind::NotZero(reg) => {
                        sink.put2(enc_16_comp_branch6(reg, false, target.as_off6().unwrap()));
                    }
                    CondBrKind::Cond(c) => {
                        emit_32(enc_32_cond_branch20(c, target.as_off20().unwrap()), sink);
                    }
                }
            }
        }

        let end_off = sink.cur_offset();
        debug_assert!((end_off - start_off) <= Inst::worst_case_size());
    }
}
