//! Emitting binary ARM32 machine code.

use super::registers::RU;
use crate::binemit::{bad_encoding, CodeSink, Reloc};
use crate::ir::condcodes::{CondCode, FloatCC, IntCC};
use crate::ir::{Block, Function, Inst, InstructionData, TrapCode};
use crate::isa::{RegUnit, StackBase, StackBaseMask, StackRef, TargetIsa};
use crate::predicates;
use crate::regalloc::RegDiversions;

include!(concat!(env!("OUT_DIR"), "/binemit-arm32.rs"));

const NULL_REG: RegUnit = 0;

const TEMP_REG: RegUnit = RU::r12 as RegUnit;
const SP_REG: RegUnit = RU::r13 as RegUnit;
const LR_REG: RegUnit = RU::r14 as RegUnit;
const PC_REG: RegUnit = RU::r15 as RegUnit;

// Condition codes
const EQ: u16 = 0x0; // Equal
const NE: u16 = 0x1; // Not equal
const CS: u16 = 0x2; // Unigned higher or same
const CC: u16 = 0x3; // Unsigned lower
const PL: u16 = 0x5; // Positive or zero
const VS: u16 = 0x6; // Overflow
const VC: u16 = 0x7; // No overflow
const HI: u16 = 0x8; // Unsigned higher
const LS: u16 = 0x9; // Unsigned lower or same
const GE: u16 = 0xa; // Greater or equal
const LT: u16 = 0xb; // Less than
const GT: u16 = 0xc; // Greater than
const LE: u16 = 0xd; // Less than or equal
const AL: u16 = 0xe; // Always

// Data processing opcodes shifted left by 4 for convenience.
const AND: u16 = 0x00;
const SUB: u16 = 0x20;
const ADD: u16 = 0x40;
const CMP: u16 = 0xa0;
const MOV: u16 = 0xd0;

// r_i RegUnit number is 64 + i and (64 + i) & 0xf == i for 0 <= i < 16.

/// Data Processing instruction (two registers).
///
///   31       24       19  15   11       3
///   cond 000 opcode S rs1  rd  00000000 rs2
///     28         21    16   12        4   0
///
/// cond = bits[3:0], opcode = bits[7:4], S = bits[8].
/// S = 1 => mofify flags
fn put_dp_rr<CS: CodeSink + ?Sized>(
    bits: u16,
    rs1: RegUnit,
    rs2: RegUnit,
    rd: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let opcode = (bits >> 4) & 0xf;
    let s = (bits >> 8) & 0x1;
    let rs1 = u32::from(rs1) & 0xf;
    let rs2 = u32::from(rs2) & 0xf;
    let rd = u32::from(rd) & 0xf;

    let mut i = rs2;
    i |= rd << 12;
    i |= rs1 << 16;
    i |= s << 20;
    i |= opcode << 21;
    i |= cond << 28;

    sink.put4(i);
}

/// Data processing instruction (one register).
/// The same as dp_rr but with 0000 on rs1 place.
fn put_dp_r<CS: CodeSink + ?Sized>(bits: u16, rs: RegUnit, rd: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let opcode = (bits >> 4) & 0xf;
    let s = (bits >> 8) & 0x1;
    let rs = u32::from(rs) & 0xf;
    let rd = u32::from(rd) & 0xf;

    let mut i = rs;
    i |= rd << 12;
    i |= s << 20;
    i |= opcode << 21;
    i |= cond << 28;

    sink.put4(i);
}

/// Data Processing instruction (register and immediate).
///
///   31   27  24       19  15   11    7
///   cond 001 opcode S  rs  rd  shift imm
///     28  25     21     16  12     8   0
///
/// cond = bits[3:0], opcode = bits[7:4], shift = bits[11:8], S = bits[12].
/// S = 1 => modify flags
fn put_dp_ri<CS: CodeSink + ?Sized>(bits: u16, rs: RegUnit, imm: u8, rd: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let opcode = (bits >> 4) & 0xf;
    let shift = (bits >> 8) & 0xf;
    let s = (bits >> 12) & 0x1;
    let rs = u32::from(rs) & 0xf;
    let imm = u32::from(imm);
    let rd = u32::from(rd) & 0xf;

    let mut i = imm;
    i |= shift << 8;
    i |= rd << 12;
    i |= rs << 16;
    i |= s << 20;
    i |= opcode << 21;
    i |= 0b001 << 25;
    i |= cond << 28;

    sink.put4(i);
}

/// Data processing instruction (one immediate).
/// The same as dp_r but with 0000 on rs place.
fn put_dp_i<CS: CodeSink + ?Sized>(bits: u16, imm: u8, rd: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let opcode = (bits >> 4) & 0xf;
    let shift = (bits >> 8) & 0xf;
    let s = (bits >> 12) & 0x1;
    let imm = u32::from(imm);
    let rd = u32::from(rd) & 0xf;

    let mut i = imm;
    i |= shift << 8;
    i |= rd << 12;
    i |= s << 20;
    i |= opcode << 21;
    i |= 0b001 << 25;
    i |= cond << 28;

    sink.put4(i);
}

/// MUL instruction.
///
///   31   27        19   15   11  7    3
///   cond 0000000 S  rd  0000 rs1 1001 rs2
///     28      21     16   12   8    4   0
///
/// cond = bits[3:0], S = bits[4].
fn put_mul<CS: CodeSink + ?Sized>(
    bits: u16,
    rs1: RegUnit,
    rs2: RegUnit,
    rd: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let s = (bits >> 4) & 0x1;
    let rs1 = u32::from(rs1) & 0xf;
    let rs2 = u32::from(rs2) & 0xf;
    let rd = u32::from(rd) & 0xf;

    let mut i = rs2;
    i |= 0b1001 << 4;
    i |= rs1 << 8;
    i |= rd << 16;
    i |= s << 20;
    i |= cond << 28;

    sink.put4(i);
}

/// MULL instruction.
///
///   31   27           19    15    11  7    3
///   cond 00001 U 0 S  rd_hi rd_lo  rs 1001 rm
///     28    23           16    12   8    4  0
///
/// cond = bits[3:0], S = bits[4], U = bits[5].
/// U = 1 => unsigned, U = 0 => signed
fn put_mull<CS: CodeSink + ?Sized>(
    bits: u16,
    rs: RegUnit,
    rm: RegUnit,
    rd_lo: RegUnit,
    rd_hi: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let s = (bits >> 4) & 0x1;
    let u = ((bits >> 5) & 0x1) ^ 0x1;
    let rm = u32::from(rm) & 0xf;
    let rs = u32::from(rs) & 0xf;
    let rd_lo = u32::from(rd_lo) & 0xf;
    let rd_hi = u32::from(rd_hi) & 0xf;

    let mut i = rm;
    i |= 0b1001 << 4;
    i |= rs << 8;
    i |= rd_lo << 12;
    i |= rd_hi << 16;
    i |= s << 20;
    i |= u << 22;
    i |= 0b0001 << 23;
    i |= cond << 28;

    sink.put4(i);
}

/// UDIV, SDIV instructions.
///
///   31   27          19   15   11  7    3
///   cond 0111 00 U 1  rd  1111  rm 0001 rn
///     28   24          16   12   8     4  0
///
/// cond = bits[3:0], U = bits[4].
/// U = 1 => UDIV, U = 0 => SDIV
fn put_div<CS: CodeSink + ?Sized>(bits: u16, rn: RegUnit, rm: RegUnit, rd: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let u = (bits >> 4) & 0x1;
    let rd = u32::from(rd) & 0xf;
    let rn = u32::from(rn) & 0xf;
    let rm = u32::from(rm) & 0xf;

    let mut i = rn;
    i |= 0b0001 << 4;
    i |= rm << 8;
    i |= 0b1111 << 12;
    i |= rd << 16;
    i |= 0b0001 << 20;
    i |= u << 21;
    i |= 0b0111 << 24;
    i |= cond << 28;

    sink.put4(i);
}

/// MOVW, MOVT instructions.
///
///   31   27         19   15   11
///   cond 00110 T 00 imm4  rd  imm12
///     28    23        16   12     0
///
/// cond = bits[3:0], T = bits[4].
/// T = 1 => MOVT, T = 0 => MOVW
fn put_mov16_i<CS: CodeSink + ?Sized>(bits: u16, imm: u16, reg: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let t = (bits >> 4) & 0x1;
    let imm = u32::from(imm);
    let reg = u32::from(reg) & 0xf;

    let mut i = imm & 0xfff;
    i |= reg << 12;
    i |= (imm >> 12) << 16;
    i |= t << 22;
    i |= 0b00110 << 23;
    i |= cond << 28;

    sink.put4(i);
}

/// Moves any 32 bit immediate to register.
/// Puts 8 bytes.
fn put_mov32_i<CS: CodeSink + ?Sized>(imm: i64, reg: RegUnit, sink: &mut CS) {
    put_mov16_i(AL, (imm & 0xffff) as u16, reg, sink);
    put_mov16_i(AL | 0x10, ((imm >> 16) & 0xffff) as u16, reg, sink);
}

/// Puts MOV instruction with rotation or shift applied.
/// Rotation or shift amount is stored in register.
///
///   31   27         19   15   11    6      3
///   cond 0001 101 S 0000  rd  rs2 0 type 1 rs1
///     28       21     16   12   8      5     0
///
/// cond = bits[3:0], type = bits[5:4], S = bits[6]
fn put_mov_rotate_r<CS: CodeSink + ?Sized>(
    bits: u16,
    rs1: RegUnit,
    rs2: RegUnit,
    rd: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let _type = (bits >> 4) & 0x3;
    let s = (bits >> 6) & 0x1;
    let rs1 = u32::from(rs1) & 0xf;
    let rs2 = u32::from(rs2) & 0xf;
    let rd = u32::from(rd) & 0xf;

    let mut i = rs1;
    i |= 0b1 << 4;
    i |= _type << 5;
    i |= rs2 << 8;
    i |= rd << 12;
    i |= s << 20;
    i |= 0b0001_101 << 21;
    i |= cond << 28;

    sink.put4(i);
}

/// Puts MOV instruction with rotation or shift applied.
/// Rotation or shift amount is specified by immediate.
///
///   31   27         19   15   11  6      3
///   cond 0001 101 S 0000  rd  imm type 0 rs
///     28       21     16   12   7    5    0
///
/// cond = bits[3:0], type = bits[5:4], S = bits[6].
fn put_mov_rotate_i<CS: CodeSink + ?Sized>(
    bits: u16,
    rs: RegUnit,
    imm: u8,
    rd: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let _type = (bits >> 4) & 0x3;
    let s = (bits >> 6) & 0x1;
    let rs = u32::from(rs) & 0xf;
    let imm = u32::from(imm) & 0x1f;
    let rd = u32::from(rd) & 0xf;

    let mut i = rs;
    i |= _type << 5;
    i |= imm << 7;
    i |= rd << 12;
    i |= s << 20;
    i |= 0b0001_101 << 21;
    i |= cond << 28;

    sink.put4(i);
}

/// Branch instruction (B, BL).
///
///   31   27    23
///   cond 101 L offset
///     28  25        0
///
/// cond = bits[3:0], L = bits[4].
/// L = 1 => BL instruction, L = 0 => B instruction
fn put_b<CS: CodeSink + ?Sized>(bits: u16, offset: u32, sink: &mut CS) {
    let cond = (bits & 0xf) as u32;
    let l = ((bits >> 4) & 0x1) as u32;

    let mut i: u32 = offset & 0xffffff;
    i |= l << 23;
    i |= 0b101 << 25;
    i |= cond << 28;

    sink.put4(i);
}

/// BX, BLX instructions.
///
///   31   27   23        15               3
///   cond 0001 0010 1111 1111 1111 00 L 1 rn
///     28   24        16         8         0
///
/// cond = bits[3:0], L = bits[4].
/// L = 1 => load PC to LR, L = 0 => do not alter LR.
fn put_bx<CS: CodeSink + ?Sized>(bits: u16, rn: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let l = (bits >> 4) & 0x1;
    let rn = u32::from(rn) & 0xf;

    let mut i = rn;
    i |= 1 << 4;
    i |= l << 5;
    i |= 0b1111_1111 << 8;
    i |= 0b0010_1111 << 16;
    i |= 0b0001 << 24;
    i |= cond << 28;

    sink.put4(i);
}

/// Puts UDF instruction for Cranelift IR trap instructions.
///
///   31   27                       8
///   1110 0111 1111 0000 0000 0000 1111 0000
///     28        20                   5
///
fn put_udf<CS: CodeSink + ?Sized>(sink: &mut CS) {
    //let bits = u32::from(bits);
    let mut i = 0;

    i |= 0b1111 << 4;
    i |= 0b0111_1111 << 20;
    i |= 0b1110 << 28;

    sink.put4(i);
}

/// Load and store instructions with register as offset.
///
///   31   27          19  15   11    3
///   cond 01111 B 0 L  rn  rt  shift offset
///     28    23         16  12     4      0
///
/// cond = bits[3:0], L = bits[4], B = bits[5], shift = bits[15:8]
/// L = 1 => load, L = 0 => store
/// B = 1 => byte quantity, B = 0 => word quantity
fn put_mem_transfer_r<CS: CodeSink + ?Sized>(
    bits: u16,
    rn: RegUnit,
    offset: RegUnit,
    rt: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let l = (bits >> 4) & 0x1;
    let b = (bits >> 5) & 0x1;
    let shift = (bits >> 8) & 0xff;
    let rn = u32::from(rn) & 0xf;
    let offset = u32::from(offset) & 0xf;
    let rt = u32::from(rt) & 0xf;

    let mut i = offset;
    i |= shift << 4;
    i |= rt << 12;
    i |= rn << 16;
    i |= l << 20;
    i |= b << 22;
    i |= 0b1111 << 23;
    i |= cond << 28;

    sink.put4(i);
}

/// Load and store instructions with immediate as offset.
///
///   31   27          19  15   11    7
///   cond 01011 B 0 L  rn  rt  shift offset
///     28    23         16  12     8      0
///
/// cond = bits[3:0], shift = bits[7:4], L = bits[8], B = bits[9].
fn put_mem_transfer_i<CS: CodeSink + ?Sized>(
    bits: u16,
    rn: RegUnit,
    offset: u8,
    rt: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let shift = (bits >> 4) & 0xf;
    let l = (bits >> 8) & 0x1;
    let b = (bits >> 9) & 0x1;
    let rn = u32::from(rn) & 0xf;
    let rt = u32::from(rt) & 0xf;

    let mut i: u32 = offset.into();
    i |= shift << 8;
    i |= rt << 12;
    i |= rn << 16;
    i |= l << 20;
    i |= b << 22;
    i |= 0b1011 << 23;
    i |= cond << 28;

    sink.put4(i);
}

/// Load and store halfword instructions with register as offset.
///
///   31   27         19  15   11        3
///   cond 000 1100 L  rn  rt  0000 1011 offset
///     28       21     16  12         4      0
///
/// cond = bits[3:0], L = bits[4].
/// L = 1 => load, L = 0 => store
fn put_mem_transfer_halfword_r<CS: CodeSink + ?Sized>(
    bits: u16,
    rn: RegUnit,
    offset: RegUnit,
    rt: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let l = (bits >> 4) & 0x1;
    let rn = u32::from(rn) & 0xf;
    let rt = u32::from(rt) & 0xf;
    let offset = u32::from(offset) & 0xf;

    let mut i: u32 = offset;
    i |= 0b1011 << 4;
    i |= rt << 12;
    i |= rn << 16;
    i |= l << 20;
    i |= 0b1100 << 21;
    i |= cond << 28;

    sink.put4(i);
}

/// Load and store halfword instructions with immediate as offset.
///
///   31   27        19  15   11    7    3
///   cond 0001110 L  rn  rt  imm4h 1011 imm4l
///     28      21     16  12     8    4     0
///
/// cond = bits[3:0], L = bits[4].
/// L = 1 => load, L = 0 => store
fn put_mem_transfer_halfword_i<CS: CodeSink + ?Sized>(
    bits: u16,
    rn: RegUnit,
    offset: u8,
    rt: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let l = (bits >> 4) & 0x1;
    let rn = u32::from(rn) & 0xf;
    let rt = u32::from(rt) & 0xf;
    let offset = u32::from(offset);

    let mut i: u32 = offset & 0xf;
    i |= 0b1011 << 4;
    i |= ((offset >> 4) & 0xf) << 8;
    i |= rt << 12;
    i |= rn << 16;
    i |= l << 20;
    i |= 0b1110 << 21;
    i |= cond << 28;

    sink.put4(i);
}

/// Puts extend instruction (UXTB, SXTB, UXTH, SXTH).
///
///   31   27          19   15   11        3
///   cond 01101 U 1 H 1111  rd  0000 0111 rm
///     28    23         16   12         4  0
///
/// cond = bits[3:0], H = bits[4], U = bits[5].
/// H = 1 => extend lower halfword from rm, H = 0 => extend last 8 bits of rm.
/// U = 1 => zero-extend, U = 0 => sign-extend
fn put_extend<CS: CodeSink + ?Sized>(bits: u16, rm: RegUnit, rd: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let h = (bits >> 4) & 0x1;
    let u = (bits >> 5) & 0x1;
    let rm = u32::from(rm) & 0xf;
    let rd = u32::from(rd) & 0xf;

    let mut i = rm;
    i |= 0b0111 << 4;
    i |= rd << 12;
    i |= 0b1111 << 16;
    i |= h << 20;
    i |= 1 << 21;
    i |= u << 22;
    i |= 0b1101 << 23;
    i |= cond << 28;

    sink.put4(i);
}

/// RBIT instruction.
///
///   31   27   23        15   11        3
///   cond 0110 1111 1111  rd  1111 0011 rm
///     28   24        16   12         4  0
///
/// cond = bits[0:3].
fn put_bitrev<CS: CodeSink + ?Sized>(bits: u16, rm: RegUnit, rd: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let rm = u32::from(rm) & 0xf;
    let rd = u32::from(rd) & 0xf;

    let mut i = rm;
    i |= 0b1111_0011 << 4;
    i |= rd << 12;
    i |= 0b1111_1111 << 16;
    i |= 0b0110 << 24;
    i |= cond << 28;

    sink.put4(i);
}

/// CLZ instruction.
///
///   31   27   23        15   11        3
///   cond 0001 0110 1111  rd  1111 0001 rm
///     28   24        16   12         4  0
///
/// cond = bits[0:3].
fn put_clz<CS: CodeSink + ?Sized>(bits: u16, rm: RegUnit, rd: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let cond = bits & 0xf;
    let rm = u32::from(rm) & 0xf;
    let rd = u32::from(rd) & 0xf;

    let mut i = rm;
    i |= 0b1111_0001 << 4;
    i |= rd << 12;
    i |= 0b0110_1111 << 16;
    i |= 0b0001 << 24;
    i |= cond << 28;

    sink.put4(i);
}

/// Common register encoding bits for vfp instructions.
///
///   31   27   23   19  15   11     7    3
///   xxxx xxxx xDxx  vn  vd  xxx sz NxMx vm
///     28   24   20   16  12   9       4  0
///
/// sz = 1 => double precision, sz = 0 => single precision.
fn vfp_regs_enc(double: bool, regs: &[RegUnit; 3], regs_double: &[bool; 3]) -> u32 {
    let mut vn = u32::from(regs[0]) & 0x1f;
    let mut vm = u32::from(regs[1]) & 0x1f;
    let mut vd = u32::from(regs[2]) & 0x1f;

    let mut i: u32 = 0;

    if double {
        i |= 1 << 8;
    }
    if regs_double[0] {
        vn /= 2;
        i |= (vn & 0xf) << 16;
        i |= (vn >> 4) << 7;
    } else {
        i |= (vn & 0x1) << 7;
        i |= (vn >> 1) << 16;
    }
    if regs_double[1] {
        vm /= 2;
        i |= vm & 0xf;
        i |= (vm >> 4) << 5;
    } else {
        i |= (vm & 0x1) << 5;
        i |= vm >> 1;
    }
    if regs_double[2] {
        vd /= 2;
        i |= (vd & 0xf) << 12;
        i |= (vd >> 4) << 22;
    } else {
        i |= (vd & 0x1) << 22;
        i |= (vd >> 1) << 12;
    }

    return i;
}

/// VFP data-processing instructions.
///
///   31   27   23   19   15   11   7    5  3
///   1110 1110 opc1 opc2 xxxx 101x opc3 x0 xxxx
///     28   24   20   16   12    8    6  4    0
///
/// sz = bits[0], vn_double = bits[1], vm_double = bits[2], vd_double = bits[3],
/// opc1 = bits[4:7], opc2 = bits[8:11], opc3 = bits[12:13].
/// If vn, vm or vd is unused, it should be equal to NULL_REG.
fn put_vfp_dp<CS: CodeSink + ?Sized>(
    bits: u16,
    vn: RegUnit,
    vm: RegUnit,
    vd: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let sz = bits & 0x1;
    let vn_double = (bits & 0x2) != 0;
    let vm_double = (bits & 0x4) != 0;
    let vd_double = (bits & 0x8) != 0;
    let opc1 = (bits >> 4) & 0xf;
    let opc2 = (bits >> 8) & 0xf;
    let opc3 = (bits >> 12) & 0x3;

    let mut i = vfp_regs_enc(sz != 0, &[vn, vm, vd], &[vn_double, vm_double, vd_double]);
    i |= opc3 << 6;
    i |= 0b101 << 9;
    i |= opc2 << 16;
    i |= opc1 << 20;
    i |= 0b1110_1110 << 24;

    sink.put4(i);
}

/// VFP data transfer instructions.
///
///   31   27   23   19  15   11  8      3
///   1110 1110 opc1  vn  rt  101 opc2 1 0000
///     28   24   20   16  12   9    5      0
///
/// opc1 = bits[0:3], opc2 = bits[4:6], sz = bits[7].
fn put_vfp_transfer<CS: CodeSink + ?Sized>(bits: u16, vn: RegUnit, rt: RegUnit, sink: &mut CS) {
    let bits = u32::from(bits);
    let opc1 = bits & 0xf;
    let opc2 = (bits >> 4) & 0x7;
    let sz = (bits >> 7) & 0x1;
    let rt = u32::from(rt) & 0xf;
    let double = sz != 0;

    let mut i = vfp_regs_enc(double, &[vn, NULL_REG, NULL_REG], &[double; 3]);
    i |= 1 << 4;
    i |= opc2 << 5;
    i |= 0b101 << 9;
    i |= rt << 12;
    i |= opc1 << 20;
    i |= 0b1110_1110 << 24;

    sink.put4(i);
}

/// VFP VMOV instruction that transfers contents
/// between two ARM core registers and D floating point register.
///
///   31   27   23     19  15   11   7    3
///   1110 1100 010 op rt2  rt  1011 M001 vm
///     28   24  21     16   12    8    4  0
///
/// op = bits[0].
/// op = 1 => D to core regs, op = 0 => core regs to D.
fn put_vfp_64_transfer<CS: CodeSink + ?Sized>(
    bits: u16,
    vm: RegUnit,
    rt: RegUnit,
    rt2: RegUnit,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let op = bits & 0x1;
    let rt = u32::from(rt) & 0xf;
    let rt2 = u32::from(rt2) & 0xf;

    let mut i = vfp_regs_enc(true, &[NULL_REG, vm, NULL_REG], &[true; 3]);
    i |= 0b1011_0001 << 4;
    i |= rt << 12;
    i |= rt2 << 16;
    i |= op << 20;
    i |= 0b1100_010 << 21;
    i |= 0b1110 << 28;

    sink.put4(i);
}

/// VFP VSTR/VLDR isntructions.
///
///   31   27   23   19  15   11     7
///   1110 1101 1D0L  rn  vd  101 sz offset
///     28   24   20   16  12   9         0
///
/// L = bits[0], sz = bits[1].
/// L = 1 => load, L = 0 => store, sz = 1 => D register, sz = 0 => S register.
fn put_vfp_mem_transfer<CS: CodeSink + ?Sized>(
    bits: u16,
    vd: RegUnit,
    rn: RegUnit,
    offset: u8,
    sink: &mut CS,
) {
    let bits = u32::from(bits);
    let l = bits & 0x1;
    let sz = (bits >> 1) & 0x1;
    let rn = u32::from(rn) & 0xf;
    let offset = u32::from(offset >> 2);
    let double = sz != 0;

    let mut i = vfp_regs_enc(double, &[NULL_REG, NULL_REG, vd], &[double; 3]);
    i |= offset;
    i |= 0b101 << 9;
    i |= rn << 16;
    i |= l << 20;
    i |= 0b1101_1 << 23;
    i |= 0b1110 << 28;

    sink.put4(i);
}

/// Cranelift IR integer condition code => ARM condition code.
fn icc2bits(cond: IntCC) -> u16 {
    use crate::ir::condcodes::IntCC::*;
    match cond {
        Equal => EQ,                      // EQ (Z==1)
        NotEqual => NE,                   // NE (Z==0)
        SignedLessThan => LT,             // LT (N!=V)
        SignedGreaterThanOrEqual => GE,   // GE (N==V)
        SignedGreaterThan => GT,          // GT (Z==0 && N==V)
        SignedLessThanOrEqual => LE,      // LE (Z==1 || N!=V)
        UnsignedLessThan => CC,           // CC|LO (C==0)
        UnsignedGreaterThanOrEqual => CS, // CS|HS (C==1),
        UnsignedGreaterThan => HI,        // HI (C==1 && Z==0)
        UnsignedLessThanOrEqual => LS,    // LS (C==0 || Z==1)
        Overflow => VS,                   // VS (V set)
        NotOverflow => VC,                // VC (V clear)
    }
}

/// Cranelift IR floating point condition code => ARM condition code.
fn fcc2bits(cond: FloatCC) -> u16 {
    use crate::ir::condcodes::FloatCC::*;
    match cond {
        Ordered => VC,
        Unordered => VS,
        Equal => EQ,
        NotEqual => NE,
        LessThan => CC,
        LessThanOrEqual => LS,
        GreaterThan => GT,
        GreaterThanOrEqual => GE,
        UnorderedOrLessThan => LT,
        UnorderedOrLessThanOrEqual => LE,
        UnorderedOrGreaterThan => HI,
        UnorderedOrGreaterThanOrEqual => PL,
        OrderedNotEqual | UnorderedOrEqual => panic!("{} not supported by fcc2bits", cond),
    }
}

// Convert a stack base to the corresponding register.
fn stk_base(base: StackBase) -> RegUnit {
    let ru = match base {
        StackBase::SP => RU::r13,
        _ => unimplemented!(),
    };
    ru as RegUnit
}

// Calculate branch instruction offset.
// Encoded offset is shifted left two bits and added ti PC during instruction execution.
fn branch_offset<CS: CodeSink + ?Sized>(destination: Block, func: &Function, sink: &mut CS) -> u32 {
    if (func.offsets[destination] & 0x3) != 0 {
        panic!("offset must be divisible by 4");
    }

    // PC is address of the current instruction + 8
    let off = func.offsets[destination]
        .wrapping_sub(sink.offset() + 8)
        .wrapping_div(4);

    off
}
