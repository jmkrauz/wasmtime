use crate::isa::arm32::inst::*;
use crate::isa::test_utils;
use crate::settings;

use alloc::vec::Vec;

#[test]
fn test_arm32_emit() {
    let flags = settings::Flags::new(settings::builder());
    let mut insns = Vec::<(Inst, &str, &str)>::new();

    // litle endian order
    insns.push((Inst::Nop0, "", "nop-zero-len"));
    insns.push((Inst::Nop2, "00BF", "nop"));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Lsl,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
        },
        "01FA02F0",
        "lsl r0, r1, r2",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Lsl,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
        },
        "09FA0AF8",
        "lsl r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Lsr,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
        },
        "21FA02F0",
        "lsr r0, r1, r2",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Lsr,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
        },
        "29FA0AF8",
        "lsr r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Asr,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
        },
        "41FA02F0",
        "asr r0, r1, r2",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Asr,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
        },
        "49FA0AF8",
        "asr r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Ror,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
        },
        "61FA02F0",
        "ror r0, r1, r2",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Ror,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
        },
        "69FA0AF8",
        "ror r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Qadd,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
        },
        "81FA82F0",
        "qadd r0, r1, r2",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Qadd,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
        },
        "89FA8AF8",
        "qadd r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Qsub,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
        },
        "81FAA2F0",
        "qsub r0, r1, r2",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Qsub,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
        },
        "89FAAAF8",
        "qsub r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Mul,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
        },
        "01FB02F0",
        "mul r0, r1, r2",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Mul,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
        },
        "09FB0AF8",
        "mul r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Udiv,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
        },
        "B1FBF2F0",
        "udiv r0, r1, r2",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Udiv,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
        },
        "B9FBFAF8",
        "udiv r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Sdiv,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
        },
        "91FBF2F0",
        "sdiv r0, r1, r2",
    ));
    insns.push((
        Inst::AluRRR {
            alu_op: ALUOp::Sdiv,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
        },
        "99FBFAF8",
        "sdiv r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::And,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "01EAC250",
        "and r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::And,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "09EA0A08",
        "and r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Bic,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "21EAC250",
        "bic r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Bic,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "29EA0A08",
        "bic r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Orr,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "41EAC250",
        "orr r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Orr,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "49EA0A08",
        "orr r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Orn,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "61EAC250",
        "orn r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Orn,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "69EA0A08",
        "orn r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Eor,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "81EAC250",
        "eor r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Eor,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "89EA0A08",
        "eor r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Add,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "01EBC250",
        "add r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Add,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "09EB0A08",
        "add r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Adds,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "11EBC250",
        "adds r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Adds,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "19EB0A08",
        "adds r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Adc,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "41EBC250",
        "adc r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Adc,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "49EB0A08",
        "adc r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Adcs,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "51EBC250",
        "adcs r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Adcs,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "59EB0A08",
        "adcs r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Sbc,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "61EBC250",
        "sbc r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Sbc,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "69EB0A08",
        "sbc r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Sbcs,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "71EBC250",
        "sbcs r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Sbcs,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "79EB0A08",
        "sbcs r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Sub,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "A1EBC250",
        "sub r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Sub,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "A9EB0A08",
        "sub r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Subs,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "B1EBC250",
        "subs r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Subs,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "B9EB0A08",
        "subs r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Rsb,
            rd: writable_rreg(0),
            rn: rreg(1),
            rm: rreg(2),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(23).unwrap(),
            )),
        },
        "C1EBC250",
        "rsb r0, r1, r2, lsl #23",
    ));
    insns.push((
        Inst::AluRRRShift {
            alu_op: ALUOp::Rsb,
            rd: writable_rreg(8),
            rn: rreg(9),
            rm: rreg(10),
            shift: None,
        },
        "C9EB0A08",
        "rsb r8, r9, r10",
    ));
    insns.push((
        Inst::AluRRShift {
            alu_op: ALUOp1::Mvn,
            rd: writable_rreg(0),
            rm: rreg(1),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(11).unwrap(),
            )),
        },
        "6FEAC120",
        "mvn r0, r1, lsl #11",
    ));
    insns.push((
        Inst::AluRRShift {
            alu_op: ALUOp1::Mvn,
            rd: writable_rreg(8),
            rm: rreg(9),
            shift: None,
        },
        "6FEA0908",
        "mvn r8, r9",
    ));
    insns.push((
        Inst::AluRRShift {
            alu_op: ALUOp1::Mov,
            rd: writable_rreg(0),
            rm: rreg(1),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSL,
                ShiftOpShiftImm::maybe_from_shift(11).unwrap(),
            )),
        },
        "4FEAC120",
        "mov r0, r1, lsl #11",
    ));
    insns.push((
        Inst::AluRRShift {
            alu_op: ALUOp1::Mov,
            rd: writable_rreg(2),
            rm: rreg(8),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::LSR,
                ShiftOpShiftImm::maybe_from_shift(27).unwrap(),
            )),
        },
        "4FEAD862",
        "mov r2, r8, lsr #27",
    ));
    insns.push((
        Inst::AluRRShift {
            alu_op: ALUOp1::Mov,
            rd: writable_rreg(9),
            rm: rreg(3),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::ASR,
                ShiftOpShiftImm::maybe_from_shift(3).unwrap(),
            )),
        },
        "4FEAE309",
        "mov r9, r3, asr #3",
    ));
    insns.push((
        Inst::AluRRShift {
            alu_op: ALUOp1::Mov,
            rd: writable_rreg(10),
            rm: rreg(11),
            shift: Some(ShiftOpAndAmt::new(
                ShiftOp::ROR,
                ShiftOpShiftImm::maybe_from_shift(7).unwrap(),
            )),
        },
        "4FEAFB1A",
        "mov r10, r11, ror #7",
    ));
    insns.push((
        Inst::AluRRRR {
            alu_op: ALUOp::Smull,
            rd_lo: writable_rreg(0),
            rd_hi: writable_rreg(1),
            rn: rreg(2),
            rm: rreg(3),
        },
        "82FB0301",
        "smull r0, r1, r2, r3",
    ));
    insns.push((
        Inst::AluRRRR {
            alu_op: ALUOp::Smull,
            rd_lo: writable_rreg(8),
            rd_hi: writable_rreg(9),
            rn: rreg(10),
            rm: rreg(11),
        },
        "8AFB0B89",
        "smull r8, r9, r10, r11",
    ));
    insns.push((
        Inst::AluRRRR {
            alu_op: ALUOp::Umull,
            rd_lo: writable_rreg(0),
            rd_hi: writable_rreg(1),
            rn: rreg(2),
            rm: rreg(3),
        },
        "A2FB0301",
        "umull r0, r1, r2, r3",
    ));
    insns.push((
        Inst::AluRRRR {
            alu_op: ALUOp::Umull,
            rd_lo: writable_rreg(8),
            rd_hi: writable_rreg(9),
            rn: rreg(10),
            rm: rreg(11),
        },
        "AAFB0B89",
        "umull r8, r9, r10, r11",
    ));
    insns.push((
        Inst::AluRRImm12 {
            alu_op: ALUOp::Add,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm12: 4095,
        },
        "01F6FF70",
        "add r0, r1, #4095",
    ));
    insns.push((
        Inst::AluRRImm12 {
            alu_op: ALUOp::Add,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm12: 0,
        },
        "09F20008",
        "add r8, r9, #0",
    ));
    insns.push((
        Inst::AluRRImm12 {
            alu_op: ALUOp::Sub,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm12: 1999,
        },
        "A1F2CF70",
        "sub r0, r1, #1999",
    ));
    insns.push((
        Inst::AluRRImm12 {
            alu_op: ALUOp::Sub,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm12: 101,
        },
        "A9F26508",
        "sub r8, r9, #101",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::And,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "01F0FF00",
        "and r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::And,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "09F00108",
        "and r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Bic,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "21F0FF00",
        "bic r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Bic,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "29F00108",
        "bic r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Orr,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "41F0FF00",
        "orr r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Orr,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "49F00108",
        "orr r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Orn,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "61F0FF00",
        "orn r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Orn,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "69F00108",
        "orn r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Eor,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "81F0FF00",
        "eor r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Eor,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "89F00108",
        "eor r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Add,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "01F1FF00",
        "add r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Add,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "09F10108",
        "add r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Adds,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "11F1FF00",
        "adds r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Adds,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "19F10108",
        "adds r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Adc,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "41F1FF00",
        "adc r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Adc,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "49F10108",
        "adc r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Adcs,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "51F1FF00",
        "adcs r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Adcs,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "59F10108",
        "adcs r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Sbc,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "61F1FF00",
        "sbc r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Sbc,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "69F10108",
        "sbc r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Sbcs,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "71F1FF00",
        "sbcs r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Sbcs,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "79F10108",
        "sbcs r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Sub,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "A1F1FF00",
        "sub r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Sub,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "A9F10108",
        "sub r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Subs,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "B1F1FF00",
        "subs r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Subs,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "B9F10108",
        "subs r8, r9, #1",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Rsb,
            rd: writable_rreg(0),
            rn: rreg(1),
            imm8: 255,
        },
        "C1F1FF00",
        "rsb r0, r1, #255",
    ));
    insns.push((
        Inst::AluRRImm8 {
            alu_op: ALUOp::Rsb,
            rd: writable_rreg(8),
            rn: rreg(9),
            imm8: 1,
        },
        "C9F10108",
        "rsb r8, r9, #1",
    ));
    insns.push((
        Inst::AluRImm8 {
            alu_op: ALUOp1::Mvn,
            rd: writable_rreg(0),
            imm8: 255,
        },
        "6FF0FF00",
        "mvn r0, #255",
    ));
    insns.push((
        Inst::AluRImm8 {
            alu_op: ALUOp1::Mvn,
            rd: writable_rreg(8),
            imm8: 1,
        },
        "6FF00108",
        "mvn r8, #1",
    ));
    insns.push((
        Inst::AluRImm8 {
            alu_op: ALUOp1::Mov,
            rd: writable_rreg(0),
            imm8: 0,
        },
        "4FF00000",
        "mov r0, #0",
    ));
    insns.push((
        Inst::AluRImm8 {
            alu_op: ALUOp1::Mov,
            rd: writable_rreg(8),
            imm8: 176,
        },
        "4FF0B008",
        "mov r8, #176",
    ));
    insns.push((
        Inst::BitOpRR {
            bit_op: BitOp::Rbit,
            rd: writable_rreg(0),
            rm: rreg(1),
        },
        "91FAA1F0",
        "rbit r0, r1",
    ));
    insns.push((
        Inst::BitOpRR {
            bit_op: BitOp::Rbit,
            rd: writable_rreg(8),
            rm: rreg(9),
        },
        "99FAA9F8",
        "rbit r8, r9",
    ));
    insns.push((
        Inst::BitOpRR {
            bit_op: BitOp::Rev,
            rd: writable_rreg(0),
            rm: rreg(1),
        },
        "91FA81F0",
        "rev r0, r1",
    ));
    insns.push((
        Inst::BitOpRR {
            bit_op: BitOp::Rev,
            rd: writable_rreg(8),
            rm: rreg(9),
        },
        "99FA89F8",
        "rev r8, r9",
    ));
    insns.push((
        Inst::BitOpRR {
            bit_op: BitOp::Clz,
            rd: writable_rreg(0),
            rm: rreg(1),
        },
        "B1FA81F0",
        "clz r0, r1",
    ));
    insns.push((
        Inst::BitOpRR {
            bit_op: BitOp::Clz,
            rd: writable_rreg(8),
            rm: rreg(9),
        },
        "B9FA89F8",
        "clz r8, r9",
    ));
    insns.push((
        Inst::Mov {
            rd: writable_rreg(0),
            rm: rreg(1),
        },
        "0846",
        "mov r0, r1",
    ));
    insns.push((
        Inst::Mov {
            rd: writable_rreg(2),
            rm: rreg(8),
        },
        "4246",
        "mov r2, r8",
    ));
    insns.push((
        Inst::Mov {
            rd: writable_rreg(9),
            rm: rreg(3),
        },
        "9946",
        "mov r9, r3",
    ));
    insns.push((
        Inst::Mov {
            rd: writable_rreg(10),
            rm: rreg(11),
        },
        "DA46",
        "mov r10, r11",
    ));
    insns.push((
        Inst::MovImm16 {
            rd: writable_rreg(0),
            imm16: 0,
        },
        "40F20000",
        "mov r0, #0",
    ));
    insns.push((
        Inst::MovImm16 {
            rd: writable_rreg(1),
            imm16: 15,
        },
        "40F20F01",
        "mov r1, #15",
    ));
    insns.push((
        Inst::MovImm16 {
            rd: writable_rreg(2),
            imm16: 255,
        },
        "40F2FF02",
        "mov r2, #255",
    ));
    insns.push((
        Inst::MovImm16 {
            rd: writable_rreg(8),
            imm16: 4095,
        },
        "40F6FF78",
        "mov r8, #4095",
    ));
    insns.push((
        Inst::MovImm16 {
            rd: writable_rreg(9),
            imm16: 65535,
        },
        "4FF6FF79",
        "mov r9, #65535",
    ));
    insns.push((
        Inst::Movt {
            rd: writable_rreg(0),
            imm16: 0,
        },
        "C0F20000",
        "movt r0, #0",
    ));
    insns.push((
        Inst::Movt {
            rd: writable_rreg(1),
            imm16: 15,
        },
        "C0F20F01",
        "movt r1, #15",
    ));
    insns.push((
        Inst::Movt {
            rd: writable_rreg(2),
            imm16: 255,
        },
        "C0F2FF02",
        "movt r2, #255",
    ));
    insns.push((
        Inst::Movt {
            rd: writable_rreg(8),
            imm16: 4095,
        },
        "C0F6FF78",
        "movt r8, #4095",
    ));
    insns.push((
        Inst::Movt {
            rd: writable_rreg(9),
            imm16: 65535,
        },
        "CFF6FF79",
        "movt r9, #65535",
    ));
    insns.push((
        Inst::Cmp {
            rn: rreg(0),
            rm: rreg(1),
        },
        "8842",
        "cmp r0, r1",
    ));
    insns.push((
        Inst::Cmp {
            rn: rreg(2),
            rm: rreg(8),
        },
        "4245",
        "cmp r2, r8",
    ));
    insns.push((
        Inst::Cmp {
            rn: rreg(9),
            rm: rreg(3),
        },
        "9945",
        "cmp r9, r3",
    ));
    insns.push((
        Inst::Cmp {
            rn: rreg(10),
            rm: rreg(11),
        },
        "DA45",
        "cmp r10, r11",
    ));
    insns.push((
        Inst::CmpImm8 {
            rn: rreg(0),
            imm8: 255,
        },
        "B0F1FF0F",
        "cmp r0, #255",
    ));
    insns.push((
        Inst::CmpImm8 {
            rn: rreg(1),
            imm8: 0,
        },
        "B1F1000F",
        "cmp r1, #0",
    ));
    insns.push((
        Inst::CmpImm8 {
            rn: rreg(8),
            imm8: 1,
        },
        "B8F1010F",
        "cmp r8, #1",
    ));

    insns.push((
        Inst::Store {
            rt: rreg(0),
            mem: MemArg::reg_plus_reg(rreg(1), rreg(2), 0),
            srcloc: None,
            bytes: ByteAmt::Word,
        },
        "41F80200",
        "str r0, [r1, r2]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(8),
            mem: MemArg::reg_plus_reg(rreg(9), rreg(10), 3),
            srcloc: None,
            bytes: ByteAmt::Word,
        },
        "49F83A80",
        "str r8, [r9, r10, lsl #3]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(0),
            mem: MemArg::RegOffset(rreg(1), 4095),
            srcloc: None,
            bytes: ByteAmt::Word,
        },
        "C1F8FF0F",
        "str r0, [r1, #4095]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(8),
            mem: MemArg::RegOffset(rreg(9), 0),
            srcloc: None,
            bytes: ByteAmt::Word,
        },
        "C9F80080",
        "str r8, [r9, #0]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(7),
            mem: MemArg::RegOffset(rreg(11), 65535),
            srcloc: None,
            bytes: ByteAmt::Word,
        },
        "4FF6FF7C4BF80C70",
        "mov ip, #65535 ; str r7, [r11, ip]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(10),
            mem: MemArg::RegOffset(rreg(4), 16777215),
            srcloc: None,
            bytes: ByteAmt::Word,
        },
        "4FF6FF7CC0F2FF0C44F80CA0",
        "mov ip, #65535 ; movt ip, #255 ; str r10, [r4, ip]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(0),
            mem: MemArg::reg_plus_reg(rreg(1), rreg(2), 0),
            srcloc: None,
            bytes: ByteAmt::Halfword,
        },
        "21F80200",
        "strh r0, [r1, r2]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(8),
            mem: MemArg::reg_plus_reg(rreg(9), rreg(10), 2),
            srcloc: None,
            bytes: ByteAmt::Halfword,
        },
        "29F82A80",
        "strh r8, [r9, r10, lsl #2]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(0),
            mem: MemArg::RegOffset(rreg(1), 3210),
            srcloc: None,
            bytes: ByteAmt::Halfword,
        },
        "A1F88A0C",
        "strh r0, [r1, #3210]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(8),
            mem: MemArg::RegOffset(rreg(9), 1),
            srcloc: None,
            bytes: ByteAmt::Halfword,
        },
        "A9F80180",
        "strh r8, [r9, #1]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(7),
            mem: MemArg::RegOffset(rreg(11), 65535),
            srcloc: None,
            bytes: ByteAmt::Halfword,
        },
        "4FF6FF7C2BF80C70",
        "mov ip, #65535 ; strh r7, [r11, ip]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(10),
            mem: MemArg::RegOffset(rreg(4), 16777215),
            srcloc: None,
            bytes: ByteAmt::Halfword,
        },
        "4FF6FF7CC0F2FF0C24F80CA0",
        "mov ip, #65535 ; movt ip, #255 ; strh r10, [r4, ip]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(0),
            mem: MemArg::reg_plus_reg(rreg(1), rreg(2), 0),
            srcloc: None,
            bytes: ByteAmt::Byte,
        },
        "01F80200",
        "strb r0, [r1, r2]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(8),
            mem: MemArg::reg_plus_reg(rreg(9), rreg(10), 1),
            srcloc: None,
            bytes: ByteAmt::Byte,
        },
        "09F81A80",
        "strb r8, [r9, r10, lsl #1]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(0),
            mem: MemArg::RegOffset(rreg(1), 4),
            srcloc: None,
            bytes: ByteAmt::Byte,
        },
        "81F80400",
        "strb r0, [r1, #4]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(8),
            mem: MemArg::RegOffset(rreg(9), 777),
            srcloc: None,
            bytes: ByteAmt::Byte,
        },
        "89F80983",
        "strb r8, [r9, #777]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(7),
            mem: MemArg::RegOffset(rreg(11), 65535),
            srcloc: None,
            bytes: ByteAmt::Byte,
        },
        "4FF6FF7C0BF80C70",
        "mov ip, #65535 ; strb r7, [r11, ip]",
    ));
    insns.push((
        Inst::Store {
            rt: rreg(10),
            mem: MemArg::RegOffset(rreg(4), 16777215),
            srcloc: None,
            bytes: ByteAmt::Byte,
        },
        "4FF6FF7CC0F2FF0C04F80CA0",
        "mov ip, #65535 ; movt ip, #255 ; strb r10, [r4, ip]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(0),
            mem: MemArg::reg_plus_reg(rreg(1), rreg(2), 0),
            srcloc: None,
            bytes: ByteAmt::Word,
            sign_extend: false,
        },
        "51F80200",
        "ldr r0, [r1, r2]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(8),
            mem: MemArg::reg_plus_reg(rreg(9), rreg(10), 1),
            srcloc: None,
            bytes: ByteAmt::Word,
            sign_extend: false,
        },
        "59F81A80",
        "ldr r8, [r9, r10, lsl #1]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(0),
            mem: MemArg::RegOffset(rreg(1), 55),
            srcloc: None,
            bytes: ByteAmt::Word,
            sign_extend: false,
        },
        "D1F83700",
        "ldr r0, [r1, #55]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(8),
            mem: MemArg::RegOffset(rreg(9), 1234),
            srcloc: None,
            bytes: ByteAmt::Word,
            sign_extend: false,
        },
        "D9F8D284",
        "ldr r8, [r9, #1234]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(7),
            mem: MemArg::RegOffset(rreg(11), 9876),
            srcloc: None,
            bytes: ByteAmt::Word,
            sign_extend: false,
        },
        "42F2946C5BF80C70",
        "mov ip, #9876 ; ldr r7, [r11, ip]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(10),
            mem: MemArg::RegOffset(rreg(4), 252645135),
            srcloc: None,
            bytes: ByteAmt::Word,
            sign_extend: false,
        },
        "40F60F7CC0F60F7C54F80CA0",
        "mov ip, #3855 ; movt ip, #3855 ; ldr r10, [r4, ip]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(0),
            mem: MemArg::reg_plus_reg(rreg(1), rreg(2), 0),
            srcloc: None,
            bytes: ByteAmt::Halfword,
            sign_extend: true,
        },
        "31F90200",
        "ldrsh r0, [r1, r2]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(8),
            mem: MemArg::reg_plus_reg(rreg(9), rreg(10), 2),
            srcloc: None,
            bytes: ByteAmt::Halfword,
            sign_extend: false,
        },
        "39F82A80",
        "ldrh r8, [r9, r10, lsl #2]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(0),
            mem: MemArg::RegOffset(rreg(1), 55),
            srcloc: None,
            bytes: ByteAmt::Halfword,
            sign_extend: false,
        },
        "B1F83700",
        "ldrh r0, [r1, #55]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(8),
            mem: MemArg::RegOffset(rreg(9), 1234),
            srcloc: None,
            bytes: ByteAmt::Halfword,
            sign_extend: true,
        },
        "B9F9D284",
        "ldrsh r8, [r9, #1234]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(7),
            mem: MemArg::RegOffset(rreg(11), 9876),
            srcloc: None,
            bytes: ByteAmt::Halfword,
            sign_extend: true,
        },
        "42F2946C3BF90C70",
        "mov ip, #9876 ; ldrsh r7, [r11, ip]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(10),
            mem: MemArg::RegOffset(rreg(4), 252645135),
            srcloc: None,
            bytes: ByteAmt::Halfword,
            sign_extend: false,
        },
        "40F60F7CC0F60F7C34F80CA0",
        "mov ip, #3855 ; movt ip, #3855 ; ldrh r10, [r4, ip]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(0),
            mem: MemArg::reg_plus_reg(rreg(1), rreg(2), 0),
            srcloc: None,
            bytes: ByteAmt::Byte,
            sign_extend: true,
        },
        "11F90200",
        "ldrsb r0, [r1, r2]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(8),
            mem: MemArg::reg_plus_reg(rreg(9), rreg(10), 3),
            srcloc: None,
            bytes: ByteAmt::Byte,
            sign_extend: false,
        },
        "19F83A80",
        "ldrb r8, [r9, r10, lsl #3]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(0),
            mem: MemArg::RegOffset(rreg(1), 55),
            srcloc: None,
            bytes: ByteAmt::Byte,
            sign_extend: false,
        },
        "91F83700",
        "ldrb r0, [r1, #55]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(8),
            mem: MemArg::RegOffset(rreg(9), 1234),
            srcloc: None,
            bytes: ByteAmt::Byte,
            sign_extend: true,
        },
        "99F9D284",
        "ldrsb r8, [r9, #1234]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(7),
            mem: MemArg::RegOffset(rreg(11), 9876),
            srcloc: None,
            bytes: ByteAmt::Byte,
            sign_extend: true,
        },
        "42F2946C1BF90C70",
        "mov ip, #9876 ; ldrsb r7, [r11, ip]",
    ));
    insns.push((
        Inst::Load {
            rt: writable_rreg(10),
            mem: MemArg::RegOffset(rreg(4), 252645135),
            srcloc: None,
            bytes: ByteAmt::Byte,
            sign_extend: false,
        },
        "40F60F7CC0F60F7C14F80CA0",
        "mov ip, #3855 ; movt ip, #3855 ; ldrb r10, [r4, ip]",
    ));
    insns.push((
        Inst::Extend {
            rd: writable_rreg(0),
            rm: rreg(1),
            from_bytes: ByteAmt::Halfword,
            signed: false,
        },
        "88B2",
        "uxth r0, r1",
    ));
    insns.push((
        Inst::Extend {
            rd: writable_rreg(8),
            rm: rreg(9),
            from_bytes: ByteAmt::Halfword,
            signed: false,
        },
        "1FFA89F8",
        "uxth r8, r9",
    ));
    insns.push((
        Inst::Extend {
            rd: writable_rreg(0),
            rm: rreg(1),
            from_bytes: ByteAmt::Byte,
            signed: false,
        },
        "C8B2",
        "uxtb r0, r1",
    ));
    insns.push((
        Inst::Extend {
            rd: writable_rreg(8),
            rm: rreg(9),
            from_bytes: ByteAmt::Byte,
            signed: false,
        },
        "5FFA89F8",
        "uxtb r8, r9",
    ));
    insns.push((
        Inst::Extend {
            rd: writable_rreg(0),
            rm: rreg(1),
            from_bytes: ByteAmt::Halfword,
            signed: true,
        },
        "08B2",
        "sxth r0, r1",
    ));
    insns.push((
        Inst::Extend {
            rd: writable_rreg(8),
            rm: rreg(9),
            from_bytes: ByteAmt::Halfword,
            signed: true,
        },
        "0FFA89F8",
        "sxth r8, r9",
    ));
    insns.push((
        Inst::Extend {
            rd: writable_rreg(0),
            rm: rreg(1),
            from_bytes: ByteAmt::Byte,
            signed: true,
        },
        "48B2",
        "sxtb r0, r1",
    ));
    insns.push((
        Inst::Extend {
            rd: writable_rreg(8),
            rm: rreg(9),
            from_bytes: ByteAmt::Byte,
            signed: true,
        },
        "4FFA89F8",
        "sxtb r8, r9",
    ));
    insns.push((
        Inst::It {
            cond: Cond::Eq,
            insts: vec![CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), true)],
        },
        "08BF0046",
        "it eq ; mov r0, r0",
    ));
    insns.push((
        Inst::It {
            cond: Cond::Ne,
            insts: vec![
                CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), true),
                CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), false),
            ],
        },
        "14BF00460046",
        "ite ne ; mov r0, r0 ; mov r0, r0",
    ));
    insns.push((
        Inst::It {
            cond: Cond::Lt,
            insts: vec![
                CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), true),
                CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), false),
                CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), true),
            ],
        },
        "B6BF004600460046",
        "itet lt ; mov r0, r0 ; mov r0, r0 ; mov r0, r0",
    ));
    insns.push((
        Inst::It {
            cond: Cond::Hs,
            insts: vec![
                CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), true),
                CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), true),
                CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), false),
                CondInst::new(Inst::mov(writable_rreg(0), rreg(0)), false),
            ],
        },
        "27BF0046004600460046",
        "ittee hs ; mov r0, r0 ; mov r0, r0 ; mov r0, r0 ; mov r0, r0",
    ));
    insns.push((Inst::Bkpt, "00BE", "bkpt #0"));
    insns.push((
        Inst::Udf {
            trap_info: (SourceLoc::default(), TrapCode::Interrupt),
        },
        "00DE",
        "udf #0",
    ));
    insns.push((
        Inst::Push {
            reg_list: smallvec![rreg(0)],
        },
        "4DF8040D",
        "push {r0}",
    ));
    insns.push((
        Inst::Push {
            reg_list: smallvec![rreg(8)],
        },
        "4DF8048D",
        "push {r8}",
    ));
    insns.push((
        Inst::Push {
            reg_list: smallvec![rreg(0), rreg(1), rreg(2), rreg(6), rreg(8)],
        },
        "2DE94701",
        "push {r0, r1, r2, r6, r8}",
    ));
    insns.push((
        Inst::Push {
            reg_list: smallvec![rreg(8), rreg(9), rreg(10)],
        },
        "2DE90007",
        "push {r8, r9, r10}",
    ));
    insns.push((
        Inst::Pop {
            reg_list: smallvec![writable_rreg(0)],
        },
        "5DF8040B",
        "pop {r0}",
    ));
    insns.push((
        Inst::Pop {
            reg_list: smallvec![writable_rreg(8)],
        },
        "5DF8048B",
        "pop {r8}",
    ));
    insns.push((
        Inst::Pop {
            reg_list: smallvec![
                writable_rreg(0),
                writable_rreg(1),
                writable_rreg(2),
                writable_rreg(6),
                writable_rreg(8),
            ],
        },
        "BDE84701",
        "pop {r0, r1, r2, r6, r8}",
    ));
    insns.push((
        Inst::Pop {
            reg_list: smallvec![writable_rreg(8), writable_rreg(9), writable_rreg(10)],
        },
        "BDE80007",
        "pop {r8, r9, r10}",
    ));
    insns.push((
        Inst::Call {
            info: Box::new(CallInfo {
                dest: ExternalName::testcase("test0"),
                uses: Vec::new(),
                defs: Vec::new(),
                loc: SourceLoc::default(),
                opcode: Opcode::Call,
            }),
        },
        "00F000D0",
        "bl 0",
    ));
    insns.push((
        Inst::CallInd {
            info: Box::new(CallIndInfo {
                rm: rreg(0),
                uses: Vec::new(),
                defs: Vec::new(),
                loc: SourceLoc::default(),
                opcode: Opcode::CallIndirect,
            }),
        },
        "8047",
        "blx r0",
    ));
    insns.push((
        Inst::CallInd {
            info: Box::new(CallIndInfo {
                rm: rreg(8),
                uses: Vec::new(),
                defs: Vec::new(),
                loc: SourceLoc::default(),
                opcode: Opcode::CallIndirect,
            }),
        },
        "C047",
        "blx r8",
    ));
    insns.push((Inst::Ret, "7047", "bx lr"));
    insns.push((
        Inst::Jump {
            dest: BranchTarget::ResolvedOffset(32),
        },
        "00F00EB8",
        "b 32",
    ));
    insns.push((
        Inst::Jump {
            dest: BranchTarget::ResolvedOffset(0xfffff4),
        },
        "FFF3F897",
        "b 16777204",
    ));
    insns.push((
        Inst::CondBr {
            taken: BranchTarget::ResolvedOffset(20),
            not_taken: BranchTarget::ResolvedOffset(68),
            cond: Cond::Eq,
        },
        "00F0088000F020B8",
        "beq 20 ; b 68",
    ));
    insns.push((
        Inst::CondBr {
            taken: BranchTarget::ResolvedOffset(6),
            not_taken: BranchTarget::ResolvedOffset(100),
            cond: Cond::Gt,
        },
        "00F3018000F030B8",
        "bgt 6 ; b 100",
    ));
    insns.push((
        Inst::OneWayCondBr {
            target: BranchTarget::ResolvedOffset(4100),
            cond: Cond::Ne,
        },
        "41F00080",
        "bne 4100",
    ));
    insns.push((
        Inst::IndirectBr {
            rm: rreg(0),
            targets: vec![],
        },
        "0047",
        "bx r0",
    ));
    insns.push((
        Inst::IndirectBr {
            rm: rreg(8),
            targets: vec![],
        },
        "4047",
        "bx r8",
    ));

    // ========================================================
    // Run the tests
    let rru = regs::create_reg_universe();
    for (insn, expected_encoding, expected_printing) in insns {
        // Check the printed text is as expected.
        let actual_printing = insn.show_rru(Some(&rru));
        assert_eq!(expected_printing, actual_printing);
        let mut sink = test_utils::TestCodeSink::new();
        let mut buffer = MachBuffer::new();
        insn.emit(&mut buffer, &flags, &mut Default::default());
        let buffer = buffer.finish();
        buffer.emit(&mut sink);
        let actual_encoding = &sink.stringify();
        assert_eq!(expected_encoding, actual_encoding, "{}", expected_printing);
    }
}
