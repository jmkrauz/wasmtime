//! Encoding tables for ARM32 ISA.

use super::registers::*;
use crate::ir::{self, Function, Inst, InstructionData, Opcode};
use crate::isa::constraints::*;
use crate::isa::enc_tables::*;
use crate::isa::encoding::{base_size, Encoding, RecipeSizing};
use crate::isa::{self, StackBaseMask, StackRef};
use crate::legalizer::isplit;
use crate::predicates;
use crate::regalloc::RegDiversions;

include!(concat!(env!("OUT_DIR"), "/encoding-arm32.rs"));
include!(concat!(env!("OUT_DIR"), "/legalize-arm32.rs"));

const MOV32_BYTES: u8 = 8;

fn get_ss_offset(inst: Inst, divert: &RegDiversions, func: &Function) -> i32 {
    let inst_data = func.dfg[inst].clone();

    let stk_option: Option<StackRef> = match inst_data.opcode() {
        Opcode::Spill => {
            let results = [func.dfg.first_result(inst)];
            Some(
                StackRef::masked(
                    divert.stack(results[0], &func.locations),
                    StackBaseMask(1),
                    &func.stack_slots,
                )
                .unwrap(),
            )
        }
        Opcode::Fill => {
            if let InstructionData::Unary { arg, .. } = func.dfg[inst] {
                let args = [arg];
                Some(
                    StackRef::masked(
                        divert.stack(args[0], &func.locations),
                        StackBaseMask(1),
                        &func.stack_slots,
                    )
                    .unwrap(),
                )
            } else {
                None
            }
        }
        Opcode::Regspill => {
            if let InstructionData::RegSpill { dst, .. } = func.dfg[inst] {
                Some(StackRef::sp(dst, &func.stack_slots))
            } else {
                None
            }
        }
        Opcode::Regfill => {
            if let InstructionData::RegFill { src, .. } = func.dfg[inst] {
                Some(StackRef::sp(src, &func.stack_slots))
            } else {
                None
            }
        }
        _ => None,
    };

    if let Some(stk) = stk_option {
        return stk.offset;
    }

    panic!("get_ss_offset: inst opcod {:?}", inst_data.opcode());
}

// Used by fill, spill, regspill and regfill recipes.
fn size_plus_maybe_mov_stack_offset_bytes(
    sizing: &RecipeSizing,
    _enc: Encoding,
    inst: Inst,
    divert: &RegDiversions,
    func: &Function,
) -> u8 {
    let mut result: u8 = sizing.base_size;

    if !predicates::is_unsigned_int(get_ss_offset(inst, divert, func), 8, 0) {
        result += MOV32_BYTES;
    }

    result
}
