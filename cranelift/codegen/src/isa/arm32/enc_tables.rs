//! Encoding tables for ARM32 ISA.

use crate::cursor::{Cursor, FuncCursor};
use crate::flowgraph::ControlFlowGraph;
use crate::ir::InstBuilder;
use super::registers::*;
use crate::ir::{self, Function, Inst, InstructionData, Opcode};
use crate::isa::constraints::*;
use crate::bitset::BitSet;
use crate::isa::enc_tables::*;
use crate::isa::encoding::{base_size, Encoding, RecipeSizing};
use crate::isa::{self, StackBaseMask, StackRef, TargetIsa};
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

fn expand_fcvt_from_uint64(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    arg: ir::Value,
    res_ty: ir::Type,
) {
    use crate::ir::types::{I32, I64, F32};
    use crate::ir::condcodes::{IntCC, FloatCC};
    use crate::ir::immediates::{Ieee32, Ieee64, Imm64};

    let int_ty = if res_ty == F32 {
        I32
    } else {
        I64
    };
    let old_block = func.layout.pp_block(inst);
    let big_arg_block = func.dfg.make_block();
    let compare_results_block = func.dfg.make_block();
    let result_block = func.dfg.make_block();

    let result = func.dfg.first_result(inst);
    func.dfg.clear_results(inst);
    func.dfg.attach_block_param(result_block, result);

    let (argl, argh) = func.dfg.replace(inst).isplit(arg);
    let mut pos = FuncCursor::new(func).after_inst(inst);
    pos.use_srcloc(inst);

    let xl = pos.ins().fcvt_from_uint(res_ty, argl);
    let v0 = pos.ins().icmp_imm(IntCC::Equal, argh, Imm64::new(0));
    pos.ins().brnz(v0, result_block, &[xl]);
    pos.ins().jump(big_arg_block, &[]);

    pos.insert_block(big_arg_block);
    let xh = pos.ins().fcvt_from_uint(res_ty, argh);
    let two_pow_32 = if res_ty == F32 {
        pos.ins().f32const("0x1.0p+32".parse::<Ieee32>().unwrap())
    } else {
        pos.ins().f64const("0x1.0p+32".parse::<Ieee64>().unwrap())
    };
    let res_lo = pos.ins().fma(xh, two_pow_32, xl);     // res_lo may be too small, so we need to
    let v1 = pos.ins().bitcast(int_ty, res_lo);         // find out if next float is better approx.
    let v2 = pos.ins().iadd_imm(v1, Imm64::new(1));
    let res_hi = pos.ins().bitcast(res_ty, v2);
    let two_pow_64 = if res_ty == F32 {
        pos.ins().f32const("0x1.0p+64".parse::<Ieee32>().unwrap())
    } else {
        pos.ins().f64const("0x1.0p+64".parse::<Ieee64>().unwrap())
    };
    let is_too_large = pos.ins().fcmp(FloatCC::GreaterThanOrEqual, res_hi, two_pow_64);
    pos.ins().brnz(is_too_large, result_block, &[res_lo]);
    pos.ins().jump(compare_results_block, &[]);

    pos.insert_block(compare_results_block);
    let i_lo = pos.ins().fcvt_to_uint(I64, res_lo);
    let i_hi = pos.ins().fcvt_to_uint(I64, res_hi);
    let v3 = pos.ins().icmp(IntCC::UnsignedLessThanOrEqual, arg, i_lo);
    let diff_lo = pos.ins().isub(arg, i_lo);
    let diff_hi = pos.ins().isub(i_hi, arg);
    let v4 = pos.ins().icmp(IntCC::UnsignedLessThanOrEqual, diff_lo, diff_hi);
    let v5 = pos.ins().bor(v3, v4);
    pos.ins().brnz(v5, result_block, &[res_lo]);    // res_lo is fine
    pos.ins().jump(result_block, &[res_hi]);        // res_hi is better approximation
    pos.insert_block(result_block);

    cfg.recompute_block(pos.func, old_block);
    cfg.recompute_block(pos.func, big_arg_block);
    cfg.recompute_block(pos.func, compare_results_block);
    cfg.recompute_block(pos.func, result_block);
}

fn expand_fcvt_from_uint(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    use crate::ir::types::{I32, I64, F32};

    let arg = match func.dfg[inst] {
        ir::InstructionData::Unary {
            opcode: ir::Opcode::FcvtFromUint,
            arg,
        } => arg,
        _ => panic!("Expected fcvt_from_uint: {}", func.dfg.display_inst(inst, None)),
    };

    let res_ty = func.dfg.ctrl_typevar(inst);
    let arg_ty = func.dfg.value_type(arg);

    if arg_ty == I32 {  // easy case
        let mut pos = FuncCursor::new(func).at_inst(inst);
        pos.use_srcloc(inst);

        let v0 = pos.ins().bitcast(F32, arg);
        pos.func.dfg.replace(inst).arm32_vcvt_uint2f(res_ty, v0);
    } else if arg_ty == I64 {
        expand_fcvt_from_uint64(inst, func, cfg, arg, res_ty);
    } else {
        panic!("Expected I32/I64, got: {}", arg_ty);
    }
}
