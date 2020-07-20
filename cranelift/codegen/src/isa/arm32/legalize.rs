#![allow(dead_code)]

use crate::cursor::{Cursor, FuncCursor};
use crate::flowgraph::ControlFlowGraph;
use crate::ir::entities::Value;
use crate::ir::{self, InstBuilder, ValueLoc};
use crate::isa::{self, TargetIsa};

pub fn legalize_inst(
    _func: &ir::Function,
    inst: &ir::InstructionData,
    ctrl_typevar: ir::Type,
) -> Option<isa::Legalize> {
    match inst.opcode() {
        ir::Opcode::BrIcmp
        | ir::Opcode::GlobalValue
        | ir::Opcode::HeapAddr
        | ir::Opcode::TableAddr
        | ir::Opcode::Trapnz
        | ir::Opcode::Trapz
        | ir::Opcode::BandImm
        | ir::Opcode::BorImm
        | ir::Opcode::BxorImm
        | ir::Opcode::IaddImm
        | ir::Opcode::IfcmpImm
        | ir::Opcode::ImulImm
        | ir::Opcode::IrsubImm
        | ir::Opcode::IshlImm
        | ir::Opcode::RotlImm
        | ir::Opcode::RotrImm
        | ir::Opcode::SdivImm
        | ir::Opcode::SremImm
        | ir::Opcode::SshrImm
        | ir::Opcode::UdivImm
        | ir::Opcode::UremImm
        | ir::Opcode::UshrImm
        | ir::Opcode::IcmpImm
        | ir::Opcode::Spill
        | ir::Opcode::Uload32
        | ir::Opcode::Sload32
        | ir::Opcode::Istore32
        | ir::Opcode::Ireduce => return Some(arm32_expand),
        op if op.is_ghost() => return None,
        _ => {}
    }
    if ctrl_typevar == ir::types::I64 {
        Some(arm32_narrow)
    } else {
        None
    }
}

include!(concat!(env!("OUT_DIR"), "/legalize-arm32.rs"));

fn narrow_rotation_common<F>(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    x: Value,
    mut y: Value,
    insert_else_2: &mut F,
) where
    F: FnMut(&mut FuncCursor, Value, Value, Value, Value) -> (Value, Value),
{
    use crate::ir::condcodes::IntCC;
    use crate::ir::immediates::Imm64;

    //  y1 = y & small_ty_mask;
    //  y2 = y & res_ty_mask;
    //
    //  if y1 == y2 {
    //      (zl, zh) = (xl, xh)
    //  } else {
    //      (zl, zh) = (xh, xl)
    //  }
    //  if y1 == 0 {
    //      (result_lo, result_hi) = (zl, zh)
    //  } else {
    //      instructions inserted by compute_result
    //  }

    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    let x_ty = pos.func.dfg.value_type(x);
    let small_ty = x_ty.half_width().expect("Can't narrow rotation operation");
    let y_ty = pos.func.dfg.value_type(y);
    let x_ty_mask = Imm64::new((x_ty.bits() - 1).into());
    let small_ty_mask = Imm64::new((small_ty.bits() - 1).into());

    let if_1 = pos.func.layout.pp_block(inst); // old block
    let else_1 = pos.func.dfg.make_block();
    let if_2 = pos.func.dfg.make_block();
    let else_2 = pos.func.dfg.make_block();
    let result_block = pos.func.dfg.make_block();

    let zl = pos.func.dfg.append_block_param(if_2, small_ty);
    let zh = pos.func.dfg.append_block_param(if_2, small_ty);
    let result_lo = pos.func.dfg.append_block_param(result_block, small_ty);
    let result_hi = pos.func.dfg.append_block_param(result_block, small_ty);

    if y_ty != small_ty {
        y = pos.ins().ireduce(small_ty, y);
    }

    let small_ty_bits = pos
        .ins()
        .iconst(small_ty, Imm64::new(small_ty.bits().into()));
    let y1 = pos.ins().band_imm(y, small_ty_mask);
    let y2 = pos.ins().band_imm(y, x_ty_mask);
    let y1_compl = pos.ins().isub(small_ty_bits, y1);
    let (xl, xh) = pos.ins().isplit(x);

    let v0 = pos.ins().icmp(IntCC::Equal, y1, y2);
    pos.ins().brz(v0, else_1, &[]);
    pos.ins().jump(if_2, &[xl, xh]);

    pos.insert_block(else_1);
    pos.ins().jump(if_2, &[xh, xl]);

    pos.insert_block(if_2);
    let v1 = pos.ins().icmp_imm(IntCC::Equal, y1, Imm64::new(0));
    pos.ins().brz(v1, else_2, &[]);
    pos.ins().jump(result_block, &[zl, zh]);

    pos.insert_block(else_2);
    let (res_lo, res_hi) = insert_else_2(&mut pos, zl, zh, y1, y1_compl);
    pos.ins().jump(result_block, &[res_lo, res_hi]);

    pos.insert_block(result_block);
    pos.func.dfg.replace(inst).iconcat(result_lo, result_hi);

    for &block in &[if_1, else_1, if_2, else_2, result_block] {
        cfg.recompute_block(pos.func, block);
    }
}

fn narrow_rotl(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    let (x, y) = match func.dfg[inst] {
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Rotl,
            args,
        } => (args[0], args[1]),
        _ => panic!("Expected rotl: {}", func.dfg.display_inst(inst, None)),
    };

    narrow_rotation_common(
        inst,
        func,
        cfg,
        x,
        y,
        &mut |pos: &mut FuncCursor, zl: Value, zh: Value, y1: Value, y1_compl: Value| {
            let v0 = pos.ins().ishl(zl, y1);
            let v1 = pos.ins().ushr(zh, y1_compl);
            let res_lo = pos.ins().bor(v0, v1);

            let v2 = pos.ins().ishl(zh, y1);
            let v3 = pos.ins().ushr(zl, y1_compl);
            let res_hi = pos.ins().bor(v2, v3);

            return (res_lo, res_hi);
        },
    );
}

fn narrow_rotr(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    let (x, y) = match func.dfg[inst] {
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Rotr,
            args,
        } => (args[0], args[1]),
        _ => panic!("Expected rotr: {}", func.dfg.display_inst(inst, None)),
    };

    narrow_rotation_common(
        inst,
        func,
        cfg,
        x,
        y,
        &mut |pos: &mut FuncCursor, zl: Value, zh: Value, y1: Value, y1_compl: Value| {
            let v0 = pos.ins().ushr(zl, y1);
            let v1 = pos.ins().ishl(zh, y1_compl);
            let res_lo = pos.ins().bor(v0, v1);

            let v2 = pos.ins().ushr(zh, y1);
            let v3 = pos.ins().ishl(zl, y1_compl);
            let res_hi = pos.ins().bor(v2, v3);

            return (res_lo, res_hi);
        },
    );
}

fn expand_spill(
    inst: ir::Inst,
    func: &mut ir::Function,
    _cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    let x = match func.dfg[inst] {
        ir::InstructionData::Unary {
            opcode: ir::Opcode::Spill,
            arg,
        } => arg,
        _ => panic!("Expected spill: {}", func.dfg.display_inst(inst, None)),
    };
    let result = func.dfg.first_result(inst);
    let loc = func.locations[result];
    let ss = match loc {
        ValueLoc::Stack(ss) => ss,
        _ => panic!("Expected stack location: {:?}", loc),
    };

    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    pos.ins().stack_store(x, ss, 0);

    pos.func.dfg.clear_results(inst);
    pos.remove_inst();
}
