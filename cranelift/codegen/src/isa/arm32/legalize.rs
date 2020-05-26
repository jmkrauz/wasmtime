use crate::cursor::{Cursor, FuncCursor};
use crate::flowgraph::ControlFlowGraph;
use crate::ir::entities::{Block, Value};
use crate::ir::types::Type;
use crate::ir::{self, InstBuilder};
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
        | ir::Opcode::StackLoad
        | ir::Opcode::StackStore
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
        | ir::Opcode::IcmpImm => return Some(arm32_expand),
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

fn expand_udiv_urem(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    use crate::ir::condcodes::IntCC;
    use crate::ir::immediates::Imm64;
    use crate::ir::TrapCode;

    let (n, d, udiv) = match func.dfg[inst] {
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Udiv,
            args,
        } => (args[0], args[1], true),
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Urem,
            args,
        } => (args[0], args[1], false),
        _ => panic!("Expected udiv/urem: {}", func.dfg.display_inst(inst, None)),
    };

    // Shift and subtract algorithm adapted from
    // https://en.wikipedia.org/wiki/Division_algorithm#Integer_division_(unsigned)_with_remainder
    //
    //   trapz d, int_divz
    //   quot = iconst 0
    //   rem = iconst 0
    //   i_0 = iconst ty_bits - 1
    //   j_0 = iconst 1 << (ty_bits - 1)
    //   jump block_loop(quot, rem, i_0, j_0)
    // block_loop(q, r, i, j):
    //   r_1 = ishl_imm r, 1
    //   v0 = band n, j
    //   v1 = ushr v0, i
    //   r_2 = bor r_1, v1
    //   v2 = icmp uge r_2, d
    //   brz v2 block_continue_loop(q, r_2, i, j)
    //   jump block_if
    // block_if():
    //   r_3 = isub r_2, d
    //   q_1 = bor q, j
    //   jump block_continue_loop(q_1, r_3, i, j)
    // block_continue_loop(q_, r_, i_, j_):
    //   i_1 = add_imm i_, -1
    //   j_1 = ushr_imm j_, 1
    //   brz j_1 block_result(q_ || r_)
    //   jump block_loop(q_, r_, i_1, j_1)
    // block_result(quotient || remainder):

    let arg_ty = func.dfg.ctrl_typevar(inst);
    let ty_bits = arg_ty.bits();
    let old_block = func.layout.pp_block(inst);
    let result = func.dfg.first_result(inst);
    func.dfg.clear_results(inst);

    let block_loop = func.dfg.make_block();
    let block_if = func.dfg.make_block();
    let block_continue_loop = func.dfg.make_block();
    let block_result = func.dfg.make_block();

    let q = func.dfg.append_block_param(block_loop, arg_ty);
    let r = func.dfg.append_block_param(block_loop, arg_ty);
    let i = func.dfg.append_block_param(block_loop, arg_ty);
    let j = func.dfg.append_block_param(block_loop, arg_ty);

    let q_ = func.dfg.append_block_param(block_continue_loop, arg_ty);
    let r_ = func.dfg.append_block_param(block_continue_loop, arg_ty);
    let i_ = func.dfg.append_block_param(block_continue_loop, arg_ty);
    let j_ = func.dfg.append_block_param(block_continue_loop, arg_ty);

    func.dfg.attach_block_param(block_result, result);
    let result_param = if udiv { q_ } else { r_ };

    let mut pos = FuncCursor::new(func).after_inst(inst);
    pos.use_srcloc(inst);

    pos.func
        .dfg
        .replace(inst)
        .trapz(d, TrapCode::IntegerDivisionByZero);
    let quot = pos.ins().iconst(arg_ty, Imm64::new(0));
    let rem = pos.ins().iconst(arg_ty, Imm64::new(0));
    let i_0 = pos.ins().iconst(arg_ty, Imm64::new((ty_bits - 1).into()));
    let j_0 = pos
        .ins()
        .iconst(arg_ty, Imm64::new((1u128 << (ty_bits - 1)) as i64));
    let minus_one = pos.ins().iconst(arg_ty, Imm64::new(-1));
    pos.ins().jump(block_loop, &[quot, rem, i_0, j_0]);

    pos.insert_block(block_loop);
    let r_1 = pos.ins().ishl_imm(r, 1);
    let v0 = pos.ins().band(n, j);
    let v1 = pos.ins().ushr(v0, i);
    let r_2 = pos.ins().bor(r_1, v1);
    let v2 = pos.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, r_2, d);
    pos.ins().brz(v2, block_continue_loop, &[q, r_2, i, j]);
    pos.ins().jump(block_if, &[]);

    pos.insert_block(block_if);
    let r_3 = pos.ins().isub(r_2, d);
    let q_1 = pos.ins().bor(q, j);
    pos.ins().jump(block_continue_loop, &[q_1, r_3, i, j]);

    pos.insert_block(block_continue_loop);
    let i_1 = pos.ins().iadd(i_, minus_one);
    let j_1 = pos.ins().ushr_imm(j_, 1);
    pos.ins().brz(j_1, block_result, &[result_param]);
    pos.ins().jump(block_loop, &[q_, r_, i_1, j_1]);

    pos.insert_block(block_result);

    for &block in &[
        old_block,
        block_loop,
        block_if,
        block_continue_loop,
        block_result,
    ] {
        cfg.recompute_block(pos.func, block);
    }
}

fn expand_sdiv_srem(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    use crate::ir::condcodes::IntCC;
    use crate::ir::immediates::Imm64;
    use crate::ir::TrapCode;

    let (n, d, sdiv) = match func.dfg[inst] {
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Sdiv,
            args,
        } => (args[0], args[1], true),
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Srem,
            args,
        } => (args[0], args[1], false),
        _ => panic!("Expected sdiv/srem: {}", func.dfg.display_inst(inst, None)),
    };

    let arg_ty = func.dfg.ctrl_typevar(inst);
    let ty_bits = arg_ty.bits();
    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    if sdiv {
        let _v0 = pos
            .ins()
            .icmp_imm(IntCC::Equal, n, Imm64::new((1u128 << (ty_bits - 1)) as i64));
        let _v1 = pos.ins().icmp_imm(IntCC::Equal, d, Imm64::new(-1));
        let _v2 = pos.ins().band(_v0, _v1);
        pos.ins().trapnz(_v2, TrapCode::IntegerOverflow);
    }

    let bits_minus_one = pos.ins().iconst(arg_ty, Imm64::new((ty_bits - 1).into()));
    let n_mask = pos.ins().sshr(n, bits_minus_one);
    let d_mask = pos.ins().sshr(d, bits_minus_one);
    let v0 = pos.ins().bxor(n_mask, d_mask);
    let v1 = pos.ins().iadd(n, n_mask);
    let v2 = pos.ins().iadd(d, d_mask);
    let n_abs = pos.ins().bxor(v1, n_mask);
    let d_abs = pos.ins().bxor(v2, d_mask);

    let (res_abs, sign) = if sdiv {
        (
            pos.ins().udiv(n_abs, d_abs),
            pos.ins().bor_imm(v0, Imm64::new(1)),
        )
    } else {
        (
            pos.ins().urem(n_abs, d_abs),
            pos.ins().bor_imm(n_mask, Imm64::new(1)),
        )
    };

    pos.func.dfg.replace(inst).imul(res_abs, sign);

    let _block = pos.func.layout.pp_block(inst);
    cfg.recompute_block(pos.func, _block);
}

// The common part of narrowing ishl, ushr and sshr.
fn narrow_shift_common<Else1Fn, Else2Fn>(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    x: Value,
    mut y: Value,
    insert_else_1: &mut Else1Fn,
    insert_else_2: &mut Else2Fn,
) where
    Else1Fn: FnMut(&mut FuncCursor, Block, Value, Value, Value, Value),
    Else2Fn: FnMut(&mut FuncCursor, Block, Value, Value, Type),
{
    use crate::ir::condcodes::IntCC;
    use crate::ir::immediates::Imm64;

    //  y1 = y & small_ty_mask
    //  y2 = y & res_ty_mask
    //
    //  if y1 == 0 {
    //      (zl, zh) = (xl, xh)
    //  } else {
    //      instructions inserted by insert_else_1
    //  }
    //  if y1 == y2 {
    //      (result_lo, result_hi) = (zl, zh)
    //  } else {
    //      instructions inserted by insert_else_2
    //  }

    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    let x_ty = pos.func.dfg.value_type(x);
    let small_ty = x_ty.half_width().expect("Can't narrow shift operation");
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
    pos.ins().brnz(y1, else_1, &[]);
    pos.ins().jump(if_2, &[xl, xh]);

    pos.insert_block(else_1);
    insert_else_1(&mut pos, if_2, xl, xh, y1, y1_compl);

    pos.insert_block(if_2);
    let v = pos.ins().icmp(IntCC::Equal, y1, y2);
    pos.ins().brz(v, else_2, &[]);
    pos.ins().jump(result_block, &[zl, zh]);

    pos.insert_block(else_2);
    insert_else_2(&mut pos, result_block, zl, zh, small_ty);

    pos.insert_block(result_block);
    pos.func.dfg.replace(inst).iconcat(result_lo, result_hi);

    for &block in &[if_1, else_1, if_2, else_2, result_block] {
        cfg.recompute_block(pos.func, block);
    }
}

fn narrow_ishl(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    use crate::ir::immediates::Imm64;

    let (x, y) = match func.dfg[inst] {
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Ishl,
            args,
        } => (args[0], args[1]),
        _ => panic!("Expected ishl: {}", func.dfg.display_inst(inst, None)),
    };

    narrow_shift_common(
        inst,
        func,
        cfg,
        x,
        y,
        &mut |pos: &mut FuncCursor,
              dest: Block,
              xl: Value,
              xh: Value,
              y1: Value,
              y1_compl: Value| {
            let zl = pos.ins().ishl(xl, y1);
            let v0 = pos.ins().ishl(xh, y1);
            let v1 = pos.ins().ushr(xl, y1_compl);
            let zh = pos.ins().bor(v0, v1);
            pos.ins().jump(dest, &[zl, zh]);
        },
        &mut |pos: &mut FuncCursor, dest: Block, zl: Value, _zh: Value, ty: Type| {
            let result_lo = pos.ins().iconst(ty, Imm64::new(0));
            let result_hi = pos.ins().copy(zl);
            pos.ins().jump(dest, &[result_lo, result_hi]);
        },
    );
}

fn narrow_ushr(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    use crate::ir::immediates::Imm64;

    let (x, y) = match func.dfg[inst] {
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Ushr,
            args,
        } => (args[0], args[1]),
        _ => panic!("Expected ushr: {}", func.dfg.display_inst(inst, None)),
    };

    narrow_shift_common(
        inst,
        func,
        cfg,
        x,
        y,
        &mut |pos: &mut FuncCursor,
              dest: Block,
              xl: Value,
              xh: Value,
              y1: Value,
              y1_compl: Value| {
            let v0 = pos.ins().ushr(xl, y1);
            let v1 = pos.ins().ishl(xh, y1_compl);
            let zl = pos.ins().bor(v0, v1);
            let zh = pos.ins().ushr(xh, y1);
            pos.ins().jump(dest, &[zl, zh]);
        },
        &mut |pos: &mut FuncCursor, dest: Block, _zl: Value, zh: Value, ty: Type| {
            let result_lo = pos.ins().copy(zh);
            let result_hi = pos.ins().iconst(ty, Imm64::new(0));
            pos.ins().jump(dest, &[result_lo, result_hi]);
        },
    );
}

fn narrow_sshr(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    use crate::ir::condcodes::IntCC;
    use crate::ir::immediates::Imm64;

    let (x, y) = match func.dfg[inst] {
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Sshr,
            args,
        } => (args[0], args[1]),
        _ => panic!("Expected sshr: {}", func.dfg.display_inst(inst, None)),
    };

    narrow_shift_common(
        inst,
        func,
        cfg,
        x,
        y,
        &mut |pos: &mut FuncCursor,
              dest: Block,
              xl: Value,
              xh: Value,
              y1: Value,
              y1_compl: Value| {
            let v0 = pos.ins().ushr(xl, y1);
            let v1 = pos.ins().ishl(xh, y1_compl);
            let zl = pos.ins().bor(v0, v1);
            let zh = pos.ins().sshr(xh, y1);
            pos.ins().jump(dest, &[zl, zh]);
        },
        &mut |pos: &mut FuncCursor, dest: Block, _zl: Value, zh: Value, ty: Type| {
            let result_lo = pos.ins().copy(zh);
            let v0 = pos
                .ins()
                .icmp_imm(IntCC::SignedGreaterThanOrEqual, zh, Imm64::new(0)); // zh >= 0 <=> x >= 0
            let v1 = pos.ins().bint(ty, v0);
            let result_hi = pos.ins().iadd_imm(v1, Imm64::new(-1)); // result_hi = if x >= 0 then 0 else -1
            pos.ins().jump(dest, &[result_lo, result_hi]);
        },
    );
}

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
