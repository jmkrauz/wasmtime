//! Legalize instructions.
//!
//! A legal instruction is one that can be mapped directly to a machine code instruction for the
//! target ISA. The `legalize_function()` function takes as input any function and transforms it
//! into an equivalent function using only legal instructions.
//!
//! The characteristics of legal instructions depend on the target ISA, so any given instruction
//! can be legal for one ISA and illegal for another.
//!
//! Besides transforming instructions, the legalizer also fills out the `function.encodings` map
//! which provides a legal encoding recipe for every instruction.
//!
//! The legalizer does not deal with register allocation constraints. These constraints are derived
//! from the encoding recipes, and solved later by the register allocator.

use crate::bitset::BitSet;
use crate::cursor::{Cursor, FuncCursor};
use crate::flowgraph::ControlFlowGraph;
use crate::ir::entities::{Block, Value};
use crate::ir::types::{Type, F32, F64, I32, I64};
use crate::ir::{self, InstBuilder, MemFlags};
use crate::isa::TargetIsa;
use crate::predicates;
use crate::timing;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

mod boundary;
mod call;
mod globalvalue;
mod heap;
mod libcall;
mod split;
mod table;

use self::call::expand_call;
use self::globalvalue::expand_global_value;
use self::heap::expand_heap_addr;
use self::libcall::expand_as_libcall;
pub use self::split::isplit;
use self::table::expand_table_addr;

enum LegalizeInstResult {
    Done,
    Legalized,
    SplitLegalizePending,
}

/// Legalize `inst` for `isa`.
fn legalize_inst(
    inst: ir::Inst,
    pos: &mut FuncCursor,
    cfg: &mut ControlFlowGraph,
    isa: &dyn TargetIsa,
) -> LegalizeInstResult {
    let opcode = pos.func.dfg[inst].opcode();

    // Check for ABI boundaries that need to be converted to the legalized signature.
    if opcode.is_call() {
        if boundary::handle_call_abi(isa, inst, pos.func, cfg) {
            return LegalizeInstResult::Legalized;
        }
    } else if opcode.is_return() {
        if boundary::handle_return_abi(inst, pos.func, cfg) {
            return LegalizeInstResult::Legalized;
        }
    } else if opcode.is_branch() {
        split::simplify_branch_arguments(&mut pos.func.dfg, inst);
    } else if opcode == ir::Opcode::Isplit {
        pos.use_srcloc(inst);

        let arg = match pos.func.dfg[inst] {
            ir::InstructionData::Unary { arg, .. } => pos.func.dfg.resolve_aliases(arg),
            _ => panic!("Expected isplit: {}", pos.func.dfg.display_inst(inst, None)),
        };

        match pos.func.dfg.value_def(arg) {
            ir::ValueDef::Result(inst, _num) => {
                if let ir::InstructionData::Binary {
                    opcode: ir::Opcode::Iconcat,
                    ..
                } = pos.func.dfg[inst]
                {
                    // `arg` was created by an `iconcat` instruction.
                } else {
                    // `arg` was not created by an `iconcat` instruction. Don't try to resolve it,
                    // as otherwise `split::isplit` will re-insert the original `isplit`, causing
                    // an endless loop.
                    return LegalizeInstResult::SplitLegalizePending;
                }
            }
            ir::ValueDef::Param(_block, _num) => {}
        }

        let res = pos.func.dfg.inst_results(inst).to_vec();
        assert_eq!(res.len(), 2);
        let (resl, resh) = (res[0], res[1]); // Prevent borrowck error

        // Remove old isplit
        pos.func.dfg.clear_results(inst);
        pos.remove_inst();

        let curpos = pos.position();
        let srcloc = pos.srcloc();
        let (xl, xh) = split::isplit(pos.func, cfg, curpos, srcloc, arg);

        pos.func.dfg.change_to_alias(resl, xl);
        pos.func.dfg.change_to_alias(resh, xh);

        return LegalizeInstResult::Legalized;
    }

    match pos.func.update_encoding(inst, isa) {
        Ok(()) => LegalizeInstResult::Done,
        Err(action) => {
            // We should transform the instruction into legal equivalents.
            // If the current instruction was replaced, we need to double back and revisit
            // the expanded sequence. This is both to assign encodings and possible to
            // expand further.
            // There's a risk of infinite looping here if the legalization patterns are
            // unsound. Should we attempt to detect that?
            if action(inst, pos.func, cfg, isa) {
                return LegalizeInstResult::Legalized;
            }

            // We don't have any pattern expansion for this instruction either.
            // Try converting it to a library call as a last resort.
            if expand_as_libcall(inst, pos.func, isa) {
                LegalizeInstResult::Legalized
            } else {
                LegalizeInstResult::Done
            }
        }
    }
}

/// Legalize `func` for `isa`.
///
/// - Transform any instructions that don't have a legal representation in `isa`.
/// - Fill out `func.encodings`.
///
pub fn legalize_function(func: &mut ir::Function, cfg: &mut ControlFlowGraph, isa: &dyn TargetIsa) {
    let _tt = timing::legalize();
    debug_assert!(cfg.is_valid());

    boundary::legalize_signatures(func, isa);

    func.encodings.resize(func.dfg.num_insts());

    let mut pos = FuncCursor::new(func);
    let func_begin = pos.position();

    // Split block params before trying to legalize instructions, so that the newly introduced
    // isplit instructions get legalized.
    while let Some(block) = pos.next_block() {
        split::split_block_params(pos.func, cfg, block);
    }

    pos.set_position(func_begin);

    // This must be a set to prevent trying to legalize `isplit` and `vsplit` twice in certain cases.
    let mut pending_splits = BTreeSet::new();

    // Process blocks in layout order. Some legalization actions may split the current block or append
    // new ones to the end. We need to make sure we visit those new blocks too.
    while let Some(_block) = pos.next_block() {
        // Keep track of the cursor position before the instruction being processed, so we can
        // double back when replacing instructions.
        let mut prev_pos = pos.position();

        while let Some(inst) = pos.next_inst() {
            match legalize_inst(inst, &mut pos, cfg, isa) {
                // Remember this position in case we need to double back.
                LegalizeInstResult::Done => prev_pos = pos.position(),

                // Go back and legalize the inserted return value conversion instructions.
                LegalizeInstResult::Legalized => pos.set_position(prev_pos),

                // The argument of a `isplit` or `vsplit` instruction didn't resolve to a
                // `iconcat` or `vconcat` instruction. Try again after legalizing the rest of
                // the instructions.
                LegalizeInstResult::SplitLegalizePending => {
                    pending_splits.insert(inst);
                }
            }
        }
    }

    // Try legalizing `isplit` and `vsplit` instructions, which could not previously be legalized.
    for inst in pending_splits {
        pos.goto_inst(inst);
        legalize_inst(inst, &mut pos, cfg, isa);
    }

    // Now that we've lowered all br_tables, we don't need the jump tables anymore.
    if !isa.flags().enable_jump_tables() {
        pos.func.jump_tables.clear();
    }
}

// Include legalization patterns that were generated by `gen_legalizer.rs` from the
// `TransformGroup` in `cranelift-codegen/meta/shared/legalize.rs`.
//
// Concretely, this defines private functions `narrow()`, and `expand()`.
include!(concat!(env!("OUT_DIR"), "/legalizer.rs"));

/// Custom expansion for conditional trap instructions.
/// TODO: Add CFG support to the Rust DSL patterns so we won't have to do this.
fn expand_cond_trap(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    // Parse the instruction.
    let trapz;
    let (arg, code) = match func.dfg[inst] {
        ir::InstructionData::CondTrap { opcode, arg, code } => {
            // We want to branch *over* an unconditional trap.
            trapz = match opcode {
                ir::Opcode::Trapz => true,
                ir::Opcode::Trapnz => false,
                _ => panic!("Expected cond trap: {}", func.dfg.display_inst(inst, None)),
            };
            (arg, code)
        }
        _ => panic!("Expected cond trap: {}", func.dfg.display_inst(inst, None)),
    };

    // Split the block after `inst`:
    //
    //     trapnz arg
    //     ..
    //
    // Becomes:
    //
    //     brz arg, new_block_resume
    //     jump new_block_trap
    //
    //   new_block_trap:
    //     trap
    //
    //   new_block_resume:
    //     ..
    let old_block = func.layout.pp_block(inst);
    let new_block_trap = func.dfg.make_block();
    let new_block_resume = func.dfg.make_block();

    // Replace trap instruction by the inverted condition.
    if trapz {
        func.dfg.replace(inst).brnz(arg, new_block_resume, &[]);
    } else {
        func.dfg.replace(inst).brz(arg, new_block_resume, &[]);
    }

    // Add jump instruction after the inverted branch.
    let mut pos = FuncCursor::new(func).after_inst(inst);
    pos.use_srcloc(inst);
    pos.ins().jump(new_block_trap, &[]);

    // Insert the new label and the unconditional trap terminator.
    pos.insert_block(new_block_trap);
    pos.ins().trap(code);

    // Insert the new label and resume the execution when the trap fails.
    pos.insert_block(new_block_resume);

    // Finally update the CFG.
    cfg.recompute_block(pos.func, old_block);
    cfg.recompute_block(pos.func, new_block_resume);
    cfg.recompute_block(pos.func, new_block_trap);
}

/// Jump tables.
fn expand_br_table(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    isa: &dyn TargetIsa,
) {
    if isa.flags().enable_jump_tables() {
        expand_br_table_jt(inst, func, cfg, isa);
    } else {
        expand_br_table_conds(inst, func, cfg, isa);
    }
}

/// Expand br_table to jump table.
fn expand_br_table_jt(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    isa: &dyn TargetIsa,
) {
    use crate::ir::condcodes::IntCC;

    let (arg, default_block, table) = match func.dfg[inst] {
        ir::InstructionData::BranchTable {
            opcode: ir::Opcode::BrTable,
            arg,
            destination,
            table,
        } => (arg, destination, table),
        _ => panic!("Expected br_table: {}", func.dfg.display_inst(inst, None)),
    };

    // Rewrite:
    //
    //     br_table $idx, default_block, $jt
    //
    // To:
    //
    //     $oob = ifcmp_imm $idx, len($jt)
    //     brif uge $oob, default_block
    //     jump fallthrough_block
    //
    //   fallthrough_block:
    //     $base = jump_table_base.i64 $jt
    //     $rel_addr = jump_table_entry.i64 $idx, $base, 4, $jt
    //     $addr = iadd $base, $rel_addr
    //     indirect_jump_table_br $addr, $jt

    let block = func.layout.pp_block(inst);
    let jump_table_block = func.dfg.make_block();

    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    // Bounds check.
    let table_size = pos.func.jump_tables[table].len() as i64;
    let oob = pos
        .ins()
        .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, arg, table_size);

    pos.ins().brnz(oob, default_block, &[]);
    pos.ins().jump(jump_table_block, &[]);
    pos.insert_block(jump_table_block);

    let addr_ty = isa.pointer_type();

    let arg = if pos.func.dfg.value_type(arg) == addr_ty {
        arg
    } else {
        pos.ins().uextend(addr_ty, arg)
    };

    let base_addr = pos.ins().jump_table_base(addr_ty, table);
    let entry = pos
        .ins()
        .jump_table_entry(arg, base_addr, I32.bytes() as u8, table);

    let addr = pos.ins().iadd(base_addr, entry);
    pos.ins().indirect_jump_table_br(addr, table);

    pos.remove_inst();
    cfg.recompute_block(pos.func, block);
    cfg.recompute_block(pos.func, jump_table_block);
}

/// Expand br_table to series of conditionals.
fn expand_br_table_conds(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    use crate::ir::condcodes::IntCC;

    let (arg, default_block, table) = match func.dfg[inst] {
        ir::InstructionData::BranchTable {
            opcode: ir::Opcode::BrTable,
            arg,
            destination,
            table,
        } => (arg, destination, table),
        _ => panic!("Expected br_table: {}", func.dfg.display_inst(inst, None)),
    };

    let block = func.layout.pp_block(inst);

    // This is a poor man's jump table using just a sequence of conditional branches.
    let table_size = func.jump_tables[table].len();
    let mut cond_failed_block = vec![];
    if table_size >= 1 {
        cond_failed_block = alloc::vec::Vec::with_capacity(table_size - 1);
        for _ in 0..table_size - 1 {
            cond_failed_block.push(func.dfg.make_block());
        }
    }

    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    // Ignore the lint for this loop as the range needs to be 0 to table_size
    #[allow(clippy::needless_range_loop)]
    for i in 0..table_size {
        let dest = pos.func.jump_tables[table].as_slice()[i];
        let t = pos.ins().icmp_imm(IntCC::Equal, arg, i as i64);
        pos.ins().brnz(t, dest, &[]);
        // Jump to the next case.
        if i < table_size - 1 {
            let block = cond_failed_block[i];
            pos.ins().jump(block, &[]);
            pos.insert_block(block);
        }
    }

    // `br_table` jumps to the default destination if nothing matches
    pos.ins().jump(default_block, &[]);

    pos.remove_inst();
    cfg.recompute_block(pos.func, block);
    for failed_block in cond_failed_block.into_iter() {
        cfg.recompute_block(pos.func, failed_block);
    }
}

/// Expand the select instruction.
///
/// Conditional moves are available in some ISAs for some register classes. The remaining selects
/// are handled by a branch.
fn expand_select(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    let (ctrl, tval, fval) = match func.dfg[inst] {
        ir::InstructionData::Ternary {
            opcode: ir::Opcode::Select,
            args,
        } => (args[0], args[1], args[2]),
        _ => panic!("Expected select: {}", func.dfg.display_inst(inst, None)),
    };

    // Replace `result = select ctrl, tval, fval` with:
    //
    //   brnz ctrl, new_block(tval)
    //   jump new_block(fval)
    // new_block(result):
    let old_block = func.layout.pp_block(inst);
    let result = func.dfg.first_result(inst);
    func.dfg.clear_results(inst);
    let new_block = func.dfg.make_block();
    func.dfg.attach_block_param(new_block, result);

    func.dfg.replace(inst).brnz(ctrl, new_block, &[tval]);
    let mut pos = FuncCursor::new(func).after_inst(inst);
    pos.use_srcloc(inst);
    pos.ins().jump(new_block, &[fval]);
    pos.insert_block(new_block);

    cfg.recompute_block(pos.func, new_block);
    cfg.recompute_block(pos.func, old_block);
}

fn expand_br_icmp(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    let (cond, a, b, destination, block_args) = match func.dfg[inst] {
        ir::InstructionData::BranchIcmp {
            cond,
            destination,
            ref args,
            ..
        } => (
            cond,
            args.get(0, &func.dfg.value_lists).unwrap(),
            args.get(1, &func.dfg.value_lists).unwrap(),
            destination,
            args.as_slice(&func.dfg.value_lists)[2..].to_vec(),
        ),
        _ => panic!("Expected br_icmp {}", func.dfg.display_inst(inst, None)),
    };

    let old_block = func.layout.pp_block(inst);
    func.dfg.clear_results(inst);

    let icmp_res = func.dfg.replace(inst).icmp(cond, a, b);
    let mut pos = FuncCursor::new(func).after_inst(inst);
    pos.use_srcloc(inst);
    pos.ins().brnz(icmp_res, destination, &block_args);

    cfg.recompute_block(pos.func, destination);
    cfg.recompute_block(pos.func, old_block);
}

/// Expand illegal `f32const` and `f64const` instructions.
fn expand_fconst(
    inst: ir::Inst,
    func: &mut ir::Function,
    _cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    let ty = func.dfg.value_type(func.dfg.first_result(inst));
    debug_assert!(!ty.is_vector(), "Only scalar fconst supported: {}", ty);

    // In the future, we may want to generate constant pool entries for these constants, but for
    // now use an `iconst` and a bit cast.
    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);
    let ival = match pos.func.dfg[inst] {
        ir::InstructionData::UnaryIeee32 {
            opcode: ir::Opcode::F32const,
            imm,
        } => pos.ins().iconst(ir::types::I32, i64::from(imm.bits())),
        ir::InstructionData::UnaryIeee64 {
            opcode: ir::Opcode::F64const,
            imm,
        } => pos.ins().iconst(ir::types::I64, imm.bits() as i64),
        _ => panic!("Expected fconst: {}", pos.func.dfg.display_inst(inst, None)),
    };
    pos.func.dfg.replace(inst).bitcast(ty, ival);
}

/// Expand illegal `stack_load` instructions.
fn expand_stack_load(
    inst: ir::Inst,
    func: &mut ir::Function,
    _cfg: &mut ControlFlowGraph,
    isa: &dyn TargetIsa,
) {
    let ty = func.dfg.value_type(func.dfg.first_result(inst));
    let addr_ty = isa.pointer_type();

    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    let (stack_slot, offset) = match pos.func.dfg[inst] {
        ir::InstructionData::StackLoad {
            opcode: _opcode,
            stack_slot,
            offset,
        } => (stack_slot, offset),
        _ => panic!(
            "Expected stack_load: {}",
            pos.func.dfg.display_inst(inst, None)
        ),
    };

    let addr = pos.ins().stack_addr(addr_ty, stack_slot, offset);

    // Stack slots are required to be accessible and aligned.
    let mflags = MemFlags::trusted();
    pos.func.dfg.replace(inst).load(ty, mflags, addr, 0);
}

/// Expand illegal `stack_store` instructions.
fn expand_stack_store(
    inst: ir::Inst,
    func: &mut ir::Function,
    _cfg: &mut ControlFlowGraph,
    isa: &dyn TargetIsa,
) {
    let addr_ty = isa.pointer_type();

    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    let (val, stack_slot, offset) = match pos.func.dfg[inst] {
        ir::InstructionData::StackStore {
            opcode: _opcode,
            arg,
            stack_slot,
            offset,
        } => (arg, stack_slot, offset),
        _ => panic!(
            "Expected stack_store: {}",
            pos.func.dfg.display_inst(inst, None)
        ),
    };

    let addr = pos.ins().stack_addr(addr_ty, stack_slot, offset);

    let mut mflags = MemFlags::new();
    // Stack slots are required to be accessible and aligned.
    mflags.set_notrap();
    mflags.set_aligned();
    pos.func.dfg.replace(inst).store(mflags, val, addr, 0);
}

/// Split a load into two parts before `iconcat`ing the result together.
fn narrow_load(
    inst: ir::Inst,
    func: &mut ir::Function,
    _cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    let (ptr, offset, flags) = match pos.func.dfg[inst] {
        ir::InstructionData::Load {
            opcode: ir::Opcode::Load,
            arg,
            offset,
            flags,
        } => (arg, offset, flags),
        _ => panic!("Expected load: {}", pos.func.dfg.display_inst(inst, None)),
    };

    let res_ty = pos.func.dfg.ctrl_typevar(inst);
    let small_ty = res_ty.half_width().expect("Can't narrow load");
    let off_step = small_ty.bytes();

    let al = pos.ins().load(small_ty, flags, ptr, offset);

    let ah = match offset.try_add_i64(off_step.into()) {
        Some(offset) => pos.ins().load(small_ty, flags, ptr, offset),
        None => {
            let off_step = pos.ins().iconst(small_ty, off_step as i64);
            let (new_ptr, ovf) = pos.ins().iadd_cout(ptr, off_step);
            pos.ins().trapnz(ovf, ir::TrapCode::HeapOutOfBounds);
            pos.ins().load(small_ty, flags, new_ptr, offset)
        }
    };
    pos.func.dfg.replace(inst).iconcat(al, ah);
}

/// Split a store into two parts after `isplit`ing the value.
fn narrow_store(
    inst: ir::Inst,
    func: &mut ir::Function,
    _cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    let (val, ptr, offset, flags) = match pos.func.dfg[inst] {
        ir::InstructionData::Store {
            opcode: ir::Opcode::Store,
            args,
            offset,
            flags,
        } => (args[0], args[1], offset, flags),
        _ => panic!("Expected store: {}", pos.func.dfg.display_inst(inst, None)),
    };

    let arg_ty = pos.func.dfg.ctrl_typevar(inst);
    let small_ty = arg_ty.half_width().expect("Can't narrow store");
    let off_step = small_ty.bytes();

    let (al, ah) = pos.ins().isplit(val);
    pos.ins().store(flags, al, ptr, offset);

    match offset.try_add_i64(off_step.into()) {
        Some(offset) => {
            pos.ins().store(flags, ah, ptr, offset);
        }
        None => {
            let off_step = pos.ins().iconst(small_ty, off_step as i64);
            let (new_ptr, ovf) = pos.ins().iadd_cout(ptr, off_step);
            pos.ins().trapnz(ovf, ir::TrapCode::HeapOutOfBounds);
            pos.ins().store(flags, ah, new_ptr, offset);
        }
    }
    pos.remove_inst();
}

/// Expands an illegal iconst value by splitting it into two.
fn narrow_iconst(
    inst: ir::Inst,
    func: &mut ir::Function,
    _cfg: &mut ControlFlowGraph,
    isa: &dyn TargetIsa,
) {
    let imm: i64 = if let ir::InstructionData::UnaryImm {
        opcode: ir::Opcode::Iconst,
        imm,
    } = &func.dfg[inst]
    {
        (*imm).into()
    } else {
        panic!("unexpected instruction in narrow_iconst");
    };

    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    let ty = pos.func.dfg.ctrl_typevar(inst);
    if isa.pointer_bits() == 32 && ty == I64 {
        let low = pos.ins().iconst(I32, imm & 0xffffffff);
        let high = pos.ins().iconst(I32, imm >> 32);
        // The instruction has as many results as iconcat, so no need to replace them.
        pos.func.dfg.replace(inst).iconcat(low, high);
        return;
    }

    unimplemented!("missing encoding or legalization for iconst.{:?}", ty);
}

fn narrow_icmp_imm(
    inst: ir::Inst,
    func: &mut ir::Function,
    _cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    use crate::ir::condcodes::{CondCode, IntCC};

    let (arg, cond, imm): (ir::Value, IntCC, i64) = match func.dfg[inst] {
        ir::InstructionData::IntCompareImm {
            opcode: ir::Opcode::IcmpImm,
            arg,
            cond,
            imm,
        } => (arg, cond, imm.into()),
        _ => panic!("unexpected instruction in narrow_icmp_imm"),
    };

    let mut pos = FuncCursor::new(func).at_inst(inst);
    pos.use_srcloc(inst);

    let ty = pos.func.dfg.ctrl_typevar(inst);
    let ty_half = ty.half_width().unwrap();

    let imm_low = pos
        .ins()
        .iconst(ty_half, imm & ((1u128 << ty_half.bits()) - 1) as i64);
    let imm_high = pos
        .ins()
        .iconst(ty_half, imm.wrapping_shr(ty_half.bits().into()));
    let (arg_low, arg_high) = pos.ins().isplit(arg);

    match cond {
        IntCC::Equal => {
            let res_low = pos.ins().icmp(cond, arg_low, imm_low);
            let res_high = pos.ins().icmp(cond, arg_high, imm_high);
            pos.func.dfg.replace(inst).band(res_low, res_high);
        }
        IntCC::NotEqual => {
            let res_low = pos.ins().icmp(cond, arg_low, imm_low);
            let res_high = pos.ins().icmp(cond, arg_high, imm_high);
            pos.func.dfg.replace(inst).bor(res_low, res_high);
        }
        IntCC::SignedGreaterThan
        | IntCC::SignedGreaterThanOrEqual
        | IntCC::SignedLessThan
        | IntCC::SignedLessThanOrEqual
        | IntCC::UnsignedGreaterThan
        | IntCC::UnsignedGreaterThanOrEqual
        | IntCC::UnsignedLessThan
        | IntCC::UnsignedLessThanOrEqual => {
            let b1 = pos.ins().icmp(cond.without_equal(), arg_high, imm_high);
            let b2 = pos
                .ins()
                .icmp(cond.inverse().without_equal(), arg_high, imm_high);
            let b3 = pos.ins().icmp(cond.unsigned(), arg_low, imm_low);
            let c1 = pos.ins().bnot(b2);
            let c2 = pos.ins().band(c1, b3);
            pos.func.dfg.replace(inst).bor(b1, c2);
        }
        _ => unimplemented!("missing legalization for condition {:?}", cond),
    }
}

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

    // Shift and subtract algorithm
    // Adapted from https://en.wikipedia.org/wiki/Division_algorithm#Integer_division_(unsigned)_with_remainder
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

fn expand_fmin_fmax(
    inst: ir::Inst,
    func: &mut ir::Function,
    cfg: &mut ControlFlowGraph,
    _isa: &dyn TargetIsa,
) {
    use crate::ir::condcodes::{FloatCC, IntCC};
    use crate::ir::immediates::{Ieee32, Ieee64};

    let (x, y, fmin) = match func.dfg[inst] {
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Fmin,
            args,
        } => (args[0], args[1], true),
        ir::InstructionData::Binary {
            opcode: ir::Opcode::Fmax,
            args,
        } => (args[0], args[1], false),
        _ => panic!("Expected fmin/fmax: {}", func.dfg.display_inst(inst, None)),
    };

    // Replace `result = fmin/fmax x, y` with:
    //
    //   potential_nan = fadd x, y
    //   v0 = fcmp uno x, y
    //   brnz v0 result_block(v0)
    //   jump no_nans_block
    // zeros_block():           // fmin and fmax operators treat -0.0 as being less than 0.0.
    //   zero = f32/64const -0/+0
    //   v1 = fcmp eq x, y
    //   x_i = bitcast.I32/I64(x)
    //   y_i = bitcast.I32.I64(y)
    //   v2 = icmp ne x_i, y_i  // x and y are -0 and 0
    //   v3 = band v1, v2
    //   brnz v3 result_block(zero)
    //   jump comparison_block
    // comparison_block():
    //   v4 = fcmp le/ge x, y
    //   brnz v4 result_block(x)
    //   jump result_block(y)
    // result_block(result):

    let old_block = func.layout.pp_block(inst);
    let zeros_block = func.dfg.make_block();
    let comparison_block = func.dfg.make_block();
    let result_block = func.dfg.make_block();

    let result = func.dfg.first_result(inst);
    func.dfg.clear_results(inst);
    func.dfg.attach_block_param(result_block, result);

    let potential_nan = func.dfg.replace(inst).fadd(x, y);
    let mut pos = FuncCursor::new(func).after_inst(inst);
    pos.use_srcloc(inst);

    let v0 = pos.ins().fcmp(FloatCC::Unordered, x, y);
    pos.ins().brnz(v0, result_block, &[potential_nan]);
    pos.ins().jump(zeros_block, &[]);

    pos.insert_block(zeros_block);
    let zero_str = if fmin { "-0x0.0p+0" } else { "0x0.0p+0" };
    let ty = pos.func.dfg.ctrl_typevar(inst);
    let (zero, int_ty) = match ty {
        F32 => (pos.ins().f32const(zero_str.parse::<Ieee32>().unwrap()), I32),
        F64 => (pos.ins().f64const(zero_str.parse::<Ieee64>().unwrap()), I64),
        _ => panic!("expected F32/F64, got {}", ty),
    };
    let v1 = pos.ins().fcmp(FloatCC::Equal, x, y);
    let x_i = pos.ins().bitcast(int_ty, x);
    let y_i = pos.ins().bitcast(int_ty, y);
    let v2 = pos.ins().icmp(IntCC::NotEqual, x_i, y_i);
    let v3 = pos.ins().band(v1, v2);
    pos.ins().brnz(v3, result_block, &[zero]);
    pos.ins().jump(comparison_block, &[]);

    pos.insert_block(comparison_block);
    let cc = if fmin {
        FloatCC::LessThanOrEqual
    } else {
        FloatCC::GreaterThanOrEqual
    };
    let v4 = pos.ins().fcmp(cc, x, y);
    pos.ins().brnz(v4, result_block, &[x]);
    pos.ins().jump(result_block, &[y]);
    pos.insert_block(result_block);

    cfg.recompute_block(pos.func, result_block);
    cfg.recompute_block(pos.func, zeros_block);
    cfg.recompute_block(pos.func, comparison_block);
    cfg.recompute_block(pos.func, old_block);
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
