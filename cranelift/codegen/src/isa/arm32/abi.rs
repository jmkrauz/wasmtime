//! ARM ABI implementation.
//! Only AAPCS is supported.

use super::registers::{D, GPR, Q, RU, S};
use crate::abi::{legalize_args, ArgAction, ArgAssigner, ValueConversion};
use crate::cursor::{Cursor, CursorPosition, EncCursor};
use crate::ir::immediates::{Imm64, Offset32};
use crate::ir::types::{F32, F64, I32};
use crate::ir::{
    self, AbiParam, ArgumentExtension, ArgumentLoc, ArgumentPurpose, InstBuilder, StackSlot, Type,
    ValueLoc,
};
use crate::isa::{CallConv, RegClass, RegUnit, TargetIsa};
use crate::regalloc::RegisterSet;
use crate::result::CodegenResult;
use crate::stack_layout::layout_stack;
use alloc::borrow::Cow;
use core::i32;
use target_lexicon::{PointerWidth, Triple};

const S_LIMIT: usize = 16;

struct Args {
    pointer_bits: u8,
    pointer_bytes: u8,
    pointer_type: Type,
    gpr_used: u32,
    gpr_limit: u32,
    s_used: [bool; S_LIMIT],
    vfp_stack_used: bool,
    offset: u32,
}

impl Args {
    fn new(bits: u8) -> Self {
        Self {
            pointer_bits: bits,
            pointer_bytes: bits / 8,
            pointer_type: Type::int(u16::from(bits)).unwrap(),
            gpr_used: 0,
            gpr_limit: 4,
            s_used: [false; S_LIMIT],
            vfp_stack_used: false,
            offset: 0,
        }
    }

    fn alloc_fpr(&mut self, ty: Type) -> Option<RegUnit> {
        if !self.vfp_stack_used {
            match ty {
                F32 => {
                    for i in 0..S_LIMIT {
                        if !self.s_used[i] {
                            self.s_used[i] = true;
                            return Some(S.unit(i));
                        }
                    }
                }
                F64 => {
                    for i in 0..S_LIMIT / 2 {
                        if !self.s_used[2 * i] && !self.s_used[2 * i + 1] {
                            self.s_used[2 * i] = true;
                            self.s_used[2 * i + 1] = true;
                            return Some(D.unit(i));
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }
}

impl ArgAssigner for Args {
    fn assign(&mut self, arg: &AbiParam) -> ArgAction {
        fn align(value: u32, to: u32) -> u32 {
            (value + to - 1) & !(to - 1)
        }

        let ty = arg.value_type;

        // Check for a legal type.
        // SIMD instructions are currently no implemented, so break down vectors
        if ty.is_vector() {
            return ValueConversion::VectorSplit.into();
        }

        // Large integers and booleans are broken down to fit in a register.
        if !ty.is_float() && ty.bits() > u16::from(self.pointer_bits) {
            // Align registers and stack to a multiple of two pointers.
            self.gpr_used = align(self.gpr_used, 2);
            self.offset = align(self.offset, 2 * u32::from(self.pointer_bytes));
            return ValueConversion::IntSplit.into();
        }

        // Small integers are extended to the size of a pointer register.
        if ty.is_int() && ty.bits() < u16::from(self.pointer_bits) {
            match arg.extension {
                ArgumentExtension::None => {}
                ArgumentExtension::Uext => return ValueConversion::Uext(self.pointer_type).into(),
                ArgumentExtension::Sext => return ValueConversion::Sext(self.pointer_type).into(),
            }
        }

        // Try to use a GPR.
        if !ty.is_float() && self.gpr_used < self.gpr_limit {
            // Assign to a register.
            let reg = GPR.unit(self.gpr_used as usize);
            self.gpr_used += 1;
            return ArgumentLoc::Reg(reg).into();
        }

        // Try to use a floating point register.
        if let Some(reg) = self.alloc_fpr(ty) {
            return ArgumentLoc::Reg(reg).into();
        }

        // Assign a stack location.
        if ty.is_float() {
            self.vfp_stack_used = true;
        }
        let loc = ArgumentLoc::Stack(self.offset as i32);
        self.offset += u32::from(ty.bytes());
        debug_assert!(self.offset <= i32::MAX as u32);
        loc.into()
    }
}

/// Legalize `sig`.
pub fn legalize_signature(sig: &mut Cow<ir::Signature>, triple: &Triple, _current: bool) {
    let bits = triple.pointer_width().unwrap().bits();

    let mut args = Args::new(bits);
    if let Some(new_params) = legalize_args(&sig.params, &mut args) {
        sig.to_mut().params = new_params;
    }

    let mut rets = Args::new(bits);
    if let Some(new_returns) = legalize_args(&sig.returns, &mut rets) {
        sig.to_mut().returns = new_returns;
    }
}

/// Get register class for a type appearing in a legalized signature.
pub fn regclass_for_abi_type(ty: ir::Type) -> RegClass {
    if ty.is_int() || ty.is_bool() {
        GPR
    } else {
        match ty.bits() {
            32 => S,
            64 => D,
            128 => Q,
            _ => panic!("Unexpected {} ABI type for arm32", ty),
        }
    }
}

/// Get the set of allocatable registers for `func`.
pub fn allocatable_registers(_func: &ir::Function) -> RegisterSet {
    let mut regs = RegisterSet::new();

    // AAPCS: "The role of register r9 is platform specific. (...)
    // The usage of this register may require that the value held is persistent across all calls."
    regs.take(GPR, RU::r9 as RegUnit);
    regs.take(GPR, RU::r12 as RegUnit); // IP is used by some recipes to hold temporary values.
    regs.take(GPR, RU::r13 as RegUnit); // SP
    regs.take(GPR, RU::r14 as RegUnit); // LR
    regs.take(GPR, RU::r15 as RegUnit); // PC

    regs
}

/// Get the set of callee-saved gpr registers.
fn callee_saved_gprs(isa: &dyn TargetIsa) -> &'static [RU] {
    match isa.triple().pointer_width().unwrap() {
        PointerWidth::U16 | PointerWidth::U64 => panic!(),
        PointerWidth::U32 => &[
            RU::r4,
            RU::r5,
            RU::r6,
            RU::r7,
            RU::r8,
            RU::r9,
            RU::r10,
            RU::r11,
        ],
    }
}

/// Get the set of callee-saved registers that are used.
fn callee_saved_regs_used(isa: &dyn TargetIsa, func: &ir::Function) -> RegisterSet {
    let mut all_callee_saved = RegisterSet::empty();
    for reg in callee_saved_gprs(isa) {
        all_callee_saved.free(GPR, *reg as RegUnit);
    }
    // Registers s16-s31 (d8-d15, q4-q7) must be preserved across subroutine calls.
    for i in S_LIMIT..4 * S_LIMIT {
        all_callee_saved.free(S, i as RegUnit);
    }

    let mut used = RegisterSet::empty();
    for (value, value_loc) in func.locations.iter() {
        if let ValueLoc::Reg(ru) = *value_loc {
            match func.dfg.value_type(value) {
                F32 => {
                    if !used.is_avail(S, ru) {
                        used.free(S, ru);
                    }
                }
                F64 => {
                    if !used.is_avail(D, ru) {
                        used.free(D, ru);
                    }
                }
                _ => {
                    if !used.is_avail(GPR, ru) {
                        used.free(GPR, ru);
                    }
                }
            }
        }
    }

    // regmove and regfill instructions may temporarily divert values into other registers,
    // and these are not reflected in `func.locations`. Scan the function for such instructions
    // and note which callee-saved registers they use.
    //
    // TODO: Consider re-evaluating how regmove/regfill/regspill work and whether it's possible
    // to avoid this step.
    for block in &func.layout {
        for inst in func.layout.block_insts(block) {
            match func.dfg[inst] {
                ir::instructions::InstructionData::RegMove { dst, .. }
                | ir::instructions::InstructionData::RegFill { dst, .. } => {
                    if GPR.contains(dst) && !used.is_avail(GPR, dst) {
                        used.free(GPR, dst);
                    }
                    if S.contains(dst) && !used.is_avail(S, dst) {
                        used.free(S, dst);
                    }
                }
                _ => (),
            }
        }
    }

    used.intersect(&all_callee_saved);
    used
}

pub fn prologue_epilogue(func: &mut ir::Function, isa: &dyn TargetIsa) -> CodegenResult<()> {
    match func.signature.call_conv {
        CallConv::SystemV => default_prologue_epilogue(func, isa),
        _ => unimplemented!("{} calling convention", func.signature.call_conv),
    }
}

fn default_prologue_epilogue(func: &mut ir::Function, isa: &dyn TargetIsa) -> CodegenResult<()> {
    let stack_align = 8;
    let pointer_width = isa.triple().pointer_width().unwrap();
    let word_size = pointer_width.bytes() as usize;
    let csrs = callee_saved_regs_used(isa, func);

    // The reserved stack area is composed of LR and callee-saved registers.
    // TODO: Check if the function body actually contains a `call` instruction. Maybe pushing LR in not necessary.
    let csr_stack_size = ((csrs.iter(GPR).len() + csrs.iter(S).len() + 1) * word_size) as i32;
    let csr_ss = func.create_stack_slot(ir::StackSlotData {
        kind: ir::StackSlotKind::IncomingArg,
        size: csr_stack_size as u32,
        offset: Some(-csr_stack_size),
    });

    let is_leaf = func.is_leaf();
    let total_stack_size = layout_stack(&mut func.stack_slots, is_leaf, stack_align)? as i32;
    let local_stack_size = i64::from(total_stack_size - csr_stack_size);

    // Add CSRs (including LR) to function signature
    let lr_arg = ir::AbiParam::special_reg(I32, ir::ArgumentPurpose::Link, RU::r14 as RegUnit);
    func.signature.params.push(lr_arg);
    func.signature.returns.push(lr_arg);

    for &(regclass, reg_type) in &[(GPR, I32), (S, F32)] {
        for csr in csrs.iter(regclass) {
            let csr_arg =
                ir::AbiParam::special_reg(reg_type, ir::ArgumentPurpose::CalleeSaved, csr);
            func.signature.params.push(csr_arg);
            func.signature.returns.push(csr_arg);
        }
    }

    // Set up the cursor and insert the prologue
    let entry_block = func.layout.entry_block().expect("missing entry block");
    let mut pos = EncCursor::new(func, isa).at_first_insertion_point(entry_block);
    insert_default_prologue(&mut pos, local_stack_size, &csrs, isa, &csr_ss);

    // Reset the cursor and insert the epilogue
    let mut pos = pos.at_position(CursorPosition::Nowhere);
    insert_default_epilogues(&mut pos, local_stack_size, &csrs, &csr_ss);

    Ok(())
}

/// Insert the prologue for a given function.
fn insert_default_prologue(
    pos: &mut EncCursor,
    stack_size: i64,
    csrs: &RegisterSet,
    isa: &dyn TargetIsa,
    csr_ss: &StackSlot,
) {
    if stack_size > 0 {
        // Check if there is a special stack limit parameter. If so insert stack check.
        if let Some(_stack_limit_arg) = pos.func.special_param(ArgumentPurpose::StackLimit) {
            unimplemented!("stack check");
        }
    }

    // Append param to entry EBB
    let block = pos.current_block().expect("missing block under cursor");
    let lr = pos.func.dfg.append_block_param(block, I32);
    pos.func.locations[lr] = ir::ValueLoc::Reg(RU::r14 as RegUnit);

    let word_size: i32 = isa.pointer_bytes().into();
    let csr_stack_size =
        ((csrs.iter(GPR).len() + csrs.iter(S).len() + 1) * word_size as usize) as i64;

    pos.ins().adjust_sp_down_imm(Imm64::new(csr_stack_size));
    pos.ins().stack_store(lr, csr_ss.clone(), Offset32::new(0));

    let mut regs_pushed = 1;
    for &(regclass, reg_type) in &[(GPR, I32), (S, F32)] {
        for reg in csrs.iter(regclass) {
            // Append param to entry EBB
            let csr_arg = pos.func.dfg.append_block_param(block, reg_type);

            // Assign it a location
            pos.func.locations[csr_arg] = ir::ValueLoc::Reg(reg);

            // Remember it so we can push it momentarily
            pos.ins().stack_store(
                csr_arg,
                csr_ss.clone(),
                Offset32::new(regs_pushed * word_size),
            );
            regs_pushed += 1;
        }
    }

    // Allocate stack frame storage.
    if stack_size > 0 {
        if isa.flags().enable_probestack() && stack_size > (1 << isa.flags().probestack_size_log2())
        {
            unimplemented!("probestack");
        } else {
            // Simply decrement the stack pointer.
            pos.ins().adjust_sp_down_imm(Imm64::new(stack_size));
        }
    }
}

/// Find all `return` instructions and insert epilogues before them.
fn insert_default_epilogues(
    pos: &mut EncCursor,
    stack_size: i64,
    csrs: &RegisterSet,
    csr_ss: &StackSlot,
) {
    while let Some(block) = pos.next_block() {
        pos.goto_last_inst(block);
        if let Some(inst) = pos.current_inst() {
            if pos.func.dfg[inst].opcode().is_return() {
                insert_default_epilogue(inst, stack_size, pos, csrs, csr_ss);
            }
        }
    }
}

/// Insert an epilogue given a specific `return` instruction..
fn insert_default_epilogue(
    inst: ir::Inst,
    stack_size: i64,
    pos: &mut EncCursor,
    csrs: &RegisterSet,
    csr_ss: &StackSlot,
) {
    if stack_size > 0 {
        pos.ins().adjust_sp_up_imm(Imm64::new(stack_size));
    }
    let reg_size = 4;
    let csr_stack_size =
        ((csrs.iter(GPR).len() + csrs.iter(S).len() + 1) * reg_size as usize) as i64;

    // Pop all the callee-saved registers, stepping backward each time to
    // preserve the correct order.
    pos.ins().adjust_sp_up_imm(Imm64::new(csr_stack_size));
    pos.prev_inst();
    let lr_ret = pos.ins().stack_load(I32, csr_ss.clone(), Offset32::new(0));
    pos.prev_inst();

    pos.func.locations[lr_ret] = ir::ValueLoc::Reg(RU::r14 as RegUnit);
    pos.func.dfg.append_inst_arg(inst, lr_ret);

    let mut regs_popped = 1;
    for &(regclass, reg_type) in &[(GPR, I32), (S, F32)] {
        for reg in csrs.iter(regclass) {
            let csr_ret = pos.ins().stack_load(
                reg_type,
                csr_ss.clone(),
                Offset32::new(reg_size * regs_popped),
            );
            pos.prev_inst();
            regs_popped += 1;

            pos.func.locations[csr_ret] = ir::ValueLoc::Reg(reg);
            pos.func.dfg.append_inst_arg(inst, csr_ret);
        }
    }
}
