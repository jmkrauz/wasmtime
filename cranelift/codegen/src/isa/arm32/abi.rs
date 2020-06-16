//! Implementation of the 32-bit ARM ABI.

use crate::abi::{legalize_args, ArgAction, ArgAssigner, ValueConversion};
use crate::ir;
use crate::ir::types;
use crate::ir::types::*;
use crate::ir::{AbiParam, ArgumentExtension, ArgumentLoc, StackSlot};
use crate::isa::arm32::{self, inst::*};
use crate::isa::{self, RegUnit};
use crate::machinst::*;
use crate::settings;
use crate::{CodegenError, CodegenResult};

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::convert::TryInto;
use core::mem;
use log::{debug, trace};
use regalloc::{RealReg, Reg, RegClass, Set, SpillSlot, Writable};
use smallvec::SmallVec;

static STACK_ARG_RET_SIZE_LIMIT: u32 = 128 * 1024 * 1024;

struct Args {
    r_used: u32,
    r_limit: u32,
    stack_offset: u32,
}

impl Args {
    fn new() -> Self {
        Self {
            r_used: 0,
            r_limit: 4,
            stack_offset: 0,
        }
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
        if !ty.is_float() && ty.bits() > 32 {
            // Align registers and stack to a multiple of two pointers.
            self.r_used = align(self.r_used, 2);
            self.stack_offset = align(self.stack_offset, 8);
            return ValueConversion::IntSplit.into();
        }

        // Small integers are extended to the size of a pointer register.
        if ty.is_int() && ty.bits() < 32 {
            match arg.extension {
                ArgumentExtension::None => {}
                ArgumentExtension::Uext => return ValueConversion::Uext(I32).into(),
                ArgumentExtension::Sext => return ValueConversion::Sext(I32).into(),
            }
        }

        // Try to use a GPR.
        if !ty.is_float() && self.r_used < self.r_limit {
            // Assign to a register.
            let reg = self.r_used as RegUnit;
            self.r_used += 1;
            return ArgumentLoc::Reg(reg).into();
        }

        if ty.is_float() {
            unimplemented!()
        }

        // Assign a stack location.
        let loc = ArgumentLoc::Stack(self.stack_offset as i32);
        self.stack_offset += 4;
        loc.into()
    }
}

pub fn legalize_signature(sig: &mut Cow<ir::Signature>) {
    let mut args = Args::new();
    if let Some(new_params) = legalize_args(&sig.params, &mut args) {
        sig.to_mut().params = new_params;
    }

    let mut rets = Args::new();
    if let Some(new_returns) = legalize_args(&sig.returns, &mut rets) {
        sig.to_mut().returns = new_returns;
    }
}

/// A location for an argument or return value.
#[derive(Clone, Copy, Debug)]
enum ABIArg {
    /// In a real register.
    Reg(RealReg, ir::Type),
    /// Arguments only: on stack, at given offset from SP at entry.
    Stack(i32, ir::Type),
}

/// Arm ABI information shared between body (callee) and caller.
struct ABISig {
    args: Vec<ABIArg>,
    rets: Vec<ABIArg>,
    stack_arg_space: u32,
    stack_ret_space: u32,
    stack_ret_arg: Option<usize>,
}

#[rustfmt::skip]
static CALLEE_SAVED_GPR: &[bool] = &[
    /* r0 - r3 */
    false, false, false, false,
    /* r4 - r11*/
    true, true, true, true, true, true, true, true,
    /* ip, sp, lr, pc*/
    false, false, false, false
];

/// Are we computing information about arguments or return values? Much of the
/// handling is factored out into common routines; this enum allows us to
/// distinguish which case we're handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArgsOrRets {
    Args,
    Rets,
}

/// Process a list of parameters or return values and allocate them to R-regs and stack slots.
///
/// Returns the list of argument locations, and the stack-space used (rounded up
/// to a 8-byte-aligned boundary).
fn compute_arg_locs(
    params: &[ir::AbiParam],
    args_or_rets: ArgsOrRets,
    add_ret_area_ptr: bool,
) -> CodegenResult<(Vec<ABIArg>, u32, Option<usize>)> {
    // See AAPCS ABI https://developer.arm.com/docs/ihi0042/latest
    // r9 is an additional callee-saved variable register.
    let mut stack_off: u32 = 0;
    let mut next_rreg: u8 = 0;
    let mut ret = vec![];

    let max_rreg: u8 = 3;

    for param in params {
        // Validate "purpose".
        match &param.purpose {
            &ir::ArgumentPurpose::VMContext | &ir::ArgumentPurpose::Normal => {}
            _ => panic!(
                "Unsupported argument purpose {:?} in signature: {:?}",
                param.purpose, params
            ),
        }

        let ty = param.value_type;
        match param.location {
            ArgumentLoc::Reg(reg) => {
                let reg: u8 = reg.try_into().unwrap();
                ret.push(ABIArg::Reg(rreg(reg).to_real_reg(), ty));
                if next_rreg <= reg {
                    next_rreg += 1;
                }
            }
            ArgumentLoc::Stack(off) => {
                ret.push(ABIArg::Stack(off, ty));
                let off: u32 = off.try_into().unwrap();
                if off > stack_off {
                    stack_off = off;
                }
            }
            ArgumentLoc::Unassigned => {
                panic!("Unassigned param location {:?}", param);
            }
        }
    }

    let extra_arg = if add_ret_area_ptr {
        debug_assert!(args_or_rets == ArgsOrRets::Args);
        if next_rreg < max_rreg {
            ret.push(ABIArg::Reg(rreg(next_rreg).to_real_reg(), I32));
        } else {
            ret.push(ABIArg::Stack(stack_off as i32 + 4, I32));
            stack_off += 4;
        }
        Some(ret.len() - 1)
    } else {
        None
    };

    // To avoid overflow issues, limit the arg/return size to something
    // reasonable -- here, 128 MB.
    if stack_off >= STACK_ARG_RET_SIZE_LIMIT {
        return Err(CodegenError::ImplLimitExceeded);
    }

    stack_off = (stack_off + 7) & !7;

    Ok((ret, stack_off, extra_arg))
}

impl ABISig {
    fn from_func_sig(sig: &ir::Signature) -> CodegenResult<ABISig> {
        // Compute args and retvals from signature. Handle retvals first,
        // because we may need to add a return-area arg to the args.
        let (rets, stack_ret_space, _) = compute_arg_locs(
            &sig.returns,
            ArgsOrRets::Rets,
            /* extra ret-area ptr = */ false,
        )?;
        let need_stack_return_area = stack_ret_space > 0;
        let (args, stack_arg_space, stack_ret_arg) =
            compute_arg_locs(&sig.params, ArgsOrRets::Args, need_stack_return_area)?;

        trace!(
            "ABISig: sig {:?} => args = {:?} rets = {:?} arg stack = {} ret stack = {} stack_ret_arg = {:?}",
            sig,
            args,
            rets,
            stack_arg_space,
            stack_ret_space,
            stack_ret_arg
        );

        Ok(ABISig {
            args,
            rets,
            stack_arg_space,
            stack_ret_space,
            stack_ret_arg,
        })
    }
}

/// ARM32 ABI object for a function body.
pub struct Arm32ABIBody {
    /// signature: arg and retval regs
    sig: ABISig,
    /// offsets to each stackslot
    stackslots: Vec<u32>,
    /// total stack size of all stackslots
    stackslots_size: u32,
    /// clobbered registers, from regalloc.
    clobbered: Set<Writable<RealReg>>,
    /// total number of spillslots, from regalloc.
    spillslots: Option<usize>,
    /// Total frame size.
    total_frame_size: Option<u32>,
    /// The register holding the return-area pointer, if needed.
    ret_area_ptr: Option<Writable<Reg>>,
    /// The settings controlling this function's compilation.
    flags: settings::Flags,
    /// Whether or not this function is a "leaf", meaning it calls no other
    /// functions
    is_leaf: bool,
}

fn in_int_reg(ty: ir::Type) -> bool {
    match ty {
        types::I8 | types::I16 | types::I32 => true,
        types::B1 | types::B8 | types::B16 | types::B32 => true,
        _ => false,
    }
}

fn in_float_reg(ty: ir::Type) -> bool {
    match ty {
        types::F32 | types::F64 => true,
        _ => false,
    }
}

fn load_stack(mem: MemArg, into_reg: Writable<Reg>, ty: Type) -> Inst {
    match ty {
        types::B1 | types::B8 | types::I8 | types::B16 | types::I16 | types::B32 | types::I32 => {
            Inst::Load {
                rt: into_reg,
                mem,
                srcloc: None,
                bits: 32,
                sign_extend: false,
            }
        }
        _ => unimplemented!("load_stack({})", ty),
    }
}

fn store_stack(mem: MemArg, from_reg: Reg, ty: Type) -> Inst {
    match ty {
        types::B1 | types::B8 | types::I8 | types::B16 | types::I16 | types::B32 | types::I32 => {
            Inst::Store {
                rt: from_reg,
                mem,
                srcloc: None,
                bits: 32,
            }
        }
        _ => unimplemented!("store_stack({})", ty),
    }
}

fn is_callee_save(r: RealReg) -> bool {
    match r.get_class() {
        RegClass::I32 => match r.get_hw_encoding() as u8 {
            n if n >= 4 && n <= 11 => true,
            _ => false,
        },
        _ => unimplemented!(),
    }
}

fn get_callee_saves(regs: Vec<Writable<RealReg>>) -> Vec<Writable<RealReg>> {
    regs.into_iter()
        .filter(|r| is_callee_save(r.to_reg()))
        .collect()
}

fn is_caller_save(r: RealReg) -> bool {
    match r.get_class() {
        RegClass::I32 => match r.get_hw_encoding() {
            n if n <= 4 => true,
            12 => true,
            _ => false,
        },
        _ => panic!("Unexpected RegClass"),
    }
}

fn get_caller_saves() -> Vec<Writable<Reg>> {
    let mut set = vec![];
    for i in 0..15 {
        let r = writable_rreg(i);
        if is_caller_save(r.to_reg().to_real_reg()) {
            set.push(r);
        }
    }
    set
}

fn gen_stack_limit(f: &ir::Function, abi: &ABISig, gv: ir::GlobalValue) -> (Reg, Vec<Inst>) {
    let mut insts = Vec::new();
    let reg = generate_gv(f, abi, gv, &mut insts);
    return (reg, insts);

    fn generate_gv(
        f: &ir::Function,
        abi: &ABISig,
        gv: ir::GlobalValue,
        insts: &mut Vec<Inst>,
    ) -> Reg {
        match f.global_values[gv] {
            // Return the direct register the vmcontext is in
            ir::GlobalValueData::VMContext => {
                get_special_purpose_param_register(f, abi, ir::ArgumentPurpose::VMContext)
                    .expect("no vmcontext parameter found")
            }
            // Load our base value into a register, then load from that register
            // in to a temporary register.
            ir::GlobalValueData::Load {
                base,
                offset,
                global_type: _,
                readonly: _,
            } => {
                let base = generate_gv(f, abi, base, insts);
                let into_reg = writable_ip_reg();
                let offset: i32 = offset.into();
                let mem = if let Some(mem) = MemArg::reg_maybe_offset(base, offset) {
                    mem
                } else {
                    insts.extend(Inst::load_constant(into_reg, offset as u32));
                    MemArg::reg_plus_reg(base, into_reg.to_reg(), 0)
                };
                insts.push(Inst::Load {
                    rt: into_reg,
                    mem,
                    srcloc: None,
                    bits: 32,
                    sign_extend: false,
                });
                return into_reg.to_reg();
            }
            ref other => panic!("global value for stack limit not supported: {}", other),
        }
    }
}

fn get_special_purpose_param_register(
    f: &ir::Function,
    abi: &ABISig,
    purpose: ir::ArgumentPurpose,
) -> Option<Reg> {
    let idx = f.signature.special_param_index(purpose)?;
    match abi.args[idx] {
        ABIArg::Reg(reg, _) => Some(reg.to_reg()),
        ABIArg::Stack(..) => None,
    }
}

impl Arm32ABIBody {
    /// Create a new body ABI instance.
    pub fn new(f: &ir::Function, flags: settings::Flags) -> CodegenResult<Self> {
        debug!("Arm32 ABI: func signature {:?}", f.signature);

        let sig = ABISig::from_func_sig(&f.signature)?;

        let call_conv = f.signature.call_conv;
        // Only this calling conventions are supported.
        assert!(
            call_conv == isa::CallConv::SystemV,
            "Unsupported calling convention: {:?}",
            call_conv
        );

        // Compute stackslot locations and total stackslot size.
        let mut stack_offset: u32 = 0;
        let mut stackslots = vec![];
        for (stackslot, data) in f.stack_slots.iter() {
            let off = stack_offset;
            stack_offset += data.size;
            stack_offset = (stack_offset + 3) & !3;
            assert_eq!(stackslot.as_u32() as usize, stackslots.len());
            stackslots.push(off);
        }

        Ok(Self {
            sig,
            stackslots,
            stackslots_size: stack_offset,
            clobbered: Set::empty(),
            spillslots: None,
            total_frame_size: None,
            ret_area_ptr: None,
            flags,
            is_leaf: f.is_leaf(),
        })
    }
}

impl ABIBody for Arm32ABIBody {
    type I = Inst;

    fn temp_needed(&self) -> bool {
        self.sig.stack_ret_arg.is_some()
    }

    fn init(&mut self, maybe_tmp: Option<Writable<Reg>>) {
        if self.sig.stack_ret_arg.is_some() {
            assert!(maybe_tmp.is_some());
            self.ret_area_ptr = maybe_tmp;
        }
    }

    fn flags(&self) -> &settings::Flags {
        &self.flags
    }

    fn liveins(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for &arg in &self.sig.args {
            if let ABIArg::Reg(r, _) = arg {
                set.insert(r);
            }
        }
        set
    }

    fn liveouts(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for &ret in &self.sig.rets {
            if let ABIArg::Reg(r, _) = ret {
                set.insert(r);
            }
        }
        set
    }

    fn num_args(&self) -> usize {
        self.sig.args.len()
    }

    fn num_retvals(&self) -> usize {
        self.sig.rets.len()
    }

    fn num_stackslots(&self) -> usize {
        self.stackslots.len()
    }

    fn gen_copy_arg_to_reg(&self, idx: usize, into_reg: Writable<Reg>) -> Inst {
        match &self.sig.args[idx] {
            &ABIArg::Reg(r, ty) => Inst::gen_move(into_reg, r.to_reg(), ty),
            &ABIArg::Stack(off, ty) => {
                let mem = if let Some(mem) = MemArg::reg_maybe_offset(sp_reg(), off) {
                    mem
                } else {
                    unimplemented!()
                };
                load_stack(mem, into_reg, ty)
            }
        }
    }

    fn gen_retval_area_setup(&self) -> Option<Inst> {
        if let Some(i) = self.sig.stack_ret_arg {
            let inst = self.gen_copy_arg_to_reg(i, self.ret_area_ptr.unwrap());
            trace!(
                "gen_retval_area_setup: inst {:?}; ptr reg is {:?}",
                inst,
                self.ret_area_ptr.unwrap().to_reg()
            );
            Some(inst)
        } else {
            trace!("gen_retval_area_setup: not needed");
            None
        }
    }

    fn gen_copy_reg_to_retval(
        &self,
        idx: usize,
        from_reg: Writable<Reg>,
        ext: ArgumentExtension,
    ) -> Vec<Inst> {
        let mut ret = Vec::new();
        match &self.sig.rets[idx] {
            &ABIArg::Reg(r, ty) => {
                let from_bits = arm32::lower::ty_bits(ty) as u8;
                let dest_reg = Writable::from_reg(r.to_reg());
                match (ext, from_bits) {
                    (ArgumentExtension::Uext, n) if n < 32 => {
                        ret.push(Inst::Extend {
                            rd: dest_reg,
                            rm: from_reg.to_reg(),
                            from_bits,
                            signed: false,
                        });
                    }
                    (ArgumentExtension::Sext, n) if n < 32 => {
                        ret.push(Inst::Extend {
                            rd: dest_reg,
                            rm: from_reg.to_reg(),
                            from_bits,
                            signed: true,
                        });
                    }
                    _ => ret.push(Inst::gen_move(dest_reg, from_reg.to_reg(), ty)),
                };
            }
            &ABIArg::Stack(off, ty) => {
                let from_bits = arm32::lower::ty_bits(ty) as u8;
                // Trash the from_reg; it should be its last use.
                match (ext, from_bits) {
                    (ArgumentExtension::Uext, n) if n < 32 => {
                        ret.push(Inst::Extend {
                            rd: from_reg,
                            rm: from_reg.to_reg(),
                            from_bits,
                            signed: false,
                        });
                    }
                    (ArgumentExtension::Sext, n) if n < 32 => {
                        ret.push(Inst::Extend {
                            rd: from_reg,
                            rm: from_reg.to_reg(),
                            from_bits,
                            signed: true,
                        });
                    }
                    _ => {}
                };
                ret.push(store_stack(
                    MemArg::reg_maybe_offset(sp_reg(), off).unwrap(),
                    from_reg.to_reg(),
                    ty,
                ));
            }
        }
        ret
    }

    fn gen_ret(&self) -> Inst {
        Inst::Ret
    }

    fn gen_epilogue_placeholder(&self) -> Inst {
        Inst::EpiloguePlaceholder {}
    }

    fn set_num_spillslots(&mut self, slots: usize) {
        self.spillslots = Some(slots);
    }

    fn set_clobbered(&mut self, clobbered: Set<Writable<RealReg>>) {
        self.clobbered = clobbered;
    }

    fn load_stackslot(
        &self,
        slot: StackSlot,
        offset: u32,
        ty: Type,
        into_reg: Writable<Reg>,
    ) -> Inst {
        let stack_off = self.stackslots[slot.as_u32() as usize];
        let sp_off = stack_off + offset;
        if let Some(mem) = MemArg::reg_maybe_offset(sp_reg(), sp_off.try_into().unwrap()) {
            load_stack(mem, into_reg, ty)
        } else {
            unimplemented!()
        }
    }

    fn store_stackslot(&self, slot: StackSlot, offset: u32, ty: Type, from_reg: Reg) -> Inst {
        let stack_off = self.stackslots[slot.as_u32() as usize];
        let sp_off = stack_off + offset;
        if let Some(mem) = MemArg::reg_maybe_offset(sp_reg(), sp_off.try_into().unwrap()) {
            store_stack(mem, from_reg, ty)
        } else {
            unimplemented!()
        }
    }

    fn stackslot_addr(&self, slot: StackSlot, offset: u32, into_reg: Writable<Reg>) -> Inst {
        let stack_off = self.stackslots[slot.as_u32() as usize];
        let sp_off = stack_off + offset;
        if sp_off & !((1 << 12) - 1) == 0 {
            Inst::AluRRImm12 {
                alu_op: ALUOp::Add,
                rd: into_reg,
                rn: sp_reg(),
                imm12: sp_off.try_into().unwrap(),
            }
        } else {
            unimplemented!()
        }
    }

    // Load from a spillslot.
    fn load_spillslot(&self, _slot: SpillSlot, _ty: Type, _into_reg: Writable<Reg>) -> Inst {
        unimplemented!()
    }

    // Store to a spillslot.
    fn store_spillslot(&self, _slot: SpillSlot, _ty: Type, _from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn gen_prologue(&mut self) -> Vec<Inst> {
        let mut insts = vec![];
        let mut reg_list = SmallVec::<[Reg; 16]>::new();

        let mut callee_saved_used = 0;

        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for reg in clobbered {
            let reg = reg.to_reg();
            match reg.get_class() {
                RegClass::I32 => {
                    reg_list.push(reg.to_reg());
                    callee_saved_used += 4;
                }
                _ => unimplemented!(),
            }
        }

        if !self.is_leaf {
            // For lr
            callee_saved_used += 4;
        }

        let total_stacksize = self.stackslots_size + 4 * self.spillslots.unwrap() as u32;
        let frame_size = if total_stacksize == 0 && callee_saved_used % 8 == 4 {
            reg_list.push(ip_reg());
            0
        } else if (total_stacksize + callee_saved_used) % 8 == 0 {
            total_stacksize
        } else {
            total_stacksize + 4
        };

        if !self.is_leaf {
            reg_list.push(lr_reg());
        }
        if !reg_list.is_empty() {
            insts.push(Inst::Push { reg_list });
        }

        insts.extend(adjust_stack(frame_size, true));

        // Stash this value.  We'll need it for the epilogue.
        debug_assert!(self.total_frame_size.is_none());
        self.total_frame_size = Some(frame_size);
        insts
    }

    fn gen_epilogue(&self) -> Vec<Inst> {
        let mut insts = vec![];

        // Undo what we did in the prologue.

        // Clear the spill area and the 8-alignment padding below it.
        let frame_size = self.total_frame_size.unwrap();
        insts.extend(adjust_stack(frame_size, false));

        let mut reg_list = SmallVec::<[Writable<Reg>; 16]>::new();
        let mut callee_saved_used = 0;

        // Restore regs.
        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for reg in clobbered.into_iter().rev() {
            let reg = reg.to_reg();
            match reg.get_class() {
                RegClass::I32 => {
                    reg_list.push(Writable::from_reg(reg.to_reg()));
                    callee_saved_used += 4;
                }
                _ => unimplemented!(),
            }
        }

        if !self.is_leaf {
            callee_saved_used += 4;
        }
        if frame_size == 0 && callee_saved_used % 8 == 4 {
            reg_list.push(writable_ip_reg());
        }
        if !self.is_leaf {
            reg_list.push(writable_pc_reg());
        }
        if !reg_list.is_empty() {
            insts.push(Inst::Pop { reg_list });
        }
        if self.is_leaf {
            insts.push(Inst::Ret {});
        }

        insts
    }

    fn frame_size(&self) -> u32 {
        self.total_frame_size
            .expect("frame size not computed before prologue generation")
    }

    fn get_spillslot_size(&self, _rc: RegClass, _ty: Type) -> u32 {
        unimplemented!()
    }

    fn gen_spill(&self, to_slot: SpillSlot, from_reg: RealReg, ty: Type) -> Inst {
        self.store_spillslot(to_slot, ty, from_reg.to_reg())
    }

    fn gen_reload(&self, to_reg: Writable<RealReg>, from_slot: SpillSlot, ty: Type) -> Inst {
        self.load_spillslot(from_slot, ty, to_reg.map(|r| r.to_reg()))
    }
}

enum CallDest {
    ExtName(ir::ExternalName, RelocDistance),
    Reg(Reg),
}

/// Arm32 ABI object for a function call.
pub struct Arm32ABICall {
    sig: ABISig,
    uses: Vec<Reg>,
    defs: Vec<Writable<Reg>>,
    dest: CallDest,
    loc: ir::SourceLoc,
    opcode: ir::Opcode,
}

fn abisig_to_uses_and_defs(sig: &ABISig) -> (Vec<Reg>, Vec<Writable<Reg>>) {
    // Compute uses: all arg regs.
    let mut uses = vec![];
    for arg in &sig.args {
        match arg {
            &ABIArg::Reg(reg, _) => uses.push(reg.to_reg()),
            _ => {}
        }
    }

    // Compute defs: all retval regs, and all caller-save (clobbered) regs.
    let mut defs = get_caller_saves();
    for ret in &sig.rets {
        match ret {
            &ABIArg::Reg(reg, _) => defs.push(Writable::from_reg(reg.to_reg())),
            _ => {}
        }
    }

    (uses, defs)
}

impl Arm32ABICall {
    /// Create a callsite ABI object for a call directly to the specified function.
    pub fn from_func(
        sig: &ir::Signature,
        extname: &ir::ExternalName,
        dist: RelocDistance,
        loc: ir::SourceLoc,
    ) -> CodegenResult<Arm32ABICall> {
        let sig = ABISig::from_func_sig(sig)?;
        let (uses, defs) = abisig_to_uses_and_defs(&sig);
        Ok(Arm32ABICall {
            sig,
            uses,
            defs,
            dest: CallDest::ExtName(extname.clone(), dist),
            loc,
            opcode: ir::Opcode::Call,
        })
    }

    /// Create a callsite ABI object for a call to a function pointer with the
    /// given signature.
    pub fn from_ptr(
        sig: &ir::Signature,
        ptr: Reg,
        loc: ir::SourceLoc,
        opcode: ir::Opcode,
    ) -> CodegenResult<Arm32ABICall> {
        let sig = ABISig::from_func_sig(sig)?;
        let (uses, defs) = abisig_to_uses_and_defs(&sig);
        Ok(Arm32ABICall {
            sig,
            uses,
            defs,
            dest: CallDest::Reg(ptr),
            loc,
            opcode,
        })
    }
}

fn adjust_stack(amount: u32, is_sub: bool) -> Vec<Inst> {
    let mut insts = vec![];
    if amount == 0 {
        return insts;
    }
    let alu_op = if is_sub { ALUOp::Sub } else { ALUOp::Add };
    if amount & !((1 << 12) - 1) == 0 {
        insts.push(Inst::AluRRImm12 {
            alu_op,
            rd: writable_sp_reg(),
            rn: sp_reg(),
            imm12: amount.try_into().unwrap(),
        });
    } else {
        for inst in Inst::load_constant(writable_ip_reg(), amount).into_vec() {
            insts.push(inst);
        }
        insts.push(Inst::AluRRRShift {
            alu_op,
            rd: writable_sp_reg(),
            rn: sp_reg(),
            rm: ip_reg(),
            shift: None,
        });
    }
    insts
}

impl ABICall for Arm32ABICall {
    type I = Inst;

    fn num_args(&self) -> usize {
        if self.sig.stack_ret_arg.is_some() {
            self.sig.args.len() - 1
        } else {
            self.sig.args.len()
        }
    }

    fn emit_stack_pre_adjust<C: LowerCtx<I = Self::I>>(&self, ctx: &mut C) {
        for inst in adjust_stack(self.sig.stack_arg_space, /* is_sub = */ true) {
            ctx.emit(inst);
        }
    }

    fn emit_stack_post_adjust<C: LowerCtx<I = Self::I>>(&self, ctx: &mut C) {
        for inst in adjust_stack(self.sig.stack_arg_space, /* is_sub = */ false) {
            ctx.emit(inst);
        }
    }

    fn emit_copy_reg_to_arg<C: LowerCtx<I = Self::I>>(
        &self,
        ctx: &mut C,
        idx: usize,
        from_reg: Reg,
    ) {
        match &self.sig.args[idx] {
            &ABIArg::Reg(reg, ty) => ctx.emit(Inst::gen_move(
                Writable::from_reg(reg.to_reg()),
                from_reg,
                ty,
            )),
            &ABIArg::Stack(off, ty) => {
                if let Some(mem) = MemArg::reg_maybe_offset(sp_reg(), off) {
                    ctx.emit(store_stack(mem, from_reg, ty));
                } else {
                    for inst in Inst::load_constant(writable_ip_reg(), off as u32) {
                        ctx.emit(inst);
                    }
                    let mem = MemArg::reg_plus_reg(sp_reg(), ip_reg(), 0);
                    ctx.emit(store_stack(mem, from_reg, ty));
                }
            }
        }
    }

    fn emit_copy_retval_to_reg<C: LowerCtx<I = Self::I>>(
        &self,
        ctx: &mut C,
        idx: usize,
        into_reg: Writable<Reg>,
    ) {
        match &self.sig.rets[idx] {
            &ABIArg::Reg(reg, ty) => ctx.emit(Inst::gen_move(into_reg, reg.to_reg(), ty)),
            &ABIArg::Stack(off, ty) => {
                if let Some(mem) = MemArg::reg_maybe_offset(sp_reg(), off) {
                    ctx.emit(load_stack(mem, into_reg, ty));
                } else {
                    for inst in Inst::load_constant(writable_ip_reg(), off as u32) {
                        ctx.emit(inst);
                    }
                    let mem = MemArg::reg_plus_reg(sp_reg(), ip_reg(), 0);
                    ctx.emit(load_stack(mem, into_reg, ty));
                }
            }
        }
    }

    fn emit_call<C: LowerCtx<I = Self::I>>(&mut self, ctx: &mut C) {
        let (uses, defs) = (
            mem::replace(&mut self.uses, vec![]),
            mem::replace(&mut self.defs, vec![]),
        );
        match &self.dest {
            &CallDest::ExtName(ref name, RelocDistance::Near) => ctx.emit(Inst::Call {
                dest: Box::new(name.clone()),
                uses: Box::new(uses),
                defs: Box::new(defs),
                loc: self.loc,
                opcode: self.opcode,
            }),
            &CallDest::ExtName(ref name, RelocDistance::Far) => {
                ctx.emit(Inst::LoadExtName {
                    rt: writable_ip_reg(),
                    name: name.clone(),
                    offset: 0,
                    srcloc: self.loc,
                });
                ctx.emit(Inst::CallInd {
                    rm: ip_reg(),
                    uses: Box::new(uses),
                    defs: Box::new(defs),
                    loc: self.loc,
                    opcode: self.opcode,
                });
            }
            &CallDest::Reg(reg) => ctx.emit(Inst::CallInd {
                rm: reg,
                uses: Box::new(uses),
                defs: Box::new(defs),
                loc: self.loc,
                opcode: self.opcode,
            }),
        }
    }
}
