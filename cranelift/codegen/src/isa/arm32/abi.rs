//! Implementation of the 32-bit ARM ABI.

use crate::ir;
use crate::ir::types;
use crate::ir::types::*;
use crate::ir::{ArgumentExtension, StackSlot};
use crate::isa;
use crate::isa::arm32::{self, inst::*};
use crate::machinst::*;
use crate::settings;

use alloc::vec::Vec;
use core::convert::TryInto;
use core::mem;
use log::debug;
use regalloc::{RealReg, Reg, RegClass, Set, SpillSlot, Writable};

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

/// Process a list of parameters or return values and allocate them to R-regs and stack slots.
///
/// Returns the list of argument locations, and the stack-space used (rounded up
/// to a 8-byte-aligned boundary).
fn compute_arg_locs(params: &[ir::AbiParam], arg_mode: bool) -> (Vec<ABIArg>, u32) {
    // See AAPCS ABI https://developer.arm.com/docs/ihi0042/latest
    // r9 is an additional callee-saved variable register.
    let mut next_rreg = 0;
    let mut next_stack: u32 = 0;
    let mut ret = vec![];

    let max_rreg = if arg_mode {
        3 // use r0-r3 for arguments
    } else {
        1 // use r0-r1 for returns
    };

    for param in params {
        // Validate "purpose".
        match &param.purpose {
            &ir::ArgumentPurpose::VMContext | &ir::ArgumentPurpose::Normal => {}
            _ => panic!(
                "Unsupported argument purpose {:?} in signature: {:?}",
                param.purpose, params
            ),
        }

        if in_int_reg(param.value_type) {
            if next_rreg <= max_rreg {
                ret.push(ABIArg::Reg(rreg(next_rreg).to_real_reg(), param.value_type));
                next_rreg += 1;
            } else {
                ret.push(ABIArg::Stack(
                    next_stack.try_into().unwrap(),
                    param.value_type,
                ));
                next_stack += 4;
            }
        } else {
            unimplemented!("param value type {}", param.value_type)
        }
    }

    next_stack = (next_stack + 7) & !7;

    (ret, next_stack)
}

impl ABISig {
    fn from_func_sig(sig: &ir::Signature) -> ABISig {
        // Compute args and retvals from signature.
        let (args, stack_arg_space) = compute_arg_locs(&sig.params, true);
        let (rets, _) = compute_arg_locs(&sig.returns, false);

        // Verify that there are no return values on the stack.
        assert!(rets.iter().all(|a| match a {
            &ABIArg::Stack(..) => false,
            _ => true,
        }));

        ABISig {
            args,
            rets,
            stack_arg_space,
        }
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
    /// Calling convention this function expects.
    call_conv: isa::CallConv,
    /// The settings controlling this function's compilation.
    flags: settings::Flags,
    /// Whether or not this function is a "leaf", meaning it calls no other
    /// functions
    is_leaf: bool,
    /// If this function has a stack limit specified, then `Reg` is where the
    /// stack limit will be located after the instructions specified have been
    /// executed.
    ///
    /// Note that this is intended for insertion into the prologue, if
    /// present. Also note that because the instructions here execute in the
    /// prologue this happens after legalization/register allocation/etc so we
    /// need to be extremely careful with each instruction. The instructions are
    /// manually register-allocated and carefully only use caller-saved
    /// registers and keep nothing live after this sequence of instructions.
    stack_limit: Option<(Reg, Vec<Inst>)>,
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
        types::B1
        | types::B8
        | types::I8
        | types::B16
        | types::I16
        | types::B32
        | types::I32 => Inst::Load {
            rt: into_reg,
            mem,
            srcloc: None,
            bits: 32,
            sign_extend: false,
        },
        _ => unimplemented!("load_stack({})", ty),
    }
}

fn store_stack(mem: MemArg, from_reg: Reg, ty: Type) -> Inst {
    match ty {
        types::B1
        | types::B8
        | types::I8
        | types::B16
        | types::I16
        | types::B32
        | types::I32 => Inst::Store {
            rt: from_reg,
            mem,
            srcloc: None,
            bits: 32
        },
        _ => unimplemented!("store_stack({})", ty),
    }
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

fn get_caller_saves_set() -> Set<Writable<Reg>> {
    let mut set = Set::empty();
    for i in 0..15 {
        let r = writable_rreg(i);
        if is_caller_save(r.to_reg().to_real_reg()) {
            set.insert(r);
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
                let mem = if let Some(mem) = MemArg::reg_maybe_offset(base, offset)
                {
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
    pub fn new(f: &ir::Function, flags: settings::Flags) -> Self {
        debug!("Arm32 ABI: func signature {:?}", f.signature);

        let sig = ABISig::from_func_sig(&f.signature);

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

        // Figure out what instructions, if any, will be needed to check the
        // stack limit. This can either be specified as a special-purpose
        // argument or as a global value which often calculates the stack limit
        // from the arguments.
        let stack_limit =
            get_special_purpose_param_register(f, &sig, ir::ArgumentPurpose::StackLimit)
                .map(|reg| (reg, Vec::new()))
                .or_else(|| f.stack_limit.map(|gv| gen_stack_limit(f, &sig, gv)));

        Self {
            sig,
            stackslots,
            stackslots_size: stack_offset,
            clobbered: Set::empty(),
            spillslots: None,
            total_frame_size: None,
            call_conv,
            flags,
            is_leaf: f.is_leaf(),
            stack_limit,
        }
    }
}

impl ABIBody for Arm32ABIBody {
    type I = Inst;

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
        _slot: StackSlot,
        _offset: u32,
        _ty: Type,
        _into_reg: Writable<Reg>,
    ) -> Inst {
        unimplemented!()
    }

    fn store_stackslot(&self, _slot: StackSlot, _offset: u32, _ty: Type, _from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn stackslot_addr(&self, _slot: StackSlot, _offset: u32, _into_reg: Writable<Reg>) -> Inst {
        unimplemented!()
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
        self.total_frame_size = Some(0);
        vec![]
    }

    fn gen_epilogue(&self) -> Vec<Inst> {
        vec![Inst::Ret]
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
    uses: Set<Reg>,
    defs: Set<Writable<Reg>>,
    dest: CallDest,
    loc: ir::SourceLoc,
    opcode: ir::Opcode,
}

fn abisig_to_uses_and_defs(sig: &ABISig) -> (Set<Reg>, Set<Writable<Reg>>) {
    // Compute uses: all arg regs.
    let mut uses = Set::empty();
    for arg in &sig.args {
        match arg {
            &ABIArg::Reg(reg, _) => uses.insert(reg.to_reg()),
            _ => {}
        }
    }

    // Compute defs: all retval regs, and all caller-save (clobbered) regs.
    let mut defs = get_caller_saves_set();
    for ret in &sig.rets {
        match ret {
            &ABIArg::Reg(reg, _) => defs.insert(Writable::from_reg(reg.to_reg())),
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
    ) -> Arm32ABICall {
        let sig = ABISig::from_func_sig(sig);
        let (uses, defs) = abisig_to_uses_and_defs(&sig);
        Arm32ABICall {
            sig,
            uses,
            defs,
            dest: CallDest::ExtName(extname.clone(), dist),
            loc,
            opcode: ir::Opcode::Call,
        }
    }

    /// Create a callsite ABI object for a call to a function pointer with the
    /// given signature.
    pub fn from_ptr(
        sig: &ir::Signature,
        ptr: Reg,
        loc: ir::SourceLoc,
        opcode: ir::Opcode,
    ) -> Arm32ABICall {
        let sig = ABISig::from_func_sig(sig);
        let (uses, defs) = abisig_to_uses_and_defs(&sig);
        Arm32ABICall {
            sig,
            uses,
            defs,
            dest: CallDest::Reg(ptr),
            loc,
            opcode,
        }
    }
}

fn adjust_stack<C: LowerCtx<I = Inst>>(ctx: &mut C, amount: u32, is_sub: bool) {
    if amount == 0 {
        return;
    }
    let alu_op = if is_sub { ALUOp::Sub } else { ALUOp::Add };
        for inst in Inst::load_constant(writable_ip_reg(), amount).into_vec() {
            ctx.emit(inst);
        }
        ctx.emit(Inst::AluRRRShift {
            alu_op,
            rd: writable_sp_reg(),
            rn: sp_reg(),
            rm: ip_reg(),
            shift: None,
        });
}

impl ABICall for Arm32ABICall {
    type I = Inst;

    fn num_args(&self) -> usize {
        self.sig.args.len()
    }

    fn emit_stack_pre_adjust<C: LowerCtx<I = Self::I>>(&self, ctx: &mut C) {
        adjust_stack(
            ctx,
            self.sig.stack_arg_space,
            /* is_sub = */ true,
        )
    }

    fn emit_stack_post_adjust<C: LowerCtx<I = Self::I>>(&self, ctx: &mut C) {
        adjust_stack(
            ctx,
            self.sig.stack_arg_space,
            /* is_sub = */ false,
        )
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
            _ => unimplemented!(),
        }
    }

    fn emit_call<C: LowerCtx<I = Self::I>>(&mut self, ctx: &mut C) {
        let (uses, defs) = (
            mem::replace(&mut self.uses, Set::empty()),
            mem::replace(&mut self.defs, Set::empty()),
        );
        match &self.dest {
            &CallDest::ExtName(ref name, RelocDistance::Near) => ctx.emit(Inst::Call {
                dest: name.clone(),
                uses,
                defs,
                loc: self.loc,
                opcode: self.opcode,
            }),
            &CallDest::ExtName(ref _name, RelocDistance::Far) => {
                unimplemented!()
            }
            &CallDest::Reg(reg) => ctx.emit(Inst::CallInd {
                rm: reg,
                uses,
                defs,
                loc: self.loc,
                opcode: self.opcode,
            }),
        }
    }
}
