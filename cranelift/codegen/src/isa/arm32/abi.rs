//! Implementation of the 32-bit ARM ABI.

use crate::ir;
use crate::ir::types;
use crate::ir::types::*;
use crate::ir::StackSlot;
use crate::isa;
use crate::isa::arm32::inst::*;
use crate::machinst::*;
use crate::settings;

use alloc::vec::Vec;

use regalloc::{RealReg, Reg, RegClass, Set, SpillSlot, Writable};

use log::debug;

/// A location for an argument or return value.
#[derive(Clone, Copy, Debug)]
enum ABIArg {
    /// In a real register.
    Reg(RealReg, ir::Type),
    /// Arguments only: on stack, at given offset from SP at entry.
    Stack(i64, ir::Type),
}

/// Arm ABI information shared between body (callee) and caller.
struct ABISig {
    args: Vec<ABIArg>,
    rets: Vec<ABIArg>,
    stack_arg_space: i64,
    call_conv: isa::CallConv,
}

/// Process a list of parameters or return values and allocate them to R-regs, S-regs, D-regs
/// and stack slots.
///
/// Returns the list of argument locations, and the stack-space used (rounded up
/// to a 16-byte-aligned boundary).
#[allow(unused)]
fn compute_arg_locs(call_conv: isa::CallConv, params: &[ir::AbiParam]) -> (Vec<ABIArg>, i64) {
    // See AAPCS ABI
    // http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ihi0042f/index.html
    
    unimplemented!()
}

impl ABISig {
    fn from_func_sig(sig: &ir::Signature) -> ABISig {
        // Compute args and retvals from signature.
        // TODO: pass in arg-mode or ret-mode. (Does not matter
        // for the types of arguments/return values that we support.)
        let (args, stack_arg_space) = compute_arg_locs(sig.call_conv, &sig.params);
        let (rets, _) = compute_arg_locs(sig.call_conv, &sig.returns);

        // Verify that there are no return values on the stack.
        assert!(rets.iter().all(|a| match a {
            &ABIArg::Stack(..) => false,
            _ => true,
        }));

        ABISig {
            args,
            rets,
            stack_arg_space,
            call_conv: sig.call_conv,
        }
    }
}

/// ARM ABI object for a function body.
pub struct ArmABIBody {
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
    frame_size: Option<u32>,
    /// Calling convention this function expects.
    call_conv: isa::CallConv,
}

fn in_int_reg(ty: ir::Type) -> bool {
    match ty {
        types::I8 | types::I16 | types::I32 | types::I64 => true,
        types::B1 | types::B8 | types::B16 | types::B32 | types::B64 => true,
        _ => false,
    }
}

fn in_float_reg(ty: ir::Type) -> bool {
    match ty {
        types::F32 | types::F64 => true,
        _ => false,
    }
}

impl ArmABIBody {
    /// Create a new body ABI instance.
    pub fn new(f: &ir::Function) -> Self {
        debug!("AArch64 ABI: func signature {:?}", f.signature);

        let sig = ABISig::from_func_sig(&f.signature);

        let call_conv = f.signature.call_conv;
        // Only these calling conventions are supported.
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

        Self {
            sig,
            stackslots,
            stackslots_size: stack_offset,
            clobbered: Set::empty(),
            spillslots: None,
            frame_size: None,
            call_conv,
        }
    }
}

#[allow(unused)]
impl ABIBody for ArmABIBody {
    type I = Inst;

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
        unimplemented!()
    }

    fn gen_copy_reg_to_retval(&self, idx: usize, from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn gen_ret(&self) -> Inst {
        unimplemented!()
    }

    fn gen_epilogue_placeholder(&self) -> Inst {
        unimplemented!()
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
        unimplemented!()
    }

    fn store_stackslot(&self, slot: StackSlot, offset: u32, ty: Type, from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn stackslot_addr(&self, slot: StackSlot, offset: u32, into_reg: Writable<Reg>) -> Inst {
        unimplemented!()
    }

    // Load from a spillslot.
    fn load_spillslot(&self, slot: SpillSlot, ty: Type, into_reg: Writable<Reg>) -> Inst {
        unimplemented!()
    }

    // Store to a spillslot.
    fn store_spillslot(&self, slot: SpillSlot, ty: Type, from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn gen_prologue(&mut self, flags: &settings::Flags) -> Vec<Inst> {
        unimplemented!()
    }

    fn gen_epilogue(&self, _flags: &settings::Flags) -> Vec<Inst> {
        unimplemented!()
    }

    fn frame_size(&self) -> u32 {
        self.frame_size
            .expect("frame size not computed before prologue generation")
    }

    fn get_spillslot_size(&self, rc: RegClass, ty: Type) -> u32 {
        unimplemented!()
    }

    fn gen_spill(&self, to_slot: SpillSlot, from_reg: RealReg, ty: Type) -> Inst {
        self.store_spillslot(to_slot, ty, from_reg.to_reg())
    }

    fn gen_reload(&self, to_reg: Writable<RealReg>, from_slot: SpillSlot, ty: Type) -> Inst {
        self.load_spillslot(from_slot, ty, to_reg.map(|r| r.to_reg()))
    }
}
