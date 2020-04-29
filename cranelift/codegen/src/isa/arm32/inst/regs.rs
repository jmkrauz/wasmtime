//! 32-bit ARM ISA definitions: registers.

use regalloc::{RealRegUniverse, Reg, RegClass, RegClassInfo, Writable, NUM_REG_CLASSES};

use std::string::ToString;

// r9 is an additional callee-saved variable register.

/// Get a reference to an r-register (integer register).
pub fn rreg(num: u8) -> Reg {
    assert!(num < 16);
    Reg::new_real(RegClass::I32, /* enc = */ num, /* index = */ num)
}

/// Get a writable reference to an r-register.
pub fn writable_rreg(num: u8) -> Writable<Reg> {
    Writable::from_reg(rreg(num))
}

/// Get a reference to the Intra-Procedure-call scratch register (r12), which is used as temporary
/// register by various instructions.
pub fn ip_reg() -> Reg {
    rreg(12)
}

/// Get a writable reference to the Intra-Procedure-call scratch register.
pub fn writable_ip_reg() -> Writable<Reg> {
    Writable::from_reg(ip_reg())
}

/// Get a reference to the stack pointer (r13).
pub fn sp_reg() -> Reg {
    rreg(13)
}

/// Get a writable reference to the stack pointer.
pub fn writable_sp_reg() -> Writable<Reg> {
    Writable::from_reg(sp_reg())
}

/// Get a reference to the link register (r14).
pub fn lr_reg() -> Reg {
    rreg(14)
}

/// Get a writable reference to the link register.
pub fn writable_lr_reg() -> Writable<Reg> {
    Writable::from_reg(lr_reg())
}

/// Get a reference to the program counter (r15).
pub fn pc_reg() -> Reg {
    rreg(15)
}

/// Get a writable reference to the program counter.
pub fn writable_pc_reg() -> Writable<Reg> {
    Writable::from_reg(pc_reg())
}

/// Create the register universe for ARM32.
pub fn create_reg_universe() -> RealRegUniverse {
    let mut regs = vec![];
    let mut allocable_by_class = [None; NUM_REG_CLASSES];

    let r_reg_base = 0u8; // in contiguous real-register index space
    let r_reg_count = 12; // to exclude ip, sp, lr  and pc.
    for i in 0u8..r_reg_count {
        let reg = Reg::new_real(
            RegClass::I32,
            /* enc = */ i,
            /* index = */ r_reg_base + i,
        )
        .to_real_reg();
        let name = format!("r{}", i);
        regs.push((reg, name));
    }
    let r_reg_last = r_reg_base + r_reg_count - 1;

    allocable_by_class[RegClass::I32.rc_to_usize()] = Some(RegClassInfo {
        first: r_reg_base as usize,
        last: r_reg_last as usize,
        suggested_scratch: None,
    });

    // Other regs, not available to the allocator.
    let allocable = regs.len();
    regs.push((ip_reg().to_real_reg(), "ip".to_string()));
    regs.push((sp_reg().to_real_reg(), "sp".to_string()));
    regs.push((lr_reg().to_real_reg(), "lr".to_string()));
    regs.push((pc_reg().to_real_reg(), "pc".to_string()));

    // Assert sanity: the indices in the register structs must match their
    // actual indices in the array.
    for (i, reg) in regs.iter().enumerate() {
        assert_eq!(i, reg.0.get_index());
    }

    RealRegUniverse {
        regs,
        allocable,
        allocable_by_class,
    }
}
