//! 32-bit ARM ISA definitions: registers.

use super::Precision;
use crate::machinst::pretty_print::ShowWithRRU;

use regalloc::{RealRegUniverse, Reg, RegClass, RegClassInfo, Writable, NUM_REG_CLASSES};

use std::string::{String, ToString};

// r9 is an additional callee-saved variable register.

pub const DREG_COUNT: u8 = 16;

/// Get a reference to an r-register (integer register).
pub fn rreg(num: u8) -> Reg {
    assert!(num < 16);
    Reg::new_real(
        RegClass::I32,
        /* enc = */ num,
        /* index = */ DREG_COUNT + num,
    )
}

/// Get a writable reference to an r-register.
pub fn writable_rreg(num: u8) -> Writable<Reg> {
    Writable::from_reg(rreg(num))
}

/// Get a reference to a d-register (FPU register).
pub fn dreg(num: u8) -> Reg {
    assert!(num < DREG_COUNT);
    Reg::new_real(RegClass::F64, /* enc = */ num, /* index = */ num)
}

/// Get a writable reference to a d-register.
pub fn writable_dreg(num: u8) -> Writable<Reg> {
    Writable::from_reg(dreg(num))
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

    let d_reg_base = 0u8;
    for i in 0u8..DREG_COUNT {
        let reg = Reg::new_real(RegClass::F64, i, d_reg_base + i).to_real_reg();
        let name = format!("d{}", i);
        regs.push((reg, name));
    }
    let d_reg_last = d_reg_base + DREG_COUNT - 1;

    let r_reg_base = d_reg_last + 1;
    assert!(r_reg_base == DREG_COUNT);
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

    allocable_by_class[RegClass::F64.rc_to_usize()] = Some(RegClassInfo {
        first: d_reg_base as usize,
        last: d_reg_last as usize,
        suggested_scratch: None,
    });
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

/// Show an FPU register.
pub fn show_freg_with_precision(
    reg: Reg,
    mb_rru: Option<&RealRegUniverse>,
    precision: Precision,
) -> String {
    let s = reg.show_rru(mb_rru);
    if reg.get_class() != RegClass::F64 {
        return s;
    }
    if precision == Precision::Single && reg.is_real() {
        format!("s{}", 2 * reg.to_real_reg().get_hw_encoding())
    } else {
        s
    }
}

/// Show a higher single precision registers from pair that creates `reg`. 
pub fn show_freg_single_hi(reg: Reg, mb_rru: Option<&RealRegUniverse>) -> String {
    let s = if reg.is_real() {
        format!("s{}", 1 + 2 * reg.to_real_reg().get_hw_encoding())
    } else {
        format!("{} hi", reg.show_rru(mb_rru))
    };
    s
}