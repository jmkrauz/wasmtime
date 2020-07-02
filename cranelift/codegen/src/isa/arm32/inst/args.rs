//! 32-bit ARM ISA definitions: instruction arguments.

use crate::isa::arm32::inst::*;

use regalloc::{RealRegUniverse, Reg};

use std::string::String;

/// A shift operator for a register or immediate.
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ShiftOp {
    LSL = 0b00,
    LSR = 0b01,
    ASR = 0b10,
    ROR = 0b11,
}

impl ShiftOp {
    /// Get the encoding of this shift op.
    pub fn bits(self) -> u8 {
        self as u8
    }
}

/// A shift operator amount.
#[derive(Clone, Copy, Debug)]
pub struct ShiftOpShiftImm(u8);

impl ShiftOpShiftImm {
    /// Maximum shift for shifted-register operands.
    pub const MAX_SHIFT: u32 = 31;

    /// Create a new shiftop shift amount, if possible.
    pub fn maybe_from_shift(shift: u32) -> Option<ShiftOpShiftImm> {
        if shift <= Self::MAX_SHIFT {
            Some(ShiftOpShiftImm(shift as u8))
        } else {
            None
        }
    }

    /// Return the shift amount.
    pub fn value(self) -> u8 {
        self.0
    }
}

/// A shift operator with an amount, guaranteed to be within range.
#[derive(Clone, Debug)]
pub struct ShiftOpAndAmt {
    op: ShiftOp,
    shift: ShiftOpShiftImm,
}

impl ShiftOpAndAmt {
    pub fn new(op: ShiftOp, shift: ShiftOpShiftImm) -> ShiftOpAndAmt {
        ShiftOpAndAmt { op, shift }
    }

    /// Get the shift op.
    pub fn op(&self) -> ShiftOp {
        self.op
    }

    /// Get the shift amount.
    pub fn amt(&self) -> ShiftOpShiftImm {
        self.shift
    }
}

/// A memory argument to load/store, encapsulating the possible addressing modes.
#[derive(Clone, Debug)]
pub enum MemArg {
    /// Register plus register offset, which can be shifted by imm2.
    RegReg(Reg, Reg, u32),

    /// Unsigned 12-bit immediate offset from reg.
    Offset12(Reg, i32),

    SPOffset(i32),

    NominalSPOffset(i32),
}

impl MemArg {
    /// Memory reference using an address in a register and an offset, if possible.
    pub fn reg_maybe_offset(reg: Reg, offset: i32) -> Option<MemArg> {
        if offset >= (1 << 12) || offset < 0 {
            None
        } else {
            Some(MemArg::Offset12(reg, offset))
        }
    }

    /// Memory reference using the sum of two registers as an address.
    pub fn reg_plus_reg(rn: Reg, rm: Reg, imm2: u32) -> MemArg {
        if (imm2 & !0b11) != 0 {
            panic!("Invalid shift amount {}", imm2);
        }
        MemArg::RegReg(rn, rm, imm2)
    }
}

//=============================================================================
// Instruction sub-components (conditions, branches and branch targets):
// definitions

/// Condition for conditional branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    Eq = 0,
    Ne = 1,
    Hs = 2,
    Lo = 3,
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
    Al = 14,
}

impl Cond {
    /// Return the inverted condition.
    pub fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,

            Cond::Hs => Cond::Lo,
            Cond::Lo => Cond::Hs,

            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,

            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,

            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,

            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,

            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,

            Cond::Al => panic!("No inverse of Al condition"),
        }
    }

    /// Return the machine encoding of this condition.
    pub fn bits(self) -> u16 {
        self as u16
    }
}

/// The kind of conditional branch: the common-case-optimized "reg-is-zero" /
/// "reg-is-nonzero" variants, or the generic one that tests the machine
/// condition codes.
#[derive(Clone, Copy, Debug)]
pub enum CondBrKind {
    /// Condition: given register is zero.
    Zero(Reg),
    /// Condition: given register is nonzero.
    NotZero(Reg),
    /// Condition: the given condition-code test is true.
    Cond(Cond),
}

impl CondBrKind {
    /// Return the inverted branch condition.
    pub fn invert(self) -> CondBrKind {
        match self {
            CondBrKind::Zero(reg) => CondBrKind::NotZero(reg),
            CondBrKind::NotZero(reg) => CondBrKind::Zero(reg),
            CondBrKind::Cond(c) => CondBrKind::Cond(c.invert()),
        }
    }
}

/// A branch target. Either unresolved (basic-block index) or resolved (offset
/// from end of current instruction).
#[derive(Clone, Copy, Debug)]
pub enum BranchTarget {
    /// An unresolved reference to a Label, as passed into
    /// `lower_branch_group()`.
    Label(MachLabel),
    /// A fixed PC offset.
    ResolvedOffset(i32),
}

impl BranchTarget {
    /// Return the target's label, if it is a label-based target.
    pub fn as_label(self) -> Option<MachLabel> {
        match self {
            BranchTarget::Label(l) => Some(l),
            _ => None,
        }
    }

    // Ready for embedding in istruction.
    fn as_offset(self) -> i32 {
        match self {
            // subtract 4 becasuse PC is current inst
            BranchTarget::ResolvedOffset(off) => ((off - 4) >> 1),
            _ => 0,
        }
    }

    // For 32-bit unconditional jump.
    pub fn as_off24(self) -> u32 {
        let off = self.as_offset();
        assert!(off < (1 << 24));
        assert!(off >= -(1 << 24));
        (off as u32) & ((1 << 24) - 1)
    }

    // For 32-bit conditional jump.
    pub fn as_off20(self) -> u32 {
        let off = self.as_offset();
        assert!(off < (1 << 20));
        assert!(off >= -(1 << 20));
        (off as u32) & ((1 << 20) - 1)
    }

    // For cbz/cbnz.
    pub fn as_off6(&self) -> u16 {
        let off = self.as_offset();
        assert!(off < (1 << 6));
        assert!(off >= 0);
        (off as u16) & ((1 << 6) - 1)
    }

    // For 16-bit unconditional jump.
    // Returns Option type to help emit short jump if it is possible.
    pub fn as_off11(self) -> Option<u16> {
        if let BranchTarget::Label(_) = self {
            return None;
        }
        let off = self.as_offset();

        if (off < (1 << 11)) && (off >= -(1 << 11)) {
            Some((off as u16) & ((1 << 11) - 1))
        } else {
            None
        }
    }
}

impl ShowWithRRU for ShiftOpAndAmt {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        format!("{:?} {}", self.op(), self.amt().value())
    }
}

impl ShowWithRRU for MemArg {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            &MemArg::RegReg(rn, rm, imm2) => {
                let shift = if imm2 != 0 {
                    format!(", LSL #{}", imm2)
                } else {
                    "".to_string()
                };
                format!(
                    "[{}, {}{}]",
                    rn.show_rru(mb_rru),
                    rm.show_rru(mb_rru),
                    shift
                )
            }
            &MemArg::Offset12(rn, off) => format!("[{}, #{}]", rn.show_rru(mb_rru), off),
            &MemArg::SPOffset(_) | &MemArg::NominalSPOffset(_) => panic!("unexpected mem mode"),
        }
    }
}

impl ShowWithRRU for Cond {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        let mut s = format!("{:?}", self);
        s.make_ascii_lowercase();
        s
    }
}

impl ShowWithRRU for BranchTarget {
    fn show_rru(&self, _mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            &BranchTarget::Label(label) => format!("label{:?}", label),
            &BranchTarget::ResolvedOffset(off) => format!("{}", off),
        }
    }
}
