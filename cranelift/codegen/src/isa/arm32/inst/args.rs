//! 32-bit ARM ISA definitions: instruction arguments.

use crate::binemit::CodeOffset;
use crate::ir::Type;
use crate::isa::arm32::inst::*;

use regalloc::{RealRegUniverse, Reg, Writable};

use core::convert::{Into, TryFrom};
use std::string::String;

/// A memory argument to load/store, encapsulating the possible addressing modes.
#[derive(Clone, Debug)]
pub enum MemArg {
    /// Register plus register offset.
    RegReg(Reg, Reg),

    /// Scaled (by size of a word) unsigned 5-bit immediate offset from reg.
    Offset5(Reg, u8),

    /// Scaled (by size of a word) unsigned 8-bit immediate offset from stack pointer.
    SPOffset(u8),
}

impl MemArg {
    /// Memory reference using an address in a register and an offset, if possible.
    pub fn reg_maybe_offset(reg: Reg, offset: i64, value_type: Type) -> Option<MemArg> {
        if (offset > 0x1f || offset < 0) {
            None
        } else {
            Some(MemArg::Offset5(reg, offset as u8))
        }
    }

    /// Memory reference using the sum of two registers as an address.
    pub fn reg_plus_reg(rn: Reg, rm: Reg) -> MemArg {
        MemArg::RegReg(rn, rm)
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
    pub fn bits(self) -> u32 {
        self as u32
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
    /// An unresolved reference to a BlockIndex, as passed into
    /// `lower_branch_group()`.
    Block(BlockIndex),
    /// A resolved reference to another instruction, after
    /// `Inst::with_block_offsets()`.
    ResolvedOffset(isize),
}

impl BranchTarget {
    /// Lower the branch target given offsets of each block.
    pub fn lower(&mut self, targets: &[CodeOffset], my_offset: CodeOffset) {
        match self {
            &mut BranchTarget::Block(bix) => {
                let bix = usize::try_from(bix).unwrap();
                assert!(bix < targets.len());
                let block_offset_in_func = targets[bix];
                let branch_offset = (block_offset_in_func as isize) - (my_offset as isize);
                *self = BranchTarget::ResolvedOffset(branch_offset);
            }
            &mut BranchTarget::ResolvedOffset(..) => {}
        }
    }

    /// Get the block index.
    pub fn as_block_index(&self) -> Option<BlockIndex> {
        match self {
            &BranchTarget::Block(bix) => Some(bix),
            _ => None,
        }
    }

    /// Get the offset as 2-byte halfwords. Returns `0` if not
    /// yet resolved (in that case, we're only computing
    /// size and the offset doesn't matter).
    pub fn as_offset_halfwords(&self) -> isize {
        match self {
            &BranchTarget::ResolvedOffset(off) => off << 1,
            _ => 0,
        }
    }

    /// Get the offset as a 11-bit offset suitable for an uncoditional jump, or `None` if overflow.
    pub fn as_off11(&self) -> Option<u16> {
        let off = self.as_offset_halfwords();
        if (off < (1 << 11)) && (off >= -(1 << 11)) {
            Some((off as u16) & ((1 << 11) - 1))
        } else {
            None
        }
    }

    /// Map the block index given a transform map.
    pub fn map(&mut self, block_index_map: &[BlockIndex]) {
        match self {
            &mut BranchTarget::Block(ref mut bix) => {
                let n = block_index_map[usize::try_from(*bix).unwrap()];
                *bix = n;
            }
            &mut BranchTarget::ResolvedOffset(_) => {}
        }
    }
}

impl ShowWithRRU for MemArg {
    fn show_rru(&self, mb_rru: Option<&RealRegUniverse>) -> String {
        match self {
            &MemArg::RegReg(rn, rm) => {
                format!("[{}, {}]", rn.show_rru(mb_rru), rm.show_rru(mb_rru),)
            }
            &MemArg::Offset5(rn, off) => format!("[{}, #{}]", rn.show_rru(mb_rru), off),
            &MemArg::SPOffset(off) => format!("[sp, #{}]", off),
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
            &BranchTarget::Block(block) => format!("block{}", block),
            &BranchTarget::ResolvedOffset(off) => format!("{}", off),
        }
    }
}
