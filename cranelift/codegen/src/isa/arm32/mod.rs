//! 32-bit ARM Instruction Set Architecture.

use crate::ir::{self, Function};
use crate::isa;
use crate::isa::Builder as IsaBuilder;
use crate::machinst::{
    compile, MachBackend, MachCompileResult, ShowWithRRU, TargetIsaAdapter, VCode,
};
use crate::result::CodegenResult;
use crate::settings;

use alloc::boxed::Box;
use regalloc::RealRegUniverse;
use std::borrow::Cow;
use target_lexicon::{Architecture, ArmArchitecture, Triple};

// New backend:
mod abi;
mod inst;
mod legalize;
mod lower;
mod lower_inst;

use inst::create_reg_universe;

/// An ARM32 backend.
pub struct Arm32Backend {
    triple: Triple,
    flags: settings::Flags,
    reg_universe: RealRegUniverse,
}

impl Arm32Backend {
    /// Create a new ARM32 backend with the given (shared) flags.
    pub fn new_with_flags(triple: Triple, flags: settings::Flags) -> Arm32Backend {
        let reg_universe = create_reg_universe();
        Arm32Backend {
            triple,
            flags,
            reg_universe,
        }
    }

    fn compile_vcode(
        &self,
        func: &Function,
        flags: settings::Flags,
    ) -> CodegenResult<VCode<inst::Inst>> {
        // This performs lowering to VCode, register-allocates the code, computes
        // block layout and finalizes branches. The result is ready for binary emission.
        let abi = Box::new(abi::Arm32ABIBody::new(func, flags));
        compile::compile::<Arm32Backend>(func, self, abi)
    }
}

impl MachBackend for Arm32Backend {
    fn compile_function(
        &self,
        func: &Function,
        want_disasm: bool,
    ) -> CodegenResult<MachCompileResult> {
        let flags = self.flags();
        let vcode = self.compile_vcode(func, flags.clone())?;
        let sections = vcode.emit();
        let frame_size = vcode.frame_size();

        let disasm = if want_disasm {
            Some(vcode.show_rru(Some(&create_reg_universe())))
        } else {
            None
        };

        Ok(MachCompileResult {
            sections,
            frame_size,
            disasm,
        })
    }

    fn name(&self) -> &'static str {
        "arm32"
    }

    fn triple(&self) -> Triple {
        self.triple.clone()
    }

    fn flags(&self) -> &settings::Flags {
        &self.flags
    }

    fn reg_universe(&self) -> &RealRegUniverse {
        &self.reg_universe
    }

    fn legalize_inst(
        &self,
        func: &ir::Function,
        inst: &ir::InstructionData,
        ctrl_typevar: ir::Type,
    ) -> Option<isa::Legalize> {
        legalize::legalize_inst(func, inst, ctrl_typevar)
    }

    fn legalize_signature(&self, sig: &mut Cow<ir::Signature>, _current: bool) {
        abi::legalize_signature(sig);
    }
}

/// Create a new `isa::Builder`.
pub fn isa_builder(triple: Triple) -> IsaBuilder {
    assert!(match triple.architecture {
        Architecture::Arm(ArmArchitecture::Arm) | Architecture::Arm(ArmArchitecture::Armv7) => true,
        _ => false,
    });
    IsaBuilder {
        triple,
        setup: settings::builder(),
        constructor: |triple, shared_flags, _| {
            let backend = Arm32Backend::new_with_flags(triple, shared_flags);
            Box::new(TargetIsaAdapter::new(backend))
        },
    }
}
