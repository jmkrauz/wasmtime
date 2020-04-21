//! 32-bit ARM Instruction Set Architecture.

use crate::ir::Function;
use crate::isa::Builder as IsaBuilder;
use crate::machinst::{
    compile, MachBackend, MachCompileResult, ShowWithRRU, TargetIsaAdapter, VCode,
};
use crate::result::CodegenResult;
use crate::settings;

use alloc::boxed::Box;

use regalloc::RealRegUniverse;
use target_lexicon::{ArmArchitecture, Architecture, Triple};

// New backend:
mod abi;
mod inst;
mod lower;

use inst::create_reg_universe;

/// An ARM backend.
pub struct ArmBackend {
    triple: Triple,
    flags: settings::Flags,
}

impl ArmBackend {
    /// Create a new ARM backend with the given (shared) flags.
    pub fn new_with_flags(triple: Triple, flags: settings::Flags) -> ArmBackend {
        ArmBackend { triple, flags }
    }

    fn compile_vcode(&self, func: &Function, flags: &settings::Flags) -> VCode<inst::Inst> {
        // This performs lowering to VCode, register-allocates the code, computes
        // block layout and finalizes branches. The result is ready for binary emission.
        let abi = Box::new(abi::ArmABIBody::new(func));
        compile::compile::<ArmBackend>(func, self, abi, flags)
    }
}

impl MachBackend for ArmBackend {
    fn compile_function(
        &self,
        func: &Function,
        want_disasm: bool,
    ) -> CodegenResult<MachCompileResult> {
        let flags = self.flags();
        let vcode = self.compile_vcode(func, flags);
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

    fn reg_universe(&self) -> RealRegUniverse {
        create_reg_universe()
    }
}

/// Create a new `isa::Builder`.
pub fn isa_builder(triple: Triple) -> IsaBuilder {
    assert!(triple.architecture == Architecture::Arm(ArmArchitecture::Armv7));
    IsaBuilder {
        triple,
        setup: settings::builder(),
        constructor: |triple, shared_flags, _| {
            let backend = ArmBackend::new_with_flags(triple, shared_flags);
            Box::new(TargetIsaAdapter::new(backend))
        },
    }
}
