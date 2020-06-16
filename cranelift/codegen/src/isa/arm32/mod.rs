//! 32-bit ARM Instruction Set Architecture.

use crate::binemit;
use crate::ir::condcodes::IntCC;
use crate::ir::{self, Function};
use crate::isa::Builder as IsaBuilder;
use crate::isa::{EncInfo, Encoding, Encodings, Legalize, RegClass, RegInfo, TargetIsa};
use crate::machinst::{compile, MachBackend, MachCompileResult, ShowWithRRU, VCode};
use crate::regalloc::RegisterSet;
use crate::result::CodegenResult;
use crate::settings::{self, Flags};

#[cfg(feature = "testing_hooks")]
use crate::regalloc::RegDiversions;

use alloc::boxed::Box;
use core::any::Any;
use regalloc::RealRegUniverse;
use std::borrow::Cow;
use std::fmt;
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
        let abi = Box::new(abi::Arm32ABIBody::new(func, flags)?);
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
        let buffer = vcode.emit();
        let frame_size = vcode.frame_size();

        let disasm = if want_disasm {
            Some(vcode.show_rru(Some(&create_reg_universe())))
        } else {
            None
        };

        let buffer = buffer.finish();

        Ok(MachCompileResult {
            buffer,
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

    fn unsigned_add_overflow_condition(&self) -> IntCC {
        IntCC::UnsignedGreaterThanOrEqual
    }

    fn unsigned_sub_overflow_condition(&self) -> IntCC {
        IntCC::UnsignedLessThan
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
            Box::new(Arm32Isa::new(backend))
        },
    }
}

pub struct Arm32Isa {
    backend: Box<dyn MachBackend + Send + Sync + 'static>,
    triple: Triple,
}

impl Arm32Isa {
    pub fn new<B: MachBackend + Send + Sync + 'static>(backend: B) -> Arm32Isa {
        let triple = backend.triple();
        Arm32Isa {
            backend: Box::new(backend),
            triple,
        }
    }
}

impl fmt::Display for Arm32Isa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MachBackend")
            .field("name", &self.backend.name())
            .field("triple", &self.backend.triple())
            .field("flags", &format!("{}", self.backend.flags()))
            .finish()
    }
}

impl TargetIsa for Arm32Isa {
    fn name(&self) -> &'static str {
        self.backend.name()
    }

    fn triple(&self) -> &Triple {
        &self.triple
    }

    fn flags(&self) -> &Flags {
        self.backend.flags()
    }

    fn register_info(&self) -> RegInfo {
        // Called from function's Display impl, so we need a stub here.
        RegInfo {
            banks: &[],
            classes: &[],
        }
    }

    fn legal_encodings<'a>(
        &'a self,
        _func: &'a ir::Function,
        _inst: &'a ir::InstructionData,
        _ctrl_typevar: ir::Type,
    ) -> Encodings<'a> {
        panic!("Should not be called when new-style backend is available!")
    }

    fn encode(
        &self,
        func: &ir::Function,
        inst: &ir::InstructionData,
        ctrl_typevar: ir::Type,
    ) -> Result<Encoding, Legalize> {
        if let Some(action) = legalize::legalize_inst(func, inst, ctrl_typevar) {
            Err(action)
        } else {
            Ok(Encoding::new(0, 0))
        }
    }

    fn encoding_info(&self) -> EncInfo {
        panic!("Should not be called when new-style backend is available!")
    }

    fn legalize_signature(&self, sig: &mut Cow<ir::Signature>, _current: bool) {
        abi::legalize_signature(sig);
    }

    fn regclass_for_abi_type(&self, _ty: ir::Type) -> RegClass {
        panic!("Should not be called when new-style backend is available!")
    }

    fn allocatable_registers(&self, _func: &ir::Function) -> RegisterSet {
        panic!("Should not be called when new-style backend is available!")
    }

    fn prologue_epilogue(&self, _func: &mut ir::Function) -> CodegenResult<()> {
        panic!("Should not be called when new-style backend is available!")
    }

    #[cfg(feature = "testing_hooks")]
    fn emit_inst(
        &self,
        _func: &ir::Function,
        _inst: ir::Inst,
        _divert: &mut RegDiversions,
        _sink: &mut dyn binemit::CodeSink,
    ) {
        panic!("Should not be called when new-style backend is available!")
    }

    /// Emit a whole function into memory.
    fn emit_function_to_memory(&self, _func: &ir::Function, _sink: &mut binemit::MemoryCodeSink) {
        panic!("Should not be called when new-style backend is available!")
    }

    fn get_mach_backend(&self) -> Option<&dyn MachBackend> {
        Some(&*self.backend)
    }

    fn unsigned_add_overflow_condition(&self) -> ir::condcodes::IntCC {
        self.backend.unsigned_add_overflow_condition()
    }

    fn unsigned_sub_overflow_condition(&self) -> ir::condcodes::IntCC {
        self.backend.unsigned_sub_overflow_condition()
    }

    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
}
