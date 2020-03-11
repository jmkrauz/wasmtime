use crate::cdsl::cpu_modes::CpuMode;
use crate::cdsl::isa::TargetIsa;
use crate::cdsl::regs::{IsaRegs, IsaRegsBuilder, RegBankBuilder, RegClassBuilder};
use crate::cdsl::settings::{SettingGroup, SettingGroupBuilder};

use crate::shared::types::Bool::B1;
use crate::shared::types::Float::{F32, F64};
use crate::shared::types::Int::{I16, I32, I8};
use crate::shared::Definitions as SharedDefinitions;

mod encodings;
mod instructions;
mod legalize;
mod recipes;

fn define_settings(_shared: &SettingGroup) -> SettingGroup {
    let mut setting = SettingGroupBuilder::new("arm32");

    let supports_div = setting.add_bool(
        "supports_div",
        "CPU supports udiv/sdiv instructions.",
        false,
    );

    setting.add_predicate("use_div", predicate!(supports_div));

    setting.build()
}

fn define_regs() -> IsaRegs {
    let mut regs = IsaRegsBuilder::new();

    let builder = RegBankBuilder::new("FloatRegs", "s")
        .units(64)
        .track_pressure(true);
    let float_regs = regs.add_bank(builder);

    let builder = RegBankBuilder::new("IntRegs", "r")
        .units(16)
        .names(vec![
            "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13",
            "r14", "r15",
        ])
        .track_pressure(true);
    let int_regs = regs.add_bank(builder);

    let builder = RegBankBuilder::new("FlagRegs", "")
        .units(1)
        .names(vec!["nzcv"])
        .track_pressure(false);
    let flag_reg = regs.add_bank(builder);

    let builder = RegClassBuilder::new_toplevel("S", float_regs).count(32);
    regs.add_class(builder);

    let builder = RegClassBuilder::new_toplevel("D", float_regs).width(2);
    regs.add_class(builder);

    let builder = RegClassBuilder::new_toplevel("Q", float_regs).width(4);
    regs.add_class(builder);

    let builder = RegClassBuilder::new_toplevel("GPR", int_regs);
    regs.add_class(builder);

    let builder = RegClassBuilder::new_toplevel("FLAG", flag_reg);
    regs.add_class(builder);

    regs.build()
}

pub(crate) fn define(shared_defs: &mut SharedDefinitions) -> TargetIsa {
    let settings = define_settings(&shared_defs.settings);
    let regs = define_regs();

    let inst_group = instructions::define(
        &mut shared_defs.all_instructions,
        &shared_defs.formats,
        &shared_defs.imm,
    );
    legalize::define(shared_defs, &inst_group);

    // CPU modes for 32-bit ARM and Thumb2.
    let mut a32 = CpuMode::new("A32");
    let mut t32 = CpuMode::new("T32");

    let expand_flags = shared_defs.transform_groups.by_name("expand_flags");
    let widen = shared_defs.transform_groups.by_name("widen");

    let arm32_expand = shared_defs.transform_groups.by_name("arm32_expand");
    let arm32_narrow = shared_defs.transform_groups.by_name("arm32_narrow");

    a32.legalize_monomorphic(expand_flags);
    a32.legalize_default(arm32_narrow);
    a32.legalize_type(B1, expand_flags);
    a32.legalize_type(I8, widen);
    a32.legalize_type(I16, widen);
    a32.legalize_type(I32, arm32_expand);
    a32.legalize_type(F32, arm32_expand);
    a32.legalize_type(F64, arm32_expand);

    t32.legalize_default(arm32_narrow);

    let recipes = recipes::define(shared_defs, &regs);

    let encodings = encodings::define(shared_defs, &settings, &recipes, &inst_group);
    a32.set_encodings(encodings.enc_a32);
    t32.set_encodings(encodings.enc_t32);
    let encodings_predicates = encodings.inst_pred_reg.extract();

    let recipes = recipes.collect();

    let cpu_modes = vec![a32, t32];

    TargetIsa::new(
        "arm32",
        inst_group,
        settings,
        regs,
        recipes,
        cpu_modes,
        encodings_predicates,
    )
}
