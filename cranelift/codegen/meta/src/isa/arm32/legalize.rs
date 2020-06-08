use crate::cdsl::xform::TransformGroupBuilder;
use crate::shared::Definitions as SharedDefinitions;

pub(crate) fn define(shared: &mut SharedDefinitions) {
    let mut expand = TransformGroupBuilder::new(
        "arm32_expand",
        r#"
    Legalize instructions by expansion.

    Use arm32-specific instructions if needed."#,
    )
    .isa("arm32")
    .chain_with(shared.transform_groups.by_name("expand_flags").id);

    let mut narrow = TransformGroupBuilder::new(
        "arm32_narrow",
        r#"
    Legalize instructions by narrowing.

    Use arm32-specific instructions if needed."#,
    )
    .isa("arm32")
    .chain_with(shared.transform_groups.by_name("narrow_flags").id);

    // List of instructions.
    let insts = &shared.instructions;
    let ishl = insts.by_name("ishl");
    let rotl = insts.by_name("rotl");
    let rotr = insts.by_name("rotr");
    let sdiv = insts.by_name("sdiv");
    let spill = insts.by_name("spill");
    let srem = insts.by_name("srem");
    let sshr = insts.by_name("sshr");
    let udiv = insts.by_name("udiv");
    let urem = insts.by_name("urem");
    let ushr = insts.by_name("ushr");

    let _imm = &shared.imm;

    // Custom legalization for division and remainder computation.
    narrow.custom_legalize(udiv, "expand_udiv_urem");
    narrow.custom_legalize(urem, "expand_udiv_urem");
    narrow.custom_legalize(sdiv, "expand_sdiv_srem");
    narrow.custom_legalize(srem, "expand_sdiv_srem");

    // Custom legalization for shifts and rotates.
    narrow.custom_legalize(ishl, "narrow_ishl");
    narrow.custom_legalize(ushr, "narrow_ushr");
    narrow.custom_legalize(sshr, "narrow_sshr");
    narrow.custom_legalize(rotr, "narrow_rotr");
    narrow.custom_legalize(rotl, "narrow_rotl");

    expand.custom_legalize(spill, "expand_spill");

    expand.build_and_add_to(&mut shared.transform_groups);
    narrow.build_and_add_to(&mut shared.transform_groups);
}
