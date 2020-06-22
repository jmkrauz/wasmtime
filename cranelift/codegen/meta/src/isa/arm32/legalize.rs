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
    let rotl = insts.by_name("rotl");
    let rotr = insts.by_name("rotr");
    let spill = insts.by_name("spill");

    let _imm = &shared.imm;

    // Custom legalization for rotations.
    narrow.custom_legalize(rotr, "narrow_rotr");
    narrow.custom_legalize(rotl, "narrow_rotl");

    expand.custom_legalize(spill, "expand_spill");

    expand.build_and_add_to(&mut shared.transform_groups);
    narrow.build_and_add_to(&mut shared.transform_groups);
}
