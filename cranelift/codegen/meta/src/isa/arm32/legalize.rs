use crate::cdsl::ast::{var, ExprBuilder, Literal};
use crate::cdsl::instructions::Bindable;
use crate::cdsl::xform::TransformGroupBuilder;
use crate::shared::types::Int::{I16, I32, I64, I8};
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

    let imm = &shared.imm;

    // List of instructions.
    let insts = &shared.instructions;

    let iconcat = insts.by_name("iconcat");
    let iconst = insts.by_name("iconst");
    let ireduce = insts.by_name("ireduce");
    let istore8 = insts.by_name("istore8");
    let istore16 = insts.by_name("istore16");
    let istore32 = insts.by_name("istore32");
    let load = insts.by_name("load");
    let rotl = insts.by_name("rotl");
    let rotr = insts.by_name("rotr");
    let sextend = insts.by_name("sextend");
    let sload8 = insts.by_name("sload8");
    let sload16 = insts.by_name("sload16");
    let sload32 = insts.by_name("sload32");
    let spill = insts.by_name("spill");
    let sshr_imm = insts.by_name("sshr_imm");
    let store = insts.by_name("store");
    let uextend = insts.by_name("uextend");
    let uload8 = insts.by_name("uload8");
    let uload16 = insts.by_name("uload16");
    let uload32 = insts.by_name("uload32");

    let a = var("a");
    let al = var("al");
    let ah = var("ah");
    let x = var("x");
    let xl = var("xl");
    let flags = var("flags");
    let offset = var("off");
    let ptr = var("ptr");

    for &(op, ty) in &[(uload8, I8), (uload16, I16), (uload32, I32)] {
        narrow.legalize(
            def!(x = op.I64(flags, ptr, offset)),
            vec![
                def!(xl = load.ty(flags, ptr, offset)),
                def!(x = uextend.I64(xl)),
            ],
        );
    }

    for &(op, ty) in &[(sload8, I8), (sload16, I16), (sload32, I32)] {
        narrow.legalize(
            def!(x = op.I64(flags, ptr, offset)),
            vec![
                def!(xl = load.ty(flags, ptr, offset)),
                def!(x = sextend.I64(xl)),
            ],
        );
    }

    for &(op, ty) in &[(istore8, I8), (istore16, I16), (istore32, I32)] {
        narrow.legalize(
            def!(op.I64(flags, x, ptr, offset)),
            vec![
                def!(xl = ireduce.ty(x)),
                def!(store.ty(flags, xl, ptr, offset)),
            ],
        );
    }

    // Expand not narrow, because control variable type is I32.
    expand.legalize(
        def!(x = uload32(flags, ptr, offset)),
        vec![
            def!(xl = load.I32(flags, ptr, offset)),
            def!(x = uextend(xl)),
        ],
    );
    expand.legalize(
        def!(x = sload32(flags, ptr, offset)),
        vec![
            def!(xl = load.I32(flags, ptr, offset)),
            def!(x = sextend(xl)),
        ],
    );

    for &ty in &[I8, I16] {
        narrow.legalize(
            def!(a = uextend.I64.ty(x)),
            vec![
                def!(al = uextend.I32.ty(x)),
                def!(ah = iconst.I32(Literal::constant(&imm.imm64, 0))),
                def!(a = iconcat(al, ah)),
            ],
        );

        narrow.legalize(
            def!(a = sextend.I64.ty(x)),
            vec![
                def!(al = sextend.I32.ty(x)),
                def!(ah = sshr_imm(al, Literal::constant(&imm.imm64, 31))),
                def!(a = iconcat(al, ah)),
            ],
        );
    }

    // Custom legalization for rotations.
    narrow.custom_legalize(rotr, "narrow_rotr");
    narrow.custom_legalize(rotl, "narrow_rotl");

    expand.custom_legalize(spill, "expand_spill");

    expand.build_and_add_to(&mut shared.transform_groups);
    narrow.build_and_add_to(&mut shared.transform_groups);
}
