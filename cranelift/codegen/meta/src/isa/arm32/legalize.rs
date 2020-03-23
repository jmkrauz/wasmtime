use crate::cdsl::ast::{var, ExprBuilder, Literal};
use crate::cdsl::instructions::{Bindable, InstructionGroup};
use crate::cdsl::xform::TransformGroupBuilder;
use crate::shared::types::Float::{F32, F64};
use crate::shared::types::Int::{I32, I64};
use crate::shared::Definitions as SharedDefinitions;

pub(crate) fn define(shared: &mut SharedDefinitions, arm32_insts: &InstructionGroup) {
    let mut expand = TransformGroupBuilder::new(
        "arm32_expand",
        r#"
    Legalize instructions by expansion.

    Use arm32-specific instructions if needed."#,
    )
    .isa("arm32")
    .chain_with(shared.transform_groups.by_name("expand_flags").id);

    // List of instructions.
    let insts = &shared.instructions;
    let bitcast = insts.by_name("bitcast");
    let f32const = insts.by_name("f32const");
    let f64const = insts.by_name("f64const");
    let fcvt_from_sint = insts.by_name("fcvt_from_sint");
    let fcvt_from_uint = insts.by_name("fcvt_from_uint");
    let fcvt_to_sint = insts.by_name("fcvt_to_sint");
    let fcvt_to_uint = insts.by_name("fcvt_to_uint");
    let fdiv = insts.by_name("fdiv");
    let fma = insts.by_name("fma");
    let fsub = insts.by_name("fsub");
    let iconcat = insts.by_name("iconcat");
    let isplit = insts.by_name("isplit");
    let load = insts.by_name("load");
    let store = insts.by_name("store");

    let arm32_vcvt_f2uint = arm32_insts.by_name("arm32_vcvt_f2uint");
    let arm32_vcvt_f2sint = arm32_insts.by_name("arm32_vcvt_f2sint");
    let arm32_vcvt_uint2f = arm32_insts.by_name("arm32_vcvt_uint2f");
    let arm32_vcvt_sint2f = arm32_insts.by_name("arm32_vcvt_sint2f");
    let arm32_vmov_d2ints = arm32_insts.by_name("arm32_vmov_d2ints");
    let arm32_vmov_ints2d = arm32_insts.by_name("arm32_vmov_ints2d");

    let _imm = &shared.imm;

    let a = var("a");
    let m = var("m");
    let x = var("x");
    let xl = var("xl");
    let xh = var("xh");
    let y = var("y");
    let yl = var("yl");
    let yh = var("yh");
    let z = var("z");

    for &(ity, fty) in &[(I32, F32), (I64, F64)] {
        expand.legalize(
            def!(x = load.fty(m, y, z)),
            vec![def!(a = load.ity(m, y, z)), def!(x = bitcast.fty(a))],
        );

        expand.legalize(
            def!(store.fty(m, x, y, z)),
            vec![def!(a = bitcast.ity(x)), def!(store.ity(m, a, y, z))],
        );
    }

    expand.legalize(
        def!(x = fcvt_to_uint.I32(y)),
        vec![def!(z = arm32_vcvt_f2uint(y)), def!(x = bitcast(z))],
    );

    expand.legalize(
        def!(x = fcvt_to_sint.I32(y)),
        vec![def!(z = arm32_vcvt_f2sint(y)), def!(x = bitcast(z))],
    );

    for &ty in &[F32, F64] {
        expand.legalize(
            def!(x = fcvt_from_uint.ty.I32(y)),
            vec![def!(z = bitcast(y)), def!(x = arm32_vcvt_uint2f(z))],
        );

        expand.legalize(
            def!(x = fcvt_from_sint.ty.I32(y)),
            vec![def!(z = bitcast(y)), def!(x = arm32_vcvt_sint2f(z))],
        );
    }

    for &op in &[fcvt_from_uint, fcvt_from_sint] {
        expand.legalize(
            def!(x = op.F32.I64(y)),
            vec![
                def!((yl, yh) = isplit(y)),
                def!(xl = fcvt_from_uint.F32(yl)),
                def!(xh = op.F32(yh)),
                def!(z = f32const(&Literal::bits(&_imm.ieee32, 0x4f80_0000))),
                def!(x = fma(xh, z, xl)),
            ],
        );

        expand.legalize(
            def!(x = op.F64.I64(y)),
            vec![
                def!((yl, yh) = isplit(y)),
                def!(xl = fcvt_from_uint.F64(yl)),
                def!(xh = op.F64(yh)),
                def!(z = f64const(&Literal::bits(&_imm.ieee64, 0x41F00_000_0000_0000))),
                def!(x = fma(xh, z, xl)),
            ],
        );
    }

    // In expand not narrow because F64 controlling type variable uses expand legalization.
    expand.legalize(
        def!(x = bitcast.F64.I64(y)),
        vec![
            def!((yl, yh) = isplit(y)),
            def!(x = arm32_vmov_ints2d(yl, yh)),
        ],
    );

    expand.build_and_add_to(&mut shared.transform_groups);

    let mut narrow = TransformGroupBuilder::new(
        "arm32_narrow",
        r#"
    Legalize instructions by narrowing.

    Use arm32-specific instructions if needed."#,
    )
    .isa("arm32")
    .chain_with(shared.transform_groups.by_name("narrow_flags").id);

    narrow.legalize(
        def!(x = bitcast.I64.F64(y)),
        vec![
            def!((xl, xh) = arm32_vmov_d2ints(y)),
            def!(x = iconcat(xl, xh)),
        ],
    );

    for &op in &[fcvt_to_uint, fcvt_to_sint] {
        narrow.legalize(
            def!(x = op.I64.F32(y)),
            vec![
                def!(z = f32const(&Literal::bits(&_imm.ieee32, 0x4f80_0000))),
                def!(yh = fdiv(y, z)),
                def!(yl = fsub(y, yh)),
                def!(xl = fcvt_to_uint.I32(yl)),
                def!(xh = op.I32(yh)),
                def!(x = iconcat(xl, xh)),
            ],
        );

        narrow.legalize(
            def!(x = op.I64.F64(y)),
            vec![
                def!(z = f64const(&Literal::bits(&_imm.ieee64, 0x41F00_000_0000_0000))),
                def!(yh = fdiv(y, z)),
                def!(yl = fsub(y, yh)),
                def!(xl = fcvt_to_uint.I32(yl)),
                def!(xh = op.I32(yh)),
                def!(x = iconcat(xl, xh)),
            ],
        );
    }

    narrow.build_and_add_to(&mut shared.transform_groups);
}
