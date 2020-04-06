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
    let band = insts.by_name("band");
    let bint = insts.by_name("bint");
    let bitcast = insts.by_name("bitcast");
    let bnot = insts.by_name("bnot");
    let bor = insts.by_name("bor");
    let bxor = insts.by_name("bxor");
    let f32const = insts.by_name("f32const");
    let f64const = insts.by_name("f64const");
    let fabs = insts.by_name("fabs");
    let fcmp = insts.by_name("fcmp");
    let fcvt_from_sint = insts.by_name("fcvt_from_sint");
    let fcvt_from_uint = insts.by_name("fcvt_from_uint");
    let fcvt_to_sint = insts.by_name("fcvt_to_sint");
    let fcvt_to_uint = insts.by_name("fcvt_to_uint");
    let fdiv = insts.by_name("fdiv");
    let fma = insts.by_name("fma");
    let fmul = insts.by_name("fmul");
    let fsub = insts.by_name("fsub");
    let iadd_imm = insts.by_name("iadd_imm");
    let icmp_imm = insts.by_name("icmp_imm");
    let iconcat = insts.by_name("iconcat");
    let imul = insts.by_name("imul");
    let imul_imm = insts.by_name("imul_imm");
    let isplit = insts.by_name("isplit");
    let load = insts.by_name("load");
    let sextend = insts.by_name("sextend");
    let stack_load = insts.by_name("stack_load");
    let stack_store = insts.by_name("stack_store");
    let store = insts.by_name("store");
    let trunc = insts.by_name("trunc");

    let arm32_vcvt_f2uint = arm32_insts.by_name("arm32_vcvt_f2uint");
    let arm32_vcvt_f2sint = arm32_insts.by_name("arm32_vcvt_f2sint");
    let arm32_vcvt_uint2f = arm32_insts.by_name("arm32_vcvt_uint2f");
    let arm32_vcvt_sint2f = arm32_insts.by_name("arm32_vcvt_sint2f");
    let arm32_vmov_d2ints = arm32_insts.by_name("arm32_vmov_d2ints");
    let arm32_vmov_ints2d = arm32_insts.by_name("arm32_vmov_ints2d");

    let _imm = &shared.imm;

    let a = var("a");
    let a1 = var("a1");
    let a2 = var("a2");
    let a3 = var("a3");
    let a4 = var("a4");
    let b = var("b");
    let b1 = var("b1");
    let b2 = var("b2");
    let c = var("c");
    let m = var("m");
    let x = var("x");
    let xl = var("xl");
    let xh = var("xh");
    let y = var("y");
    let yl = var("yl");
    let yh = var("yh");
    let z = var("z");

    let f32_two_pow_32 = 0x4f80_0000;
    let f64_two_pow_32 = 0x41F0_0000_0000_0000;
    let f32_zero = 0x0000_0000;
    let f64_zero = 0x0000_0000_0000_0000;

    for &(fty, ity) in &[(F32, I32), (F64, I64)] {
        expand.legalize(
            def!(x = bnot.fty(y)),
            vec![
                def!(a = bitcast.ity(y)),
                def!(b = bnot(a)),
                def!(x = bitcast.fty(b)),
            ],
        );

        for &op in &[band, bor, bxor] {
            expand.legalize(
                def!(x = op.fty(y, z)),
                vec![
                    def!(a = bitcast.ity(y)),
                    def!(b = bitcast.ity(z)),
                    def!(c = op(a, b)),
                    def!(x = bitcast.fty(c)),
                ],
            );
        }
    }

    for &(fty, ity) in &[(F32, I32), (F64, I64)] {
        expand.legalize(
            def!(x = load.fty(m, y, z)),
            vec![def!(a = load.ity(m, y, z)), def!(x = bitcast.fty(a))],
        );

        expand.legalize(
            def!(x = stack_load.fty(m, y)),
            vec![def!(a = stack_load.ity(m, y)), def!(x = bitcast.fty(a))],
        );

        expand.legalize(
            def!(store.fty(m, x, y, z)),
            vec![def!(a = bitcast.ity(x)), def!(store.ity(m, a, y, z))],
        );

        expand.legalize(
            def!(stack_store.fty(x, m, y)),
            vec![def!(a = bitcast.ity(x)), def!(stack_store.ity(a, m, y))],
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

    for &(ty, const_op, literal) in &[
        (F32, f32const, &Literal::bits(&_imm.ieee32, f32_two_pow_32)),
        (F64, f64const, &Literal::bits(&_imm.ieee64, f64_two_pow_32)),
    ] {
        expand.legalize(
            def!(x = fcvt_from_uint.ty.I64(y)),
            vec![
                def!((yl, yh) = isplit(y)),
                def!(xl = fcvt_from_uint.ty(yl)),
                def!(xh = fcvt_from_uint.ty(yh)),
                def!(z = const_op(literal)),
                def!(x = fma(xh, z, xl)),
            ],
        );

        narrow.legalize(
            def!(x = fcvt_to_uint.I64.ty(y)),
            vec![
                def!(z = const_op(literal)),
                def!(a = fdiv(y, z)),
                def!(yh = trunc(a)),
                def!(b = fmul(yh, z)),
                def!(yl = fsub(y, b)),
                def!(xl = fcvt_to_uint.I32(yl)),
                def!(xh = fcvt_to_uint.I32(yh)),
                def!(x = iconcat(xl, xh)),
            ],
        );
    }

    for &(ty, const_op, literal) in &[
        (F32, f32const, &Literal::bits(&_imm.ieee32, f32_zero)),
        (F64, f64const, &Literal::bits(&_imm.ieee64, f64_zero)),
    ] {
        expand.legalize(
            def!(x = fcvt_from_sint.ty.I64(y)),
            vec![
                def!(
                    a = icmp_imm(
                        &Literal::enumerator_for(&_imm.intcc, "sge"),
                        y,
                        &Literal::constant(&_imm.imm64, 0)
                    )
                ),
                def!(a1 = bint.I32(a)),
                def!(a2 = imul_imm(a1, &Literal::constant(&_imm.imm64, 2))),
                def!(a3 = iadd_imm(a2, &Literal::constant(&_imm.imm64, -1))),
                def!(a4 = sextend.I64(a3)),
                def!(b = imul(y, a4)),
                def!(b1 = fcvt_from_uint.ty(b)),
                def!(b2 = fcvt_from_sint.ty(a3)),
                def!(x = fmul(b1, b2)),
            ],
        );

        narrow.legalize(
            def!(x = fcvt_to_sint.I64.ty(y)),
            vec![
                def!(a = const_op(literal)),
                def!(a1 = fcmp(Literal::enumerator_for(&_imm.floatcc, "ge"), y, a)),
                def!(a2 = bint.I32(a1)),
                def!(a3 = imul_imm(a2, Literal::constant(&_imm.imm64, 2))),
                def!(a4 = iadd_imm(a3, Literal::constant(&_imm.imm64, -1))),
                def!(b = sextend.I64(a4)),
                def!(b1 = fabs(y)),
                def!(b2 = fcvt_to_uint.I64(b1)),
                def!(x = imul(b2, b)),
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

    narrow.legalize(
        def!(x = bitcast.I64.F64(y)),
        vec![
            def!((xl, xh) = arm32_vmov_d2ints(y)),
            def!(x = iconcat(xl, xh)),
        ],
    );

    narrow.build_and_add_to(&mut shared.transform_groups);
}
