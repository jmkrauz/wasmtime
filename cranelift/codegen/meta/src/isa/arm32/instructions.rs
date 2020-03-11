#![allow(non_snake_case)]

use crate::cdsl::instructions::{
    AllInstructions, InstructionBuilder as Inst, InstructionGroup, InstructionGroupBuilder,
};
use crate::cdsl::operands::Operand;
use crate::cdsl::types::{LaneType, ValueType};
use crate::cdsl::typevar::{Interval, TypeSetBuilder, TypeVar};
use crate::shared::formats::Formats;
use crate::shared::immediates::Immediates;
use crate::shared::types;

pub(crate) fn define(
    mut all_instructions: &mut AllInstructions,
    formats: &Formats,
    _immediates: &Immediates,
) -> InstructionGroup {
    let mut ig = InstructionGroupBuilder::new(&mut all_instructions);

    let _iflags: &TypeVar = &ValueType::Special(types::Flag::IFlags.into()).into();
    let f32_: &TypeVar = &ValueType::from(LaneType::from(types::Float::F32)).into();
    let f64_: &TypeVar = &ValueType::from(LaneType::from(types::Float::F64)).into();

    let iWord = &TypeVar::new(
        "iWord",
        "A scalar integer machine word",
        TypeSetBuilder::new().ints(32..32).build(),
    );
    let x = &Operand::new("x", iWord);
    let y = &Operand::new("y", iWord);

    let Float = &TypeVar::new(
        "Float",
        "A scalar floating point number",
        TypeSetBuilder::new()
            .floats(Interval::All)
            .simd_lanes(Interval::All)
            .build(),
    );
    let vd = &Operand::new("vd", Float);
    let vm = &Operand::new("vm", Float);

    let s_vd = &Operand::new("s_vd", f32_);
    let s_vm = &Operand::new("s_vm", f32_);

    let d_vm = &Operand::new("d_vm", f64_);

    ig.push(
        Inst::new(
            "arm32_vcvt_f2uint",
            r#"
        Floating-point to unsigned integer conversion.

        Converts `vm` to an unsigned integer rounding towards zero.
        If `vm` is NaN or if the unsigned integral value cannot be
        represented in arm core register, this instruction traps.

        Result is stored in S floating-point register.
        "#,
            &formats.unary,
        )
        .operands_in(vec![vm])
        .operands_out(vec![s_vd])
        .can_trap(true),
    );

    ig.push(
        Inst::new(
            "arm32_vcvt_f2sint",
            r#"
        Floating-point to signed integer conversion.

        Converts `vm` to an signed integer rounding towards zero.
        If `vm` is NaN or if the unsigned integral value cannot be
        represented in arm core register, this instruction traps.

        Result is stored in S floating-point register.
        "#,
            &formats.unary,
        )
        .operands_in(vec![vm])
        .operands_out(vec![s_vd])
        .can_trap(true),
    );

    ig.push(
        Inst::new(
            "arm32_vcvt_sint2f",
            r#"
        Signed integer to floating-point conversion.

        Converts `vm` to a floating-point value rounding
        to the nearest integer, ties to even.
        "#,
            &formats.unary,
        )
        .operands_in(vec![s_vm])
        .operands_out(vec![vd]),
    );

    ig.push(
        Inst::new(
            "arm32_vcvt_uint2f",
            r#"
        Unsigned integer to floating-point conversion.

        Converts `vm` to a floating-point value rounding
        to the nearest integer, ties to even.
        "#,
            &formats.unary,
        )
        .operands_in(vec![s_vm])
        .operands_out(vec![vd]),
    );

    ig.push(
        Inst::new(
            "arm32_vmov_d2ints",
            r#"
        Move contents of D floating-point register to two ARM core registers.
        "#,
            &formats.unary,
        )
        .operands_in(vec![d_vm])
        .operands_out(vec![x, y]),
    );

    ig.push(
        Inst::new(
            "arm32_vmov_ints2d",
            r#"
        Move contents of two ARM core registers to D floating-point register.
        "#,
            &formats.binary,
        )
        .operands_in(vec![x, y])
        .operands_out(vec![d_vm]),
    );

    ig.build()
}
