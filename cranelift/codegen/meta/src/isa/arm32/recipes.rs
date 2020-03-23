use std::collections::HashMap;

use crate::cdsl::ast::Literal;
use crate::cdsl::formats::InstructionFormat;
use crate::cdsl::instructions::InstructionPredicate;
use crate::cdsl::recipes::{
    EncodingRecipeBuilder, EncodingRecipeNumber, OperandConstraint, Recipes, Register, Stack,
};
use crate::cdsl::regs::IsaRegs;
use crate::shared::Definitions as SharedDefinitions;

/// An helper to create recipes and use them when defining the arm32 encodings.
pub(crate) struct RecipeGroup {
    /// The actualy list of recipes explicitly created in this file.
    pub recipes: Recipes,

    /// Provides fast lookup from a name to an encoding recipe.
    name_to_recipe: HashMap<String, EncodingRecipeNumber>,
}

impl RecipeGroup {
    fn new() -> Self {
        Self {
            recipes: Recipes::new(),
            name_to_recipe: HashMap::new(),
        }
    }

    fn push(&mut self, builder: EncodingRecipeBuilder) {
        assert!(
            self.name_to_recipe.get(&builder.name).is_none(),
            format!("arm32 recipe '{}' created twice", builder.name)
        );
        let name = builder.name.clone();
        let number = self.recipes.push(builder.build());
        self.name_to_recipe.insert(name, number);
    }

    pub fn by_name(&self, name: &str) -> EncodingRecipeNumber {
        let number = *self
            .name_to_recipe
            .get(name)
            .expect(&format!("unknown arm32 recipe name {}", name));
        number
    }

    pub fn collect(self) -> Recipes {
        self.recipes
    }
}

/// Returns a predicate checking that the "cond" field of the instruction contains one of the
/// directly supported floating point condition codes.
fn supported_floatccs_predicate(
    supported_cc: &[Literal],
    format: &InstructionFormat,
) -> InstructionPredicate {
    supported_cc
        .iter()
        .fold(InstructionPredicate::new(), |pred, literal| {
            pred.or(InstructionPredicate::new_is_field_equal(
                format,
                "cond",
                literal.to_rust_code(),
            ))
        })
}

pub(crate) fn define(shared_defs: &SharedDefinitions, regs: &IsaRegs) -> RecipeGroup {
    let formats = &shared_defs.formats;

    let floatcc = &shared_defs.imm.floatcc;
    let supported_floatccs: Vec<Literal> = [
        "ord", "eq", "lt", "le", "gt", "ge", "uno", "ne", "ult", "ule", "ugt", "uge",
    ]
    .iter()
    .map(|name| Literal::enumerator_for(floatcc, name))
    .collect();

    // Register classes shorthands.
    let gpr = regs.class_by_name("GPR");
    let s_reg = regs.class_by_name("S");
    let d_reg = regs.class_by_name("D");
    let flag = regs.class_by_name("FLAG");

    let reg_nzcv = Register::new(flag, regs.regunit_by_name(flag, "nzcv"));
    let stack_gpr = Stack::new(gpr);
    let stack_s_reg = Stack::new(s_reg);
    let stack_d_reg = Stack::new(d_reg);

    // Definitions.
    let mut recipes = RecipeGroup::new();

    // Constants
    recipes.push(
        EncodingRecipeBuilder::new("iconst_u16", &formats.unary_imm, 4)
            .operands_out(vec![gpr])
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.unary_imm,
                "imm",
                16,
                0,
            ))
            .clobbers_flags(false)
            .emit(
                r#"
                    let imm: i64 = imm.into();
                    put_mov16_i(AL, imm as u16, out_reg0, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("iconst", &formats.unary_imm, 8)
            .operands_out(vec![gpr])
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.unary_imm,
                        "imm",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.unary_imm,
                        "imm",
                        32,
                        0,
                    )),
            )
            .clobbers_flags(false)
            .emit("put_mov32_i(imm.into(), out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("bconst", &formats.unary_bool, 4)
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit(
                r#"
                    let imm: u8 = if imm { 1 } else { 0 };
                    put_dp_i(MOV | AL, imm, out_reg0, sink);
                "#,
            ),
    );

    // Data processing, two registers.
    recipes.push(
        EncodingRecipeBuilder::new("dp_rr", &formats.binary, 4)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![gpr])
            .emit("put_dp_rr(bits, in_reg0, in_reg1, out_reg0, sink);"),
    );

    // Data processing, register and 8-bit immediate.
    recipes.push(
        EncodingRecipeBuilder::new("dp_ri", &formats.binary_imm, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.binary_imm,
                "imm",
                8,
                0,
            ))
            .emit(
                r#"
                    let imm: i64 = imm.into();
                    put_dp_ri(bits, in_reg0, (imm & 0xff) as u8, out_reg0, sink);
                "#,
            ),
    );

    // Data processing, one register.
    recipes.push(
        EncodingRecipeBuilder::new("dp_r", &formats.unary, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .emit("put_dp_r(bits, in_reg0, out_reg0, sink);"),
    );

    // Data processing, one 8-bit immediate
    recipes.push(
        EncodingRecipeBuilder::new("dp_i_u8", &formats.unary_imm, 4)
            .operands_out(vec![gpr])
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.unary_imm,
                "imm",
                8,
                0,
            ))
            .emit(
                r#"
                    let imm: i64 = imm.into();
                    put_dp_i(bits, (imm & 0xff) as u8, out_reg0, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("dp_out_iflags", &formats.binary, 4)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![
                OperandConstraint::RegClass(gpr),
                OperandConstraint::FixedReg(reg_nzcv),
            ])
            .clobbers_flags(true)
            .emit("put_dp_rr(bits, in_reg0, in_reg1, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("dp_in_iflags", &formats.ternary, 4)
            .operands_in(vec![
                OperandConstraint::RegClass(gpr),
                OperandConstraint::RegClass(gpr),
                OperandConstraint::FixedReg(reg_nzcv),
            ])
            .operands_out(vec![gpr])
            .emit("put_dp_rr(bits, in_reg0, in_reg1, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("dp_io_iflags", &formats.ternary, 4)
            .operands_in(vec![
                OperandConstraint::RegClass(gpr),
                OperandConstraint::RegClass(gpr),
                OperandConstraint::FixedReg(reg_nzcv),
            ])
            .operands_out(vec![
                OperandConstraint::RegClass(gpr),
                OperandConstraint::FixedReg(reg_nzcv),
            ])
            .clobbers_flags(true)
            .emit("put_dp_rr(bits, in_reg0, in_reg1, out_reg0, sink);"),
    );

    // For regmove Cranelift IR instruction.
    recipes.push(
        EncodingRecipeBuilder::new("regmov", &formats.reg_move, 4)
            .operands_in(vec![gpr])
            .clobbers_flags(false)
            .emit("put_dp_r(MOV | AL, src, dst, sink);"),
    );

    // Sign or zero-extension.
    recipes.push(
        EncodingRecipeBuilder::new("extend", &formats.unary, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit("put_extend(bits, in_reg0, out_reg0, sink);"),
    );

    // Null conversion.
    recipes.push(
        EncodingRecipeBuilder::new("null", &formats.unary, 0)
            .operands_in(vec![gpr])
            .operands_out(vec![0])
            .clobbers_flags(false)
            .emit(""),
    );

    recipes.push(
        EncodingRecipeBuilder::new("stacknull", &formats.unary, 0)
            .operands_in(vec![stack_gpr])
            .operands_out(vec![stack_gpr])
            .clobbers_flags(false)
            .emit(""),
    );

    recipes.push(
        EncodingRecipeBuilder::new("fillnull", &formats.unary, 0)
            .operands_in(vec![stack_gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit(""),
    );

    recipes.push(
        EncodingRecipeBuilder::new("copy_special", &formats.copy_special, 4)
            .clobbers_flags(false)
            .emit("put_dp_r(MOV | AL, src, dst, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("copy_to_ssa", &formats.copy_to_ssa, 4)
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit("put_dp_r(MOV | AL, src, out_reg0, sink);"),
    );

    // Rotations/shifts, amount specified by register.
    recipes.push(
        EncodingRecipeBuilder::new("rotate_r", &formats.binary, 8)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![gpr])
            .emit(
                r#"
                    put_dp_ri(AND | AL, in_reg1, 0x1f, TEMP_REG, sink);               // 5 bit mask
                    put_mov_rotate_r(bits, in_reg0, TEMP_REG, out_reg0, sink);
                "#,
            ),
    );

    // Rotations/shifts, amount specified by immediate.
    recipes.push(
        EncodingRecipeBuilder::new("rotate_imm", &formats.binary_imm, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.binary_imm,
                        "imm",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.binary_imm,
                        "imm",
                        32,
                        0,
                    )),
            )
            .emit(
                r#"
                    let imm: i64 = imm.into();
                    put_mov_rotate_i(bits, in_reg0, (imm & 0x1f) as u8, out_reg0, sink);
                "#,
            ),
    );

    // Reverse bits.
    recipes.push(
        EncodingRecipeBuilder::new("revbit", &formats.unary, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit("put_bitrev(AL, in_reg0, out_reg0, sink);"),
    );

    // Count leading zeros.
    recipes.push(
        EncodingRecipeBuilder::new("clz", &formats.unary, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit("put_clz(AL, in_reg0, out_reg0, sink);"),
    );

    // Count trailing zeros.
    recipes.push(
        EncodingRecipeBuilder::new("ctz", &formats.unary, 8)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit(
                r#"
                    put_bitrev(AL, in_reg0, TEMP_REG, sink);
                    put_clz(AL, TEMP_REG, out_reg0, sink);
                "#,
            ),
    );

    // Multiply, register (imul).
    recipes.push(
        EncodingRecipeBuilder::new("mul_rr", &formats.binary, 4)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![gpr])
            .emit("put_mul(bits, in_reg0, in_reg1, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("mulhi", &formats.binary, 4)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![gpr])
            .emit("put_mull(bits, in_reg0, in_reg1, TEMP_REG, out_reg0, sink);"),
    );

    // Division
    recipes.push(
        EncodingRecipeBuilder::new("div", &formats.binary, 4)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit("put_div(bits, in_reg0, in_reg1, out_reg0, sink);"),
    );

    // Comparison, set flags, two registers.
    recipes.push(
        EncodingRecipeBuilder::new("ifcmp", &formats.binary, 4)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![reg_nzcv])
            .clobbers_flags(true)
            .emit("put_dp_rr(CMP | AL | 0x100, in_reg0, in_reg1, NULL_REG, sink);"),
    );

    // Comparison, set flags, register and 8-bit immediate.
    recipes.push(
        EncodingRecipeBuilder::new("ifcmp_imm", &formats.binary_imm, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![reg_nzcv])
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.binary_imm,
                "imm",
                8,
                0,
            ))
            .clobbers_flags(true)
            .emit(
                r#"
                    let imm: i64 = imm.into();
                    put_dp_ri(CMP | AL | 0x1000, in_reg0, (imm & 0xff) as u8, NULL_REG, sink);
                "#,
            ),
    );

    // Comparison, set flags, register and SP.
    recipes.push(
        EncodingRecipeBuilder::new("ifcmp_sp", &formats.unary, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![reg_nzcv])
            .clobbers_flags(true)
            .emit("put_dp_rr(CMP | AL | 0x100, in_reg0, SP_REG, NULL_REG, sink);"),
    );

    // Comparison, return true/false, two registers.
    recipes.push(
        EncodingRecipeBuilder::new("icmp", &formats.int_compare, 12)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(true)
            .emit(
                r#"
                    put_dp_rr(CMP | AL | 0x100, in_reg0, in_reg1, NULL_REG, sink);  // CMP in_reg0, in_reg1
                    let cond_bits = icc2bits(cond);
                    put_dp_i(MOV | AL, 0x0, out_reg0, sink);                        // MOV out_reg0, #0
                    put_dp_i(MOV | (cond_bits & 0xf), 0x1, out_reg0, sink);         // MOV<cond> out_reg0, #1
                "#,
            ),
    );

    // Comparison, return true/false, register and 8-bit immediate.
    recipes.push(
        EncodingRecipeBuilder::new("icmp_imm", &formats.int_compare_imm, 12)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(true)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(&formats.int_compare_imm, "imm", 8, 0))
            .emit(
                r#"
                    let imm: i64 = imm.into();
                    put_dp_ri(CMP | AL | 0x1000, in_reg0, (imm & 0xff) as u8, NULL_REG, sink);  // CMP in_reg0, in_reg1
                    let cond_bits = icc2bits(cond);
                    put_dp_i(MOV | AL, 0x0, out_reg0, sink);                                    // MOV out_reg0, #0
                    put_dp_i(MOV | (cond_bits & 0xf) | 0xd0, 0x1, out_reg0, sink);              // MOV<cond> out_reg0, #1
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("get_iflags", &formats.int_cond, 8)
            .operands_in(vec![reg_nzcv])
            .operands_out(vec![gpr])
            .emit(
                r#"
                    let cond_bits = icc2bits(cond);
                    put_dp_i(MOV | AL, 0x0, out_reg0, sink);                        // MOV out_reg0, #0
                    put_dp_i(MOV | (cond_bits & 0xf), 0x1, out_reg0, sink);         // MOV<cond> out_reg0, #1
                "#
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("select_iflags", &formats.int_select, 8)
            .operands_in(vec![
                OperandConstraint::FixedReg(reg_nzcv),
                OperandConstraint::RegClass(gpr),
                OperandConstraint::RegClass(gpr),
            ])
            .operands_out(vec![gpr])
            .emit(
                r#"
                    let cond_bits_1 = icc2bits(cond);
                    let cond_bits_2 = icc2bits(cond.inverse());
                    put_dp_r(MOV | cond_bits_1, in_reg1, out_reg0, sink);
                    put_dp_r(MOV | cond_bits_2, in_reg2, out_reg0, sink);
                "#,
            ),
    );

    // Branch (B, BL).
    recipes.push(
        EncodingRecipeBuilder::new("branch", &formats.jump, 4)
            .branch_range((8, 26))
            .clobbers_flags(false)
            .emit(
                r#"
                    let offset = branch_offset(destination, func, sink);
                    put_b(bits, offset, sink);
                "#,
            ),
    );

    // Test integer flags and branch if condition is true.
    recipes.push(
        EncodingRecipeBuilder::new("branch_iflags", &formats.branch_int, 4)
            .branch_range((8, 26))
            .clobbers_flags(false)
            .emit(
                r#"
                    let cond_bits = icc2bits(cond);
                    let offset = branch_offset(destination, func, sink);
                    put_b(cond_bits, offset, sink);
                "#,
            ),
    );

    // Branch under certain condition (brz and brnz Cranelift IR instructions).
    // Test then branch.
    recipes.push(
        EncodingRecipeBuilder::new("test_branch", &formats.branch, 8)
            .operands_in(vec![gpr])
            .branch_range((12, 26))
            .clobbers_flags(true)
            .emit(
                r#"
                    put_dp_ri(CMP | AL | 0x1000, in_reg0, 0x0, NULL_REG, sink);
                    let offset = branch_offset(destination, func, sink);
                    put_b(bits, offset, sink);
                "#,
            ),
    );

    // Comparison and branch (br_icmp Cranelift IR instruction).
    recipes.push(
        EncodingRecipeBuilder::new("icmp_branch", &formats.branch_icmp, 8)
            .operands_in(vec![gpr, gpr])
            .branch_range((8, 26))
            .clobbers_flags(true)
            .emit(
                r#"
                    put_dp_rr(CMP | AL | 0x100, in_reg0, in_reg1, NULL_REG, sink);
                    let cond_bits = icc2bits(cond);
                    let offset = branch_offset(destination, func, sink);
                    put_b(cond_bits & 0xf, offset, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("indirect_jump", &formats.indirect_jump, 4)
            .operands_in(vec![gpr])
            .clobbers_flags(false)
            .emit("put_bx(AL, in_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("jt_entry", &formats.branch_table_entry, 4)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_field_equal(
                        &formats.branch_table_entry,
                        "imm",
                        "1".into(),
                    ))
                    .or(InstructionPredicate::new_is_field_equal(
                        &formats.branch_table_entry,
                        "imm",
                        "4".into(),
                    )),
            )
            .emit(
                r#"
                    let bits: u16 = match imm {
                        1 => AL | 0x30,                 // load byte
                        4 => AL | 0x10 | (2 << 11),     // load word, apply lsl #log2(imm) to the offset
                        _ => unreachable!()
                    };
                    put_mem_transfer_r(bits, in_reg1, in_reg0, out_reg0, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("jt_base", &formats.branch_table_base, 12)
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit(
                r#"
                    let addr = func.jt_offsets[table].wrapping_sub(sink.offset() + 16);
                    put_mov32_i(addr.into(), out_reg0, sink);
                    put_dp_rr(ADD | AL, out_reg0, PC_REG, out_reg0, sink);
                "#,
            ),
    );

    // Trap
    recipes.push(
        EncodingRecipeBuilder::new("trap", &formats.trap, 4)
            .clobbers_flags(false)
            .emit(
                r#"
                    sink.trap(code, func.srclocs[inst]);
                    put_udf(sink);
                "#,
            ),
    );

    // Test integer flags and trap if condition is true.
    recipes.push(
        EncodingRecipeBuilder::new("trap_iflags", &formats.int_cond_trap, 8)
            .operands_in(vec![reg_nzcv])
            .clobbers_flags(false)
            .emit(
                r#"
                    let cond_bits = icc2bits(cond.inverse());
                    put_b(cond_bits, 0, sink);
                    sink.trap(code, func.srclocs[inst]);
                    put_udf(sink);
                "#,
            ),
    );

    // Load from memory.
    recipes.push(
        EncodingRecipeBuilder::new("load_u8", &formats.load, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.load, "offset", 8, 0,
            ))
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    let offset: i32 = offset.into();
                    put_mem_transfer_i(bits | 0x100, in_reg0, (offset & 0xff) as u8, out_reg0, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("load", &formats.load, 12)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.load,
                        "offset",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.load,
                        "offset",
                        32,
                        0,
                    )),
            )
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    put_mov32_i(offset.into(), TEMP_REG, sink);
                    put_mem_transfer_r(bits | 0x10, in_reg0, TEMP_REG, out_reg0, sink);
                "#,
            ),
    );

    // Load 16 bits from memory to out_reg0
    recipes.push(
        EncodingRecipeBuilder::new("load16_u8", &formats.load, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.load, "offset", 8, 0,
            ))
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    let offset: i64 = offset.into();
                    put_mem_transfer_halfword_i(bits | 0x10, in_reg0, (offset & 0xff) as u8, out_reg0, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("load16", &formats.load, 12)
            .operands_in(vec![gpr])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.load,
                        "offset",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.load,
                        "offset",
                        32,
                        0,
                    )),
            )
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    put_mov32_i(offset.into(), TEMP_REG, sink);
                    put_mem_transfer_halfword_r(bits | 0x10, in_reg0, TEMP_REG, out_reg0, sink);
                "#,
            ),
    );

    // Store to memory, 8-bit offset.
    recipes.push(
        EncodingRecipeBuilder::new("store_u8", &formats.store, 4)
            .operands_in(vec![gpr, gpr])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.store,
                "offset",
                8,
                0,
            ))
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    let offset: i64 = offset.into();
                    put_mem_transfer_i(bits, in_reg1, (offset & 0xff) as u8, in_reg0, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("store", &formats.store, 12)
            .operands_in(vec![gpr, gpr])
            .clobbers_flags(false)
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.store,
                        "offset",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.store,
                        "offset",
                        32,
                        0,
                    )),
            )
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    put_mov32_i(offset.into(), TEMP_REG, sink);
                    put_mem_transfer_r(bits, in_reg1, TEMP_REG, in_reg0, sink);
                "#,
            ),
    );

    // Store to memory lower 16 bits of in_reg0
    recipes.push(
        EncodingRecipeBuilder::new("store16_u8", &formats.store, 4)
            .operands_in(vec![gpr, gpr])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.store, "offset", 8, 0,
            ))
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    let offset: i64 = offset.into();
                    put_mem_transfer_halfword_i(bits, in_reg1, (offset & 0xff) as u8, in_reg0, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("store16", &formats.store, 12)
            .operands_in(vec![gpr, gpr])
            .clobbers_flags(false)
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.store,
                        "offset",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.store,
                        "offset",
                        32,
                        0,
                    )),
            )
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    put_mov32_i(offset.into(), TEMP_REG, sink);
                    put_mem_transfer_halfword_r(bits, in_reg1, TEMP_REG, in_reg0, sink);
                "#,
            ),
    );

    // Get the address of a stack slot, 8-bit offset.
    recipes.push(
        EncodingRecipeBuilder::new("stack_addr_u8", &formats.stack_load, 4)
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.stack_load,
                "offset",
                8,
                0,
            ))
            .emit(
                r#"
                    let sp = StackRef::sp(stack_slot, &func.stack_slots);
                    let base = stk_base(sp.base);
                    let offset: i64 = offset.into();
                    put_dp_ri(ADD | AL, base, (offset & 0xff) as u8, out_reg0, sink);
                "#,
            ),
    );

    // Get the address of a stack slot.
    recipes.push(
        EncodingRecipeBuilder::new("stack_addr", &formats.stack_load, 12)
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.stack_load,
                        "offset",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.stack_load,
                        "offset",
                        32,
                        0,
                    )),
            )
            .emit(
                r#"
                    let sp = StackRef::sp(stack_slot, &func.stack_slots);
                    let base = stk_base(sp.base);
                    put_mov32_i(offset.into(), TEMP_REG, sink);
                    put_dp_rr(ADD | AL, base, TEMP_REG, out_reg0, sink);
                "#,
            ),
    );

    // Stack load, 8-bit offset.
    recipes.push(
        EncodingRecipeBuilder::new("stack_load_u8", &formats.stack_load, 4)
            .operands_out(vec![gpr])
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.stack_load,
                "offset",
                8,
                0,
            ))
            .clobbers_flags(false)
            .emit(
                r#"
                let sp = StackRef::sp(stack_slot, &func.stack_slots);
                let base = stk_base(sp.base);
                let offset: i64 = offset.into();
                put_mem_transfer_i(AL | 0x100, base, (offset & 0xff) as u8, out_reg0, sink);
            "#,
            ),
    );

    // Stack load.
    recipes.push(
        EncodingRecipeBuilder::new("stack_load", &formats.stack_load, 12)
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.stack_load,
                        "offset",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.stack_load,
                        "offset",
                        32,
                        0,
                    )),
            )
            .emit(
                r#"
                let sp = StackRef::sp(stack_slot, &func.stack_slots);
                let base = stk_base(sp.base);
                put_mov32_i(offset.into(), TEMP_REG, sink);
                put_mem_transfer_r(AL | 0x10, base, TEMP_REG, out_reg0, sink);
            "#,
            ),
    );

    // Stack store, 8-bit offset.
    recipes.push(
        EncodingRecipeBuilder::new("stack_store_u8", &formats.stack_store, 4)
            .operands_in(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.stack_store,
                "offset",
                8,
                0,
            ))
            .emit(
                r#"
                sink.trap(TrapCode::StackOverflow, func.srclocs[inst]);
                let sp = StackRef::sp(stack_slot, &func.stack_slots);
                let base = stk_base(sp.base);
                let offset: i64 = offset.into();
                put_mem_transfer_i(AL, base, (offset & 0xff) as u8, in_reg0, sink);
            "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("stack_store", &formats.stack_store, 12)
            .operands_in(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.stack_store,
                        "offset",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.stack_store,
                        "offset",
                        32,
                        0,
                    )),
            )
            .emit(
                r#"
                sink.trap(TrapCode::StackOverflow, func.srclocs[inst]);
                let sp = StackRef::sp(stack_slot, &func.stack_slots);
                let base = stk_base(sp.base);
                put_mov32_i(offset.into(), TEMP_REG, sink);
                put_mem_transfer_r(AL, base, TEMP_REG, in_reg0, sink);
            "#,
            ),
    );

    // Adjust SP, delta is register.
    recipes.push(
        EncodingRecipeBuilder::new("adjustsp", &formats.unary, 4)
            .operands_in(vec![gpr])
            .emit("put_dp_rr(SUB | AL, SP_REG, in_reg0, SP_REG, sink);"),
    );

    // Adjust SP, delta is 8-bit immediate.
    recipes.push(
        EncodingRecipeBuilder::new("adjustsp_imm_u8", &formats.unary_imm, 4)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.unary_imm,
                "imm",
                8,
                0,
            ))
            .emit(
                r#"
                    let imm: i64 = imm.into();
                    put_dp_ri(bits, SP_REG, (imm & 0xff) as u8, SP_REG, sink);      // ADD or SUB
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("adjustsp_imm", &formats.unary_imm, 12)
            .inst_predicate(
                InstructionPredicate::new()
                    .or(InstructionPredicate::new_is_signed_int(
                        &formats.unary_imm,
                        "imm",
                        32,
                        0,
                    ))
                    .or(InstructionPredicate::new_is_unsigned_int(
                        &formats.unary_imm,
                        "imm",
                        32,
                        0,
                    )),
            )
            .emit(
                r#"
                    put_mov32_i(imm.into(), TEMP_REG, sink);
                    put_dp_rr(bits, SP_REG, TEMP_REG, SP_REG, sink);      // ADD or SUB
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("spill", &formats.unary, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![stack_gpr])
            .compute_size("size_plus_maybe_mov_stack_offset_bytes")
            .clobbers_flags(false)
            .emit(
                r#"
                    sink.trap(TrapCode::StackOverflow, func.srclocs[inst]);
                    let base = stk_base(out_stk0.base);
                    if predicates::is_unsigned_int(out_stk0.offset, 8, 0) {
                        put_mem_transfer_i(AL, base, (out_stk0.offset & 0xff) as u8, in_reg0, sink);
                    } else {
                        put_mov32_i(out_stk0.offset as i64, TEMP_REG, sink);
                        put_mem_transfer_r(AL, base, TEMP_REG, in_reg0, sink);
                    }
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("fill", &formats.unary, 4)
            .operands_in(vec![stack_gpr])
            .operands_out(vec![gpr])
            .compute_size("size_plus_maybe_mov_stack_offset_bytes")
            .clobbers_flags(false)
            .emit(
                r#"
                    let base = stk_base(in_stk0.base);
                    if predicates::is_unsigned_int(in_stk0.offset, 8, 0) {
                        put_mem_transfer_i(AL | 0x100, base, (in_stk0.offset & 0xff) as u8, out_reg0, sink);
                    } else {
                        put_mov32_i(in_stk0.offset as i64, TEMP_REG, sink);
                        put_mem_transfer_r(AL | 0x10, base, TEMP_REG, out_reg0, sink);
                    }
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("regspill", &formats.reg_spill, 4)
            .operands_in(vec![gpr])
            .compute_size("size_plus_maybe_mov_stack_offset_bytes")
            .clobbers_flags(false)
            .emit(
                r#"
                    sink.trap(TrapCode::StackOverflow, func.srclocs[inst]);
                    let dst = StackRef::sp(dst, &func.stack_slots);
                    let base = stk_base(dst.base);
                    if predicates::is_unsigned_int(dst.offset, 8, 0) {
                        put_mem_transfer_i(AL, base, (dst.offset & 0xff) as u8, src, sink);
                    } else {
                        put_mov32_i(dst.offset as i64, TEMP_REG, sink);
                        put_mem_transfer_r(0xe, base, TEMP_REG, src, sink);
                    }
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("regfill", &formats.reg_fill, 4)
            .operands_in(vec![stack_gpr])
            .compute_size("size_plus_maybe_mov_stack_offset_bytes")
            .clobbers_flags(false)
            .emit(
                r#"
                    let src = StackRef::sp(src, &func.stack_slots);
                    let base = stk_base(src.base);
                    if predicates::is_unsigned_int(src.offset, 8, 0) {
                        put_mem_transfer_i(AL | 0x100, base, (src.offset & 0xff) as u8, dst, sink);
                    } else {
                        put_mov32_i(src.offset as i64, TEMP_REG, sink);
                        put_mem_transfer_r(0x1e, base, TEMP_REG, dst, sink);
                    }
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("func_addr", &formats.func_addr, 8)
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit(
                r#"
                    sink.reloc_external(Reloc::Arm32Call, &func.dfg.ext_funcs[func_ref].name, 0);
                    put_mov32_i(0, out_reg0, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("call_indirect", &formats.call_indirect, 4)
            .operands_in(vec![gpr])
            .clobbers_flags(false)
            .emit(
                r#"
                    sink.trap(TrapCode::StackOverflow, func.srclocs[inst]);
                    put_bx(AL | 0x10, in_reg0, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("return", &formats.multiary, 4)
            .clobbers_flags(false)
            .emit("put_bx(AL, LR_REG, sink);"),
    );

    // Floating point instructions

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_cmp", &formats.float_compare, 16)
            .operands_in(vec![s_reg, s_reg])
            .operands_out(vec![gpr])
            .clobbers_flags(true)
            .inst_predicate(supported_floatccs_predicate(
                    &supported_floatccs[..],
                    &*formats.float_compare,
                ))
            .emit(
                r#"
                    let cond_bits = fcc2bits(cond);
                    put_vfp_dp(bits, NULL_REG, in_reg1, in_reg0, sink);         // VCMP
                    put_vfp_transfer(0x0f, 0x2, 0xf, sink);                     // VMRS
                    put_dp_i(MOV | AL, 0x0, out_reg0, sink);                    // MOV out_reg0, #0
                    put_dp_i(MOV | (cond_bits & 0xf), 0x1, out_reg0, sink);     // MOV<cond> out_reg0, #1
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_cmp", &formats.float_compare, 16)
            .operands_in(vec![d_reg, d_reg])
            .operands_out(vec![gpr])
            .clobbers_flags(true)
            .inst_predicate(supported_floatccs_predicate(
                    &supported_floatccs[..],
                    &*formats.float_compare,
                ))
            .emit(
                r#"
                    let cond_bits = fcc2bits(cond);
                    put_vfp_dp(bits, NULL_REG, in_reg1, in_reg0, sink);         // VCMP
                    put_vfp_transfer(0x0f, 0x2, 0xf, sink);                     // VMRS
                    put_dp_i(MOV | AL, 0x0, out_reg0, sink);                    // MOV out_reg0, #0
                    put_dp_i(MOV | (cond_bits & 0xf), 0x1, out_reg0, sink);     // MOV<cond> out_reg0, #1
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_cmp_fflags", &formats.binary, 8)
            .operands_in(vec![s_reg, s_reg])
            .operands_out(vec![reg_nzcv])
            .clobbers_flags(true)
            .emit(
                r#"
                    put_vfp_dp(bits, NULL_REG, in_reg1, in_reg0, sink);     // VCMP
                    put_vfp_transfer(0x0f, 0x2, 0xf, sink);                 // VMRS
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_cmp_fflags", &formats.binary, 8)
            .operands_in(vec![d_reg, d_reg])
            .operands_out(vec![reg_nzcv])
            .clobbers_flags(true)
            .emit(
                r#"
                    put_vfp_dp(bits, NULL_REG, in_reg1, in_reg0, sink);     // VCMP
                    put_vfp_transfer(0x0f, 0x2, 0xf, sink);                 // VMRS
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("get_fflags", &formats.float_cond, 8)
            .operands_in(vec![reg_nzcv])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .inst_predicate(supported_floatccs_predicate(
                    &supported_floatccs[..],
                    &*formats.float_cond,
                ))
            .emit(
                r#"
                    let cond_bits = fcc2bits(cond);
                    put_dp_i(MOV | AL, 0x0, out_reg0, sink);                        // MOV out_reg0, #0
                    put_dp_i(MOV | (cond_bits & 0xf), 0x1, out_reg0, sink);         // MOV<cond> out_reg0, #1
                "#
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_dp_rr", &formats.binary, 4)
            .operands_in(vec![s_reg, s_reg])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, in_reg0, in_reg1, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_dp_rr", &formats.binary, 4)
            .operands_in(vec![d_reg, d_reg])
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, in_reg0, in_reg1, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_dp_r", &formats.unary, 4)
            .operands_in(vec![s_reg])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_dp_r", &formats.unary, 4)
            .operands_in(vec![d_reg])
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_fma", &formats.ternary, 4)
            .operands_in(vec![s_reg, s_reg, s_reg])
            .operands_out(vec![2])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, in_reg0, in_reg1, in_reg2, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_fma", &formats.ternary, 4)
            .operands_in(vec![d_reg, d_reg, d_reg])
            .operands_out(vec![2])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, in_reg0, in_reg1, in_reg2, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_vmov", &formats.unary, 4)
            .operands_in(vec![s_reg])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_vmov", &formats.unary, 4)
            .operands_in(vec![d_reg])
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s2int_mov", &formats.unary, 4)
            .operands_in(vec![s_reg])
            .operands_out(vec![gpr])
            .clobbers_flags(false)
            .emit("put_vfp_transfer(bits, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_int2s_mov", &formats.unary, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_transfer(bits, out_reg0, in_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s2int_convert", &formats.unary, 4)
            .operands_in(vec![s_reg])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d2int_convert", &formats.unary, 4)
            .operands_in(vec![d_reg])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_int2s_convert", &formats.unary, 4)
            .operands_in(vec![s_reg])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_int2d_convert", &formats.unary, 4)
            .operands_in(vec![s_reg])
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s2d_convert", &formats.unary, 4)
            .operands_in(vec![s_reg])
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d2s_convert", &formats.unary, 4)
            .operands_in(vec![d_reg])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, in_reg0, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_vmov_d2ints", &formats.unary, 4)
            .operands_in(vec![d_reg])
            .operands_out(vec![gpr, gpr])
            .clobbers_flags(false)
            .emit("put_vfp_64_transfer(bits, in_reg0, out_reg0, out_reg1, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_vmov_ints2d", &formats.binary, 4)
            .operands_in(vec![gpr, gpr])
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .emit("put_vfp_64_transfer(bits, out_reg0, in_reg0, in_reg1, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_load", &formats.load, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.load,
                "offset",
                10,
                2,
            ))
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    let offset: i32 = offset.into();
                    put_vfp_mem_transfer(bits, out_reg0, in_reg0, ((offset >> 2) & 0xff) as u8, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_load", &formats.load, 4)
            .operands_in(vec![gpr])
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.load,
                "offset",
                10,
                2,
            ))
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    let offset: i32 = offset.into();
                    put_vfp_mem_transfer(bits, out_reg0, in_reg0, ((offset >> 2) & 0xff) as u8, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_store", &formats.store, 4)
            .operands_in(vec![s_reg, gpr])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.store,
                "offset",
                10,
                2,
            ))
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    let offset: i32 = offset.into();
                    put_vfp_mem_transfer(bits, in_reg0, in_reg1, ((offset >> 2) & 0xff) as u8, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_store", &formats.store, 4)
            .operands_in(vec![d_reg, gpr])
            .clobbers_flags(false)
            .inst_predicate(InstructionPredicate::new_is_unsigned_int(
                &formats.store,
                "offset",
                10,
                2,
            ))
            .emit(
                r#"
                    if !flags.notrap() {
                        sink.trap(TrapCode::HeapOutOfBounds, func.srclocs[inst]);
                    }
                    let offset: i32 = offset.into();
                    put_vfp_mem_transfer(bits, in_reg0, in_reg1, ((offset >> 2) & 0xff) as u8, sink);
                "#,
            ),
    );

    // TODO: find better method for these than adding and subtracting offset to/from SP.

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_spill", &formats.unary, 20)
            .operands_in(vec![s_reg])
            .operands_out(vec![stack_s_reg])
            .clobbers_flags(false)
            .emit(
                r#"
                    sink.trap(TrapCode::StackOverflow, func.srclocs[inst]);
                    let base = stk_base(out_stk0.base);
                    put_mov32_i(out_stk0.offset as i64, TEMP_REG, sink);
                    put_dp_rr(AL | ADD, base, TEMP_REG, base, sink);
                    put_vfp_mem_transfer(bits, in_reg0, base, 0x0, sink);
                    put_dp_rr(AL | SUB, base, TEMP_REG, base, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_spill", &formats.unary, 20)
            .operands_in(vec![d_reg])
            .operands_out(vec![stack_d_reg])
            .clobbers_flags(false)
            .emit(
                r#"
                    sink.trap(TrapCode::StackOverflow, func.srclocs[inst]);
                    let base = stk_base(out_stk0.base);
                    put_mov32_i(out_stk0.offset as i64, TEMP_REG, sink);
                    put_dp_rr(AL | ADD, base, TEMP_REG, base, sink);
                    put_vfp_mem_transfer(bits, in_reg0, base, 0x0, sink);
                    put_dp_rr(AL | SUB, base, TEMP_REG, base, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_fill", &formats.unary, 20)
            .operands_in(vec![stack_s_reg])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit(
                r#"
                    let base = stk_base(in_stk0.base);
                    put_mov32_i(in_stk0.offset as i64, TEMP_REG, sink);
                    put_dp_rr(AL | ADD, base, TEMP_REG, base, sink);
                    put_vfp_mem_transfer(bits, out_reg0, base, 0x0, sink);
                    put_dp_rr(AL | SUB, base, TEMP_REG, base, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_fill", &formats.unary, 20)
            .operands_in(vec![stack_d_reg])
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .emit(
                r#"
                    let base = stk_base(in_stk0.base);
                    put_mov32_i(in_stk0.offset as i64, TEMP_REG, sink);
                    put_dp_rr(AL | ADD, base, TEMP_REG, base, sink);
                    put_vfp_mem_transfer(bits, out_reg0, base, 0x0, sink);
                    put_dp_rr(AL | SUB, base, TEMP_REG, base, sink);
                "#,
            ),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_s_regmove", &formats.reg_move, 4)
            .operands_in(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, src, dst, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("vfp_d_regmove", &formats.reg_move, 4)
            .operands_in(vec![d_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, src, dst, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("stacknull_s", &formats.unary, 0)
            .operands_in(vec![stack_s_reg])
            .operands_out(vec![stack_s_reg])
            .clobbers_flags(false)
            .emit(""),
    );

    recipes.push(
        EncodingRecipeBuilder::new("stacknull_d", &formats.unary, 0)
            .operands_in(vec![stack_d_reg])
            .operands_out(vec![stack_d_reg])
            .clobbers_flags(false)
            .emit(""),
    );

    recipes.push(
        EncodingRecipeBuilder::new("fillnull_s", &formats.unary, 0)
            .operands_in(vec![stack_s_reg])
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit(""),
    );

    recipes.push(
        EncodingRecipeBuilder::new("fillnull_d", &formats.unary, 0)
            .operands_in(vec![stack_d_reg])
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .emit(""),
    );

    recipes.push(
        EncodingRecipeBuilder::new("copy_to_ssa_s", &formats.copy_to_ssa, 4)
            .operands_out(vec![s_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, src, out_reg0, sink);"),
    );

    recipes.push(
        EncodingRecipeBuilder::new("copy_to_ssa_d", &formats.copy_to_ssa, 4)
            .operands_out(vec![d_reg])
            .clobbers_flags(false)
            .emit("put_vfp_dp(bits, NULL_REG, src, out_reg0, sink);"),
    );

    recipes
}
