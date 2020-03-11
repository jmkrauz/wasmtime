use crate::cdsl::encodings::{Encoding, EncodingBuilder};
use crate::cdsl::instructions::{
    Bindable, InstSpec, InstructionGroup, InstructionPredicateRegistry,
};
use crate::cdsl::recipes::{EncodingRecipeNumber, Recipes};
use crate::cdsl::settings::SettingGroup;
use crate::shared::types::Bool::{B1, B16, B32, B8};
use crate::shared::types::Float::{F32, F64};
use crate::shared::types::Int::{I16, I32, I8};
use crate::shared::Definitions as SharedDefinitions;

use super::recipes::RecipeGroup;
use crate::cdsl::instructions::BindParameter::Any;

pub(crate) struct PerCpuModeEncodings<'defs> {
    pub inst_pred_reg: InstructionPredicateRegistry,
    pub enc_a32: Vec<Encoding>,
    pub enc_t32: Vec<Encoding>,
    recipes: &'defs Recipes,
}

impl<'defs> PerCpuModeEncodings<'defs> {
    fn new(recipes: &'defs Recipes) -> Self {
        Self {
            inst_pred_reg: InstructionPredicateRegistry::new(),
            enc_a32: Vec::new(),
            enc_t32: Vec::new(),
            recipes,
        }
    }
    fn enc(
        &self,
        inst: impl Into<InstSpec>,
        recipe: EncodingRecipeNumber,
        bits: u16,
    ) -> EncodingBuilder {
        EncodingBuilder::new(inst.into(), recipe, bits)
    }
    fn add_a32(&mut self, encoding: EncodingBuilder) {
        self.enc_a32
            .push(encoding.build(self.recipes, &mut self.inst_pred_reg));
    }
    #[allow(dead_code)]
    fn add_t32(&mut self, encoding: EncodingBuilder) {
        self.enc_t32
            .push(encoding.build(self.recipes, &mut self.inst_pred_reg));
    }
}

pub(crate) fn define<'defs>(
    shared_defs: &'defs SharedDefinitions,
    isa_settings: &SettingGroup,
    recipes: &'defs RecipeGroup,
    arm32_insts: &InstructionGroup,
) -> PerCpuModeEncodings<'defs> {
    let shared = &shared_defs.instructions;

    // Shorthands for instructions.
    let adjust_sp_down = shared.by_name("adjust_sp_down");
    let adjust_sp_down_imm = shared.by_name("adjust_sp_down_imm");
    let adjust_sp_up_imm = shared.by_name("adjust_sp_up_imm");
    let band = shared.by_name("band");
    let band_imm = shared.by_name("band_imm");
    let bint = shared.by_name("bint");
    let bitcast = shared.by_name("bitcast");
    let bitrev = shared.by_name("bitrev");
    let bconst = shared.by_name("bconst");
    let bnot = shared.by_name("bnot");
    let bor = shared.by_name("bor");
    let bor_imm = shared.by_name("bor_imm");
    let brif = shared.by_name("brif");
    let br_icmp = shared.by_name("br_icmp");
    let brnz = shared.by_name("brnz");
    let brz = shared.by_name("brz");
    let bxor = shared.by_name("bxor");
    let bxor_imm = shared.by_name("bxor_imm");
    let call_indirect = shared.by_name("call_indirect");
    let clz = shared.by_name("clz");
    let ctz = shared.by_name("ctz");
    let copy = shared.by_name("copy");
    let copy_nop = shared.by_name("copy_nop");
    let copy_special = shared.by_name("copy_special");
    let copy_to_ssa = shared.by_name("copy_to_ssa");
    let fabs = shared.by_name("fabs");
    let fadd = shared.by_name("fadd");
    let fcmp = shared.by_name("fcmp");
    let fdemote = shared.by_name("fdemote");
    let fdiv = shared.by_name("fdiv");
    let ffcmp = shared.by_name("ffcmp");
    let fill = shared.by_name("fill");
    let fill_nop = shared.by_name("fill_nop");
    let fma = shared.by_name("fma");
    let fmul = shared.by_name("fmul");
    let fneg = shared.by_name("fneg");
    let fpromote = shared.by_name("fpromote");
    let fsub = shared.by_name("fsub");
    let func_addr = shared.by_name("func_addr");
    let iadd = shared.by_name("iadd");
    let iadd_ifcarry = shared.by_name("iadd_ifcarry");
    let iadd_ifcin = shared.by_name("iadd_ifcin");
    let iadd_ifcout = shared.by_name("iadd_ifcout");
    let iadd_imm = shared.by_name("iadd_imm");
    let icmp = shared.by_name("icmp");
    let icmp_imm = shared.by_name("icmp_imm");
    let iconst = shared.by_name("iconst");
    let ifcmp = shared.by_name("ifcmp");
    let ifcmp_imm = shared.by_name("ifcmp_imm");
    let ifcmp_sp = shared.by_name("ifcmp_sp");
    let imul = shared.by_name("imul");
    let indirect_jump_table_br = shared.by_name("indirect_jump_table_br");
    let ireduce = shared.by_name("ireduce");
    let ishl = shared.by_name("ishl");
    let ishl_imm = shared.by_name("ishl_imm");
    let istore8 = shared.by_name("istore8");
    let istore16 = shared.by_name("istore16");
    let isub = shared.by_name("isub");
    let isub_ifbin = shared.by_name("isub_ifbin");
    let isub_ifborrow = shared.by_name("isub_ifborrow");
    let isub_ifbout = shared.by_name("isub_ifbout");
    let jump = shared.by_name("jump");
    let jump_table_base = shared.by_name("jump_table_base");
    let jump_table_entry = shared.by_name("jump_table_entry");
    let load = shared.by_name("load");
    let regfill = shared.by_name("regfill");
    let regmove = shared.by_name("regmove");
    let regspill = shared.by_name("regspill");
    let return_ = shared.by_name("return");
    let rotr = shared.by_name("rotr");
    let rotr_imm = shared.by_name("rotr_imm");
    let sdiv = shared.by_name("sdiv");
    let selectif = shared.by_name("selectif");
    let sextend = shared.by_name("sextend");
    let smulhi = shared.by_name("smulhi");
    let spill = shared.by_name("spill");
    let sqrt = shared.by_name("sqrt");
    let sshr = shared.by_name("sshr");
    let sshr_imm = shared.by_name("sshr_imm");
    let store = shared.by_name("store");
    let stack_addr = shared.by_name("stack_addr");
    let stack_load = shared.by_name("stack_load");
    let stack_store = shared.by_name("stack_store");
    let trap = shared.by_name("trap");
    let trapif = shared.by_name("trapif");
    let trueif = shared.by_name("trueif");
    let trueff = shared.by_name("trueff");
    let udiv = shared.by_name("udiv");
    let uextend = shared.by_name("uextend");
    let uload8 = shared.by_name("uload8");
    let uload16 = shared.by_name("uload16");
    let umulhi = shared.by_name("umulhi");
    let ushr = shared.by_name("ushr");
    let ushr_imm = shared.by_name("ushr_imm");

    let arm32_vcvt_f2uint = arm32_insts.by_name("arm32_vcvt_f2uint");
    let arm32_vcvt_f2sint = arm32_insts.by_name("arm32_vcvt_f2sint");
    let arm32_vcvt_uint2f = arm32_insts.by_name("arm32_vcvt_uint2f");
    let arm32_vcvt_sint2f = arm32_insts.by_name("arm32_vcvt_sint2f");
    let arm32_vmov_d2ints = arm32_insts.by_name("arm32_vmov_d2ints");
    let arm32_vmov_ints2d = arm32_insts.by_name("arm32_vmov_ints2d");

    // Recipes shorthands.
    let r_adjustsp = recipes.by_name("adjustsp");
    let r_adjustsp_imm = recipes.by_name("adjustsp_imm");
    let r_adjustsp_imm_u8 = recipes.by_name("adjustsp_imm_u8");
    let r_bconst = recipes.by_name("bconst");
    let r_branch = recipes.by_name("branch");
    let r_branch_iflags = recipes.by_name("branch_iflags");
    let r_call_indirect = recipes.by_name("call_indirect");
    let r_clz = recipes.by_name("clz");
    let r_copy_special = recipes.by_name("copy_special");
    let r_copy_to_ssa = recipes.by_name("copy_to_ssa");
    let r_copy_to_ssa_d = recipes.by_name("copy_to_ssa_d");
    let r_copy_to_ssa_s = recipes.by_name("copy_to_ssa_s");
    let r_ctz = recipes.by_name("ctz");
    let r_div = recipes.by_name("div");
    let r_dp_in_iflags = recipes.by_name("dp_in_iflags");
    let r_dp_io_iflags = recipes.by_name("dp_io_iflags");
    let r_dp_out_iflags = recipes.by_name("dp_out_iflags");
    let r_dp_ri = recipes.by_name("dp_ri");
    let r_dp_r = recipes.by_name("dp_r");
    let r_dp_rr = recipes.by_name("dp_rr");
    let r_extend = recipes.by_name("extend");
    let r_fill = recipes.by_name("fill");
    let r_fillnull = recipes.by_name("fillnull");
    let r_fillnull_d = recipes.by_name("fillnull_d");
    let r_fillnull_s = recipes.by_name("fillnull_s");
    let r_func_addr = recipes.by_name("func_addr");
    let r_get_fflags = recipes.by_name("get_fflags");
    let r_get_iflags = recipes.by_name("get_iflags");
    let r_icmp = recipes.by_name("icmp");
    let r_icmp_branch = recipes.by_name("icmp_branch");
    let r_icmp_imm = recipes.by_name("icmp_imm");
    let r_iconst = recipes.by_name("iconst");
    let r_iconst_u16 = recipes.by_name("iconst_u16");
    let r_ifcmp = recipes.by_name("ifcmp");
    let r_ifcmp_imm = recipes.by_name("ifcmp_imm");
    let r_ifcmp_sp = recipes.by_name("ifcmp_sp");
    let r_indirect_jump = recipes.by_name("indirect_jump");
    let r_jt_base = recipes.by_name("jt_base");
    let r_jt_entry = recipes.by_name("jt_entry");
    let r_load = recipes.by_name("load");
    let r_load_u8 = recipes.by_name("load_u8");
    let r_load16 = recipes.by_name("load16");
    let r_load16_u8 = recipes.by_name("load16_u8");
    let r_mulhi = recipes.by_name("mulhi");
    let r_mul_rr = recipes.by_name("mul_rr");
    let r_null = recipes.by_name("null");
    let r_regfill = recipes.by_name("regfill");
    let r_regmov = recipes.by_name("regmov");
    let r_regspill = recipes.by_name("regspill");
    let r_revbit = recipes.by_name("revbit");
    let r_return = recipes.by_name("return");
    let r_rotate_imm = recipes.by_name("rotate_imm");
    let r_rotate_r = recipes.by_name("rotate_r");
    let r_select_iflags = recipes.by_name("select_iflags");
    let r_spill = recipes.by_name("spill");
    let r_stack_addr = recipes.by_name("stack_addr");
    let r_stack_addr_u8 = recipes.by_name("stack_addr_u8");
    let r_stack_load = recipes.by_name("stack_load");
    let r_stack_load_u8 = recipes.by_name("stack_load_u8");
    let r_stacknull = recipes.by_name("stacknull");
    let r_stacknull_s = recipes.by_name("stacknull_s");
    let r_stacknull_d = recipes.by_name("stacknull_d");
    let r_stack_store = recipes.by_name("stack_store");
    let r_stack_store_u8 = recipes.by_name("stack_store_u8");
    let r_store = recipes.by_name("store");
    let r_store16 = recipes.by_name("store16");
    let r_store16_u8 = recipes.by_name("store16_u8");
    let r_store_u8 = recipes.by_name("store_u8");
    let r_test_branch = recipes.by_name("test_branch");
    let r_trap = recipes.by_name("trap");
    let r_trap_iflags = recipes.by_name("trap_iflags");
    let r_vfp_d_cmp = recipes.by_name("vfp_d_cmp");
    let r_vfp_d_cmp_fflags = recipes.by_name("vfp_d_cmp_fflags");
    let r_vfp_d_dp_r = recipes.by_name("vfp_d_dp_r");
    let r_vfp_d_dp_rr = recipes.by_name("vfp_d_dp_rr");
    let r_vfp_d_fill = recipes.by_name("vfp_d_fill");
    let r_vfp_d_fma = recipes.by_name("vfp_d_fma");
    let r_vfp_d_load = recipes.by_name("vfp_d_load");
    let r_vfp_d_regmove = recipes.by_name("vfp_d_regmove");
    let r_vfp_d_spill = recipes.by_name("vfp_d_spill");
    let r_vfp_d_store = recipes.by_name("vfp_d_store");
    let r_vfp_d_vmov = recipes.by_name("vfp_d_vmov");
    let r_vfp_d2int_convert = recipes.by_name("vfp_d2int_convert");
    let r_vfp_d2s_convert = recipes.by_name("vfp_d2s_convert");
    let r_vfp_int2d_convert = recipes.by_name("vfp_int2d_convert");
    let r_vfp_int2s_mov = recipes.by_name("vfp_int2s_mov");
    let r_vfp_int2s_convert = recipes.by_name("vfp_int2s_convert");
    let r_vfp_s_cmp = recipes.by_name("vfp_s_cmp");
    let r_vfp_s_cmp_fflags = recipes.by_name("vfp_s_cmp_fflags");
    let r_vfp_s_dp_r = recipes.by_name("vfp_s_dp_r");
    let r_vfp_s_dp_rr = recipes.by_name("vfp_s_dp_rr");
    let r_vfp_s_fill = recipes.by_name("vfp_s_fill");
    let r_vfp_s_fma = recipes.by_name("vfp_s_fma");
    let r_vfp_s_load = recipes.by_name("vfp_s_load");
    let r_vfp_s_regmove = recipes.by_name("vfp_s_regmove");
    let r_vfp_s_spill = recipes.by_name("vfp_s_spill");
    let r_vfp_s_store = recipes.by_name("vfp_s_store");
    let r_vfp_s_vmov = recipes.by_name("vfp_s_vmov");
    let r_vfp_s2d_convert = recipes.by_name("vfp_s2d_convert");
    let r_vfp_s2int_convert = recipes.by_name("vfp_s2int_convert");
    let r_vfp_s2int_mov = recipes.by_name("vfp_s2int_mov");
    let r_vfp_vmov_d2ints = recipes.by_name("vfp_vmov_d2ints");
    let r_vfp_vmov_ints2d = recipes.by_name("vfp_vmov_ints2d");

    // Predicates shorthands.
    let use_div = isa_settings.predicate_by_name("use_div");

    // Condition codes
    let eq: u16 = 0x0; // Equal
    let ne: u16 = 0x1; // Not equal
    let al: u16 = 0xe; // Always

    // Data processing opcodes
    let and: u16 = 0x00;
    let eor: u16 = 0x10;
    let sub: u16 = 0x20;
    let add: u16 = 0x40;
    let adc: u16 = 0x50;
    let sbc: u16 = 0x60;
    let orr: u16 = 0xc0;
    let mov: u16 = 0xd0;
    let mvn: u16 = 0xf0;

    let null_bits: u16 = 0x0;

    // Definitions.
    let mut e = PerCpuModeEncodings::new(&recipes.recipes);

    // Constants
    e.add_a32(e.enc(iconst.bind(I32), r_iconst_u16, null_bits));
    e.add_a32(e.enc(iconst.bind(I32), r_iconst, null_bits));

    for &ty in &[B1, B8, B16, B32] {
        e.add_a32(e.enc(bconst.bind(ty), r_bconst, null_bits));
    }

    // Data processing instructions.
    e.add_a32(e.enc(iadd.bind(I32), r_dp_rr, al | add));
    e.add_a32(e.enc(iadd_ifcout.bind(I32), r_dp_out_iflags, al | add | 0x100));
    e.add_a32(e.enc(iadd_ifcin.bind(I32), r_dp_in_iflags, al | adc));
    e.add_a32(e.enc(iadd_ifcarry.bind(I32), r_dp_io_iflags, al | adc | 0x100));

    e.add_a32(e.enc(isub.bind(I32), r_dp_rr, al | sub));
    e.add_a32(e.enc(isub_ifbout.bind(I32), r_dp_out_iflags, al | sub | 0x100));
    e.add_a32(e.enc(isub_ifbin.bind(I32), r_dp_in_iflags, al | sbc));
    e.add_a32(e.enc(isub_ifborrow.bind(I32), r_dp_io_iflags, al | sbc | 0x100));

    e.add_a32(e.enc(iadd_imm.bind(I32), r_dp_ri, al | add));

    e.add_a32(e.enc(band.bind(I32), r_dp_rr, al | and));
    e.add_a32(e.enc(band.bind(B1), r_dp_rr, al | and));
    e.add_a32(e.enc(bor.bind(I32), r_dp_rr, al | orr));
    e.add_a32(e.enc(bor.bind(B1), r_dp_rr, al | orr));
    e.add_a32(e.enc(bxor.bind(I32), r_dp_rr, al | eor));
    e.add_a32(e.enc(bxor.bind(B1), r_dp_rr, al | eor));

    e.add_a32(e.enc(band_imm.bind(I32), r_dp_ri, al | and));
    e.add_a32(e.enc(bor_imm.bind(I32), r_dp_ri, al | orr));
    e.add_a32(e.enc(bxor_imm.bind(I32), r_dp_ri, al | eor));

    e.add_a32(e.enc(bnot.bind(I32), r_dp_r, al | mvn));
    e.add_a32(e.enc(bnot.bind(B1), r_dp_r, al | mvn));

    e.add_a32(e.enc(imul.bind(I32), r_mul_rr, al));
    e.add_a32(e.enc(umulhi.bind(I32), r_mulhi, al | 0x20));
    e.add_a32(e.enc(smulhi.bind(I32), r_mulhi, al));

    e.add_a32(
        e.enc(udiv.bind(I32), r_div, al | 0x10)
            .isa_predicate(use_div),
    );
    e.add_a32(e.enc(sdiv.bind(I32), r_div, al).isa_predicate(use_div));

    e.add_a32(e.enc(rotr.bind(I32).bind(I32), r_rotate_r, al | 0x30));
    e.add_a32(e.enc(rotr_imm.bind(I32), r_rotate_imm, al | 0x30));
    e.add_a32(e.enc(ushr.bind(I32).bind(I32), r_rotate_r, al | 0x10));
    e.add_a32(e.enc(ushr_imm.bind(I32), r_rotate_imm, al | 0x10));
    e.add_a32(e.enc(ishl.bind(I32).bind(I32), r_rotate_r, al));
    e.add_a32(e.enc(ishl_imm.bind(I32), r_rotate_imm, al));
    e.add_a32(e.enc(sshr.bind(I32).bind(I32), r_rotate_r, al | 0x20));
    e.add_a32(e.enc(sshr_imm.bind(I32), r_rotate_imm, al | 0x20));

    e.add_a32(e.enc(bitrev.bind(I32), r_revbit, null_bits));
    e.add_a32(e.enc(clz.bind(I32), r_clz, null_bits));
    e.add_a32(e.enc(ctz.bind(I32), r_ctz, null_bits));

    // Conversions
    e.add_a32(e.enc(ireduce.bind(I16).bind(I32), r_null, null_bits));
    e.add_a32(e.enc(ireduce.bind(I8).bind(I32), r_null, null_bits));
    e.add_a32(e.enc(ireduce.bind(I8).bind(I16), r_null, null_bits));

    e.add_a32(e.enc(bint.bind(I32).bind(B1), r_null, null_bits));

    e.add_a32(e.enc(uextend.bind(I32).bind(I16), r_extend, al | 0x30));
    e.add_a32(e.enc(uextend.bind(I32).bind(I8), r_extend, al | 0x20));
    e.add_a32(e.enc(uextend.bind(I16).bind(I8), r_extend, al | 0x20));

    e.add_a32(e.enc(sextend.bind(I32).bind(I16), r_extend, al | 0x10));
    e.add_a32(e.enc(sextend.bind(I32).bind(I8), r_extend, al));
    e.add_a32(e.enc(sextend.bind(I16).bind(I8), r_extend, al));

    for &ty in &[I8, I16, I32] {
        e.add_a32(e.enc(copy.bind(ty), r_dp_r, al | mov));
        e.add_a32(e.enc(regmove.bind(ty), r_regmov, null_bits));
        e.add_a32(e.enc(copy_nop.bind(ty), r_stacknull, null_bits));
        e.add_a32(e.enc(fill_nop.bind(ty), r_fillnull, null_bits));
    }

    e.add_a32(e.enc(copy_special, r_copy_special, null_bits));
    e.add_a32(e.enc(copy_to_ssa.bind(I32), r_copy_to_ssa, null_bits));

    e.add_a32(e.enc(copy.bind(B1), r_dp_r, al | mov));
    e.add_a32(e.enc(regmove.bind(B1), r_regmov, null_bits));
    e.add_a32(e.enc(copy_nop.bind(B1), r_stacknull, null_bits));
    e.add_a32(e.enc(fill_nop.bind(B1), r_fillnull, null_bits));
    e.add_a32(e.enc(copy_to_ssa.bind(B1), r_copy_to_ssa, null_bits));

    // Comparisons
    e.add_a32(e.enc(ifcmp.bind(I32), r_ifcmp, null_bits));
    e.add_a32(e.enc(ifcmp_imm.bind(I32), r_ifcmp_imm, null_bits));

    e.add_a32(e.enc(ifcmp_sp.bind(I32), r_ifcmp_sp, null_bits));

    e.add_a32(e.enc(icmp.bind(I32), r_icmp, null_bits));
    e.add_a32(e.enc(icmp_imm.bind(I32), r_icmp_imm, null_bits));

    e.add_a32(e.enc(trueif, r_get_iflags, null_bits));

    e.add_a32(e.enc(selectif.bind(I32), r_select_iflags, null_bits));

    // Branches
    e.add_a32(e.enc(jump, r_branch, al));

    e.add_a32(e.enc(brif, r_branch_iflags, null_bits));

    e.add_a32(e.enc(brz.bind(I32), r_test_branch, eq));
    e.add_a32(e.enc(brz.bind(B1), r_test_branch, eq));
    e.add_a32(e.enc(brnz.bind(I32), r_test_branch, ne));
    e.add_a32(e.enc(brnz.bind(B1), r_test_branch, ne));

    e.add_a32(e.enc(br_icmp.bind(I32), r_icmp_branch, null_bits));

    e.add_a32(e.enc(indirect_jump_table_br.bind(I32), r_indirect_jump, null_bits));

    e.add_a32(e.enc(jump_table_base.bind(I32), r_jt_base, null_bits));
    e.add_a32(e.enc(jump_table_entry.bind(I32), r_jt_entry, null_bits));

    // Traps
    e.add_a32(e.enc(trap, r_trap, null_bits));
    e.add_a32(e.enc(trapif, r_trap_iflags, null_bits));

    // Memory transfer
    e.add_a32(e.enc(load.bind(I32).bind(Any), r_load_u8, al));
    e.add_a32(e.enc(load.bind(I32).bind(Any), r_load, al));
    e.add_a32(e.enc(uload16.bind(I32).bind(Any), r_load16_u8, al));
    e.add_a32(e.enc(uload16.bind(I32).bind(Any), r_load16, al));
    e.add_a32(e.enc(uload8.bind(I32).bind(Any), r_load_u8, al | 0x200));
    e.add_a32(e.enc(uload8.bind(I32).bind(Any), r_load, al | 0x20));

    e.add_a32(e.enc(store.bind(I32).bind(Any), r_store_u8, al));
    e.add_a32(e.enc(store.bind(I32).bind(Any), r_store, al));
    e.add_a32(e.enc(istore16.bind(I32).bind(Any), r_store16_u8, al));
    e.add_a32(e.enc(istore16.bind(I32).bind(Any), r_store16, al));
    e.add_a32(e.enc(istore8.bind(I32).bind(Any), r_store_u8, al | 0x200));
    e.add_a32(e.enc(istore8.bind(I32).bind(Any), r_store, al | 0x20));

    // Stack
    e.add_a32(e.enc(stack_addr.bind(I32), r_stack_addr_u8, null_bits));
    e.add_a32(e.enc(stack_addr.bind(I32), r_stack_addr, null_bits));

    e.add_a32(e.enc(stack_load.bind(I32), r_stack_load_u8, null_bits));
    e.add_a32(e.enc(stack_load.bind(I32), r_stack_load, null_bits));

    e.add_a32(e.enc(stack_store.bind(I32), r_stack_store_u8, null_bits));
    e.add_a32(e.enc(stack_store.bind(I32), r_stack_store, null_bits));

    e.add_a32(e.enc(adjust_sp_down.bind(I32), r_adjustsp, null_bits));
    e.add_a32(e.enc(adjust_sp_down_imm, r_adjustsp_imm_u8, al | sub));
    e.add_a32(e.enc(adjust_sp_down_imm, r_adjustsp_imm, al | sub));
    e.add_a32(e.enc(adjust_sp_up_imm, r_adjustsp_imm_u8, al | add));
    e.add_a32(e.enc(adjust_sp_up_imm, r_adjustsp_imm, al | add));

    e.add_a32(e.enc(spill.bind(I32), r_spill, null_bits));
    e.add_a32(e.enc(spill.bind(B1), r_spill, null_bits));
    e.add_a32(e.enc(fill.bind(I32), r_fill, null_bits));
    e.add_a32(e.enc(fill.bind(B1), r_fill, null_bits));
    e.add_a32(e.enc(regspill.bind(I32), r_regspill, null_bits));
    e.add_a32(e.enc(regspill.bind(B1), r_regspill, null_bits));
    e.add_a32(e.enc(regfill.bind(I32), r_regfill, null_bits));
    e.add_a32(e.enc(regfill.bind(B1), r_regfill, null_bits));

    // Call/return
    e.add_a32(e.enc(func_addr.bind(I32), r_func_addr, null_bits));
    e.add_a32(e.enc(call_indirect.bind(I32), r_call_indirect, null_bits));
    e.add_a32(e.enc(return_, r_return, null_bits));

    // Floating point instructions
    e.add_a32(e.enc(fcmp.bind(F32), r_vfp_s_cmp, 0x14b0));
    e.add_a32(e.enc(fcmp.bind(F64), r_vfp_d_cmp, 0x14bb));

    e.add_a32(e.enc(ffcmp.bind(F32), r_vfp_s_cmp_fflags, 0x14b0));
    e.add_a32(e.enc(ffcmp.bind(F64), r_vfp_d_cmp_fflags, 0x14bb));

    e.add_a32(e.enc(trueff, r_get_fflags, null_bits));

    e.add_a32(e.enc(fadd.bind(F32), r_vfp_s_dp_rr, 0x0030));
    e.add_a32(e.enc(fadd.bind(F64), r_vfp_d_dp_rr, 0x003f));

    e.add_a32(e.enc(fsub.bind(F32), r_vfp_s_dp_rr, 0x1030));
    e.add_a32(e.enc(fsub.bind(F64), r_vfp_d_dp_rr, 0x103f));

    e.add_a32(e.enc(fmul.bind(F32), r_vfp_s_dp_rr, 0x0020));
    e.add_a32(e.enc(fmul.bind(F64), r_vfp_d_dp_rr, 0x002f));

    e.add_a32(e.enc(fdiv.bind(F32), r_vfp_s_dp_rr, 0x0080));
    e.add_a32(e.enc(fdiv.bind(F64), r_vfp_d_dp_rr, 0x008f));

    e.add_a32(e.enc(sqrt.bind(F32), r_vfp_s_dp_r, 0x31b0));
    e.add_a32(e.enc(sqrt.bind(F64), r_vfp_d_dp_r, 0x31bb));

    e.add_a32(e.enc(fma.bind(F32), r_vfp_s_fma, 0x00a0));
    e.add_a32(e.enc(fma.bind(F64), r_vfp_d_fma, 0x00af));

    e.add_a32(e.enc(fneg.bind(F32), r_vfp_s_dp_r, 0x11b0));
    e.add_a32(e.enc(fneg.bind(F64), r_vfp_d_dp_r, 0x11bb));

    e.add_a32(e.enc(fabs.bind(F32), r_vfp_s_dp_r, 0x30b0));
    e.add_a32(e.enc(fabs.bind(F64), r_vfp_d_dp_r, 0x30bb));

    e.add_a32(e.enc(copy.bind(F32), r_vfp_s_vmov, 0x10b0));
    e.add_a32(e.enc(copy.bind(F64), r_vfp_d_vmov, 0x10bb));

    e.add_a32(e.enc(fpromote.bind(F64).bind(F32), r_vfp_s2d_convert, 0x37b8));
    e.add_a32(e.enc(fdemote.bind(F32).bind(F64), r_vfp_d2s_convert, 0x37b5));

    e.add_a32(e.enc(bitcast.bind(F32).bind(I32), r_vfp_int2s_mov, 0x00));
    e.add_a32(e.enc(bitcast.bind(I32).bind(F32), r_vfp_s2int_mov, 0x01));

    e.add_a32(e.enc(arm32_vcvt_f2uint.bind(F32), r_vfp_s2int_convert, 0x3cb0));
    e.add_a32(e.enc(arm32_vcvt_f2uint.bind(F64), r_vfp_d2int_convert, 0x3cb5));

    e.add_a32(e.enc(arm32_vcvt_f2sint.bind(F32), r_vfp_s2int_convert, 0x3db0));
    e.add_a32(e.enc(arm32_vcvt_f2sint.bind(F64), r_vfp_d2int_convert, 0x7db5));

    e.add_a32(e.enc(arm32_vcvt_uint2f.bind(F32), r_vfp_int2s_convert, 0x18b0));
    e.add_a32(e.enc(arm32_vcvt_uint2f.bind(F64), r_vfp_int2d_convert, 0x58b9));

    e.add_a32(e.enc(arm32_vcvt_sint2f.bind(F32), r_vfp_int2s_convert, 0x38b0));
    e.add_a32(e.enc(arm32_vcvt_sint2f.bind(F64), r_vfp_int2d_convert, 0x78b9));

    e.add_a32(e.enc(arm32_vmov_d2ints, r_vfp_vmov_d2ints, 0x1));
    e.add_a32(e.enc(arm32_vmov_ints2d, r_vfp_vmov_ints2d, 0x0));

    e.add_a32(e.enc(load.bind(F32).bind(Any), r_vfp_s_load, 0x1));
    e.add_a32(e.enc(load.bind(F64).bind(Any), r_vfp_d_load, 0x3));

    e.add_a32(e.enc(store.bind(F32).bind(Any), r_vfp_s_store, 0x0));
    e.add_a32(e.enc(store.bind(F64).bind(Any), r_vfp_d_store, 0x2));

    e.add_a32(e.enc(spill.bind(F32), r_vfp_s_spill, 0x0));
    e.add_a32(e.enc(spill.bind(F64), r_vfp_d_spill, 0x2));

    e.add_a32(e.enc(fill.bind(F32), r_vfp_s_fill, 0x1));
    e.add_a32(e.enc(fill.bind(F64), r_vfp_d_fill, 0x3));

    e.add_a32(e.enc(regmove.bind(F32), r_vfp_s_regmove, 0x10b0));
    e.add_a32(e.enc(regmove.bind(F64), r_vfp_d_regmove, 0x10bb));

    e.add_a32(e.enc(copy_nop.bind(F32), r_stacknull_s, null_bits));
    e.add_a32(e.enc(copy_nop.bind(F64), r_stacknull_d, null_bits));

    e.add_a32(e.enc(fill_nop.bind(F32), r_fillnull_s, null_bits));
    e.add_a32(e.enc(fill_nop.bind(F64), r_fillnull_d, null_bits));

    e.add_a32(e.enc(copy_to_ssa.bind(F32), r_copy_to_ssa_s, 0x10b0));
    e.add_a32(e.enc(copy_to_ssa.bind(F64), r_copy_to_ssa_d, 0x10bb));

    e
}
