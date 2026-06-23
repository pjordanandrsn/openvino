// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/moe.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace internal {

MOE::MOE(const OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

const MOE::Config& MOE::get_config() const {
    return m_config;
}

void MOE::set_config(const Config& config) {
    m_config = config;
}

std::shared_ptr<ov::Node> MOE::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(internal_MOE_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<MOE>(new_args, m_config);
}

void MOE::validate_and_infer_types() {
    OV_OP_SCOPE(internal_MOE_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() >= 6,
                          "MOE requires at least 6 inputs (hidden_states, routing_weights, "
                          "topk_indices, and at least 3 weight tensors). Got: ",
                          get_input_size());

    const auto& hidden_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          hidden_et.is_real(),
                          "hidden_states (input 0) must have a floating-point element type. Got: ",
                          hidden_et);

    const auto& routing_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          routing_et.is_real(),
                          "routing_weights (input 1) must have a floating-point element type. Got: ",
                          routing_et);

    const auto& indices_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          indices_et.is_integral_number(),
                          "topk_indices (input 2) must have an integral element type. Got: ",
                          indices_et);

    set_output_type(0, hidden_et, get_input_partial_shape(0));
}

bool MOE::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_MOE_visit_attributes);
    visitor.on_attribute("expert_type", m_config.expert_type);
    visitor.on_attribute("expert_alpha", m_config.expert_alpha);
    visitor.on_attribute("expert_beta", m_config.expert_beta);
    visitor.on_attribute("gate_idx", m_config.gate_idx);

    return true;
}

}  // namespace internal
}  // namespace op

std::ostream& operator<<(std::ostream& s, const ov::op::internal::MOE::Expert_type& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<ov::op::internal::MOE::Expert_type>& EnumNames<ov::op::internal::MOE::Expert_type>::get() {
    static auto enum_names = EnumNames<ov::op::internal::MOE::Expert_type>(
        "ov::op::internal::MOE::Expert_type",
        {
            {"gemm2_bias_swiglu_clamp", ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP},
            {"gemm3_swiglu", ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU},
        });
    return enum_names;
}
}  // namespace ov
