// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/moe.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace ov;
using namespace ov::op::internal;
using namespace testing;

namespace {

// Build a minimal valid GEMM3_SWIGLU MOE with 6 inputs.
// hidden: [batch, hidden], routing: [ne, batch, topk, 1], indices: [batch, topk],
// w0/w1: [ne, inter, hidden], w2: [ne, hidden, inter]
shared_ptr<MOE> make_moe(element::Type hidden_et = element::f32,
                         element::Type routing_et = element::f32,
                         element::Type indices_et = element::i64,
                         size_t num_inputs = 6) {
    const PartialShape hidden_shape{32, 2048};
    const PartialShape routing_shape{8, 1, 32, 1};
    const PartialShape indices_shape{32, 2};
    const Shape w_shape{8, 768, 2048};
    const Shape w2_shape{8, 2048, 768};

    auto hidden = make_shared<op::v0::Parameter>(hidden_et, hidden_shape);
    auto routing = make_shared<op::v0::Parameter>(routing_et, routing_shape);
    auto indices = make_shared<op::v0::Parameter>(indices_et, indices_shape);
    auto w0 = op::v0::Constant::create(hidden_et, w_shape, {1.f});
    auto w1 = op::v0::Constant::create(hidden_et, w_shape, {1.f});
    auto w2 = op::v0::Constant::create(hidden_et, w2_shape, {1.f});

    OutputVector inputs{hidden, routing, indices, w0, w1, w2};
    // Pad with extra inputs when num_inputs > 6 (shared-expert scenario).
    for (size_t i = 6; i < num_inputs; ++i) {
        inputs.push_back(op::v0::Constant::create(hidden_et, w_shape, {1.f}));
    }

    MOE::Config cfg;
    cfg.expert_type = MOE::Expert_type::GEMM3_SWIGLU;
    return make_shared<MOE>(inputs, cfg);
}

}  // namespace

// ---------------------------------------------------------------------------
// Happy-path: output type and shape propagate from hidden_states (input 0).
// ---------------------------------------------------------------------------

TEST(type_prop_moe, gemm3_output_type_and_shape) {
    auto op = make_moe();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{32, 2048}));
}

TEST(type_prop_moe, gemm3_dynamic_hidden_shape_propagated) {
    const PartialShape hidden_shape{-1, 2048};
    auto hidden = make_shared<op::v0::Parameter>(element::f32, hidden_shape);
    auto routing = make_shared<op::v0::Parameter>(element::f32, PartialShape{8, 1, -1, 1});
    auto indices = make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, 2});
    auto w = op::v0::Constant::create(element::f32, Shape{8, 768, 2048}, {1.f});
    auto w2 = op::v0::Constant::create(element::f32, Shape{8, 2048, 768}, {1.f});

    MOE::Config cfg;
    cfg.expert_type = MOE::Expert_type::GEMM3_SWIGLU;
    auto op = make_shared<MOE>(OutputVector{hidden, routing, indices, w, w, w2}, cfg);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 2048}));
}

TEST(type_prop_moe, accepts_f16_hidden_states) {
    auto op = make_moe(element::f16, element::f16, element::i32);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
}

TEST(type_prop_moe, accepts_i32_topk_indices) {
    auto op = make_moe(element::f32, element::f32, element::i32);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop_moe, accepts_ten_inputs_shared_expert) {
    // FuseMOESharedExpert produces 10-input MOEs; validation must allow this.
    auto op = make_moe(element::f32, element::f32, element::i64, 10);
    EXPECT_EQ(op->get_input_size(), 10);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

// ---------------------------------------------------------------------------
// Negative: too few inputs.
// ---------------------------------------------------------------------------

TEST(type_prop_moe, too_few_inputs_throws) {
    auto hidden = make_shared<op::v0::Parameter>(element::f32, PartialShape{32, 2048});
    auto routing = make_shared<op::v0::Parameter>(element::f32, PartialShape{8, 1, 32, 1});
    auto indices = make_shared<op::v0::Parameter>(element::i64, PartialShape{32, 2});
    auto w = op::v0::Constant::create(element::f32, Shape{8, 768, 2048}, {1.f});

    MOE::Config cfg;
    cfg.expert_type = MOE::Expert_type::GEMM3_SWIGLU;
    // Only 5 inputs — missing the third weight tensor.
    OV_EXPECT_THROW(make_shared<MOE>(OutputVector{hidden, routing, indices, w, w}, cfg),
                    NodeValidationFailure,
                    HasSubstr("at least 6 inputs"));
}

// ---------------------------------------------------------------------------
// Negative: wrong element types.
// ---------------------------------------------------------------------------

TEST(type_prop_moe, integral_hidden_states_throws) {
    OV_EXPECT_THROW(make_moe(element::i32, element::f32, element::i64),
                    NodeValidationFailure,
                    HasSubstr("hidden_states (input 0) must have a floating-point"));
}

TEST(type_prop_moe, integral_routing_weights_throws) {
    OV_EXPECT_THROW(make_moe(element::f32, element::i32, element::i64),
                    NodeValidationFailure,
                    HasSubstr("routing_weights (input 1) must have a floating-point"));
}

TEST(type_prop_moe, float_topk_indices_throws) {
    OV_EXPECT_THROW(make_moe(element::f32, element::f32, element::f32),
                    NodeValidationFailure,
                    HasSubstr("topk_indices (input 2) must have an integral"));
}
