#include "../op_table.h"
#include "../utils.h"

#include <array>
#include <cstdint>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/pad.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/subtract.hpp>
#include <utility>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> input,
                                            const std::array<int32_t, 8> & pads,
                                            const ov::Shape & input_shape) {
    ov::Output<ov::Node> result = std::move(input);

    const std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    const std::array<int32_t, 4> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    for (size_t axis = 0; axis < input_shape.size(); ++axis) {
        const int64_t input_dim = static_cast<int64_t>(input_shape[axis]);
        const int64_t pad_begin = pads_begin[axis];
        const int64_t pad_end = pads_end[axis];

        if (pad_begin == 0 && pad_end == 0) {
            continue;
        }

        FRONT_END_CHECK_IMPLEMENTED(input_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(input_dim + pad_begin + pad_end));
        for (int64_t index = 0; index < static_cast<int64_t>(indices.size()); ++index) {
            int64_t wrapped = (index - pad_begin) % input_dim;
            if (wrapped < 0) {
                wrapped += input_dim;
            }
            indices[static_cast<size_t>(index)] = wrapped;
        }

        auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        result = std::make_shared<ov::op::v8::Gather>(result, gather_indices, gather_axis);
    }

    return result;
}

}  // namespace

// The cacheless attention (embedding models) pads K, V and the KQ bias along the token axis up to the KV size of the
// mask: GGML_PAD(n_tokens, 256) - n_tokens. The token axis is dynamic and the converted model is reused for other
// token counts, so a constant pad would only fit the graph it was converted from: pad up to the mask's KV size.
static bool translate_mask_kv_pad(const NodeContext & context,
                                  const ov::Output<ov::Node> & input,
                                  const std::array<int32_t, 8> & pads,
                                  ov::Output<ov::Node> & res) {
    if (!context.has_input("self_kq_mask")) {
        return false;
    }
    const auto & shape = input.get_partial_shape();
    if (shape.rank().is_dynamic() || shape.rank().get_length() != 4) {
        return false;
    }
    int64_t axis = -1;
    for (int64_t i = 0; i < 4; ++i) {
        // ggml dim 3 - i
        const bool padded = pads[2 * (3 - i)] != 0 || pads[2 * (3 - i) + 1] != 0;
        if (shape[i].is_dynamic()) {
            if (axis != -1 || pads[2 * (3 - i)] != 0) {
                return false;
            }
            axis = i;
        } else if (padded) {
            return false;
        }
    }
    if (axis == -1) {
        return false;
    }

    auto mask = context.get_input("self_kq_mask");
    auto mask_shape = std::make_shared<ov::op::v3::ShapeOf>(mask, ov::element::i64);
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto n_kv = std::make_shared<ov::op::v8::Gather>(
        mask_shape, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}), gather_axis);
    auto n_cur = std::make_shared<ov::op::v8::Gather>(
        input_shape, ov::op::v0::Constant::create(ov::element::i64, {1}, {axis}), gather_axis);
    auto pad_len = std::make_shared<ov::op::v1::Subtract>(n_kv, n_cur);

    // pads_end = one_hot(axis) * pad_len
    std::vector<int64_t> one_hot(4, 0);
    one_hot[axis] = 1;
    auto pads_end = std::make_shared<ov::op::v1::Multiply>(
        ov::op::v0::Constant::create(ov::element::i64, {4}, one_hot), pad_len);
    auto pads_begin = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>(4, 0));
    auto pad_value = ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{}, {0});
    res = std::make_shared<ov::op::v1::Pad>(input, pads_begin, pads_end, pad_value, ov::op::PadMode::CONSTANT);
    return true;
}

OutputVector translate_pad(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);

    const int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "PAD requires output op params");

    const std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    const bool circular = op_params[8] != 0;

    ov::Output<ov::Node> kv_pad;
    if (!circular && translate_mask_kv_pad(context, input, pads, kv_pad)) {
        return rename_outputs_with_suffix({kv_pad}, context.get_name());
    }

    if (context.get_input_shape(0) == context.get_output_shape()) {
        return rename_outputs_with_suffix({input}, context.get_name());
    }

    if (circular) {
        auto res = translate_circular_pad(input, pads, context.get_input_shape(0).to_shape());
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    const std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    const std::vector<int64_t> pads_end = {pads[7], pads[5], pads[3], pads[1]};

    auto pads_begin_node = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    auto pads_end_node = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
    auto pad_value = ov::op::v0::Constant::create(context.get_input_type(0), ov::Shape{}, {0});
    auto res =
        std::make_shared<ov::op::v1::Pad>(input, pads_begin_node, pads_end_node, pad_value, ov::op::PadMode::CONSTANT);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
