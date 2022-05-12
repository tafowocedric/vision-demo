#pragma once

#include <map>
#include <iterator>
#include <math.h>

#include <torch.hpp>
#include <json.hpp>

using json = nlohmann::json;

// model utils

/**
 * @brief Convolutional 2D Layer Options
 *
 * @param in_planes channel input size
 * @param out_planes channel outpus size
 * @param kerner_size kernel size
 * @param stride stride
 * @param padding padding
 * @param groups group
 * @param with_bias bias
 * @param dilation dilation
 * @return torch::nn::Conv2dOptions
 */
inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                             int64_t stride = 1, int64_t padding = 0, int groups = 1,
                                             bool with_bias = true, int dilation = 1)
{
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.bias(with_bias);
    conv_options.groups(groups);
    conv_options.dilation(dilation);
    return conv_options;
}

/**
 * @brief Dropout Layer Options
 *
 * @param p droupout precentage
 * @param inplace inplace
 * @return torch::nn::Dropout2dOptions
 */
inline torch::nn::Dropout2dOptions dropout_options(float p, bool inplace)
{
    torch::nn::Dropout2dOptions dropoutoptions(p);
    dropoutoptions.inplace(inplace);
    return dropoutoptions;
}

/**
 * @brief Max Pool Layer Options
 *
 * @param kerner_size kernel size
 * @param stride stride
 * @return torch::nn::MaxPool2dOptions
 */
inline torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride)
{
    torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
    maxpool_options.stride(stride);
    return maxpool_options;
}

/**
 * @brief Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
 *   They are decoded into center-size coordinates.
 *   This is the inverse of the function above.
 *
 * @param gcxgcy encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
 * @param priors_cxcy prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
 * @return torch::Tensor decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
 */
inline torch::Tensor gcxgcy_to_cxcy(torch::Tensor gcxgcy, torch::Tensor priors_cxcy)
{
    torch::Tensor cx = gcxgcy.index({"...", torch::indexing::Slice({torch::indexing::None, 2})}) *
                       priors_cxcy.index({"...", torch::indexing::Slice({2, torch::indexing::None})});

    torch::Tensor cy = priors_cxcy.index({"...", torch::indexing::Slice({torch::indexing::None, 2})});

    torch::Tensor g_w = gcxgcy.index({"...", torch::indexing::Slice({2, torch::indexing::None})});
    torch::Tensor g_h = priors_cxcy.index({"...", torch::indexing::Slice({2, torch::indexing::None})});

    return torch::cat({cx / 10 + cy, torch::exp(g_w / 5) * g_h}, 1);
}

/**
 * @brief Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
 *   For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
 *   For the size coordinates, scale by the size of the prior box, and convert to the log-space.
 *   In the model, we are predicting bounding box coordinates in this encoded form.
 *
 * @param cxcy bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
 * @param priors_cxcy prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
 * @return torch::Tensor encoded bounding boxes, a tensor of size (n_priors, 4)
 */
inline torch::Tensor cxcy_to_gcxgcy(torch::Tensor cxcy, torch::Tensor priors_cxcy)
{
    torch::Tensor g_c_x = cxcy.index({"...", torch::indexing::Slice({torch::indexing::None, 2})}) -
                          priors_cxcy.index({"...", torch::indexing::Slice({torch::indexing::None, 2})});

    torch::Tensor g_c_y = priors_cxcy.index({"...", torch::indexing::Slice({2, torch::indexing::None})}) / 10;

    torch::Tensor g_w = cxcy.index({"...", torch::indexing::Slice({2, torch::indexing::None})}) /
                        priors_cxcy.index({"...", torch::indexing::Slice({2, torch::indexing::None})});

    return torch::cat({g_c_x / g_c_y, torch::log(g_w) * 5}, 1); // g_c_x, g_c_y, g_w, g_h
}

/**
 * @brief Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
 *
 * @param cxcy bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
 * @return torch::Tensor bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
 */
inline torch::Tensor cxcy_to_xy(torch::Tensor cxcy)
{
    torch::Tensor x = cxcy.index({"...", torch::indexing::Slice({torch::indexing::None, 2})});
    torch::Tensor y = cxcy.index({"...", torch::indexing::Slice({2, torch::indexing::None})});

    return torch::cat({x - (y / 2), x + (y / 2)}, 1); // x_min, y_min, x_max, y_max
}

/**
 * @brief Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
 *
 * @param xy bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
 * @return torch::Tensor bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
 */
inline torch::Tensor xy_to_cxcy(torch::Tensor xy)
{
    torch::Tensor c_x = xy.index({"...", torch::indexing::Slice({2, torch::indexing::None})});
    torch::Tensor c_y = xy.index({"...", torch::indexing::Slice({torch::indexing::None, 2})});

    return torch::cat({(c_x + c_y) / 2, (c_x - c_y)}, 1); // c_x, c_y, w, h
}

/**
 * @brief Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
 *
 * @param set_1 set 1, a tensor of dimensions (n1, 4)
 * @param set_2 set 2, a tensor of dimensions (n2, 4)
 * @return torch::Tensor intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
 */
inline torch::Tensor find_intersection(torch::Tensor set_1, torch::Tensor set_2)
{
    torch::Tensor lower_bounds = torch::max(
        set_1.index({"...", torch::indexing::Slice(torch::indexing::None, 2)}).unsqueeze(1),
        set_2.index({"...", torch::indexing::Slice(torch::indexing::None, 2)}).unsqueeze(0)); // (n1, n2, 2)

    torch::Tensor upper_bounds = torch::min(
        set_1.index({"...", torch::indexing::Slice(2, torch::indexing::None)}).unsqueeze(1),
        set_2.index({"...", torch::indexing::Slice(2, torch::indexing::None)}).unsqueeze(0)); // (n1, n2, 2)

    torch::Tensor intersection_dims = torch::clamp(upper_bounds - lower_bounds, 0);   // (n1, n2, 2)
    return intersection_dims.index({"...", 0}) * intersection_dims.index({"...", 1}); // (n1, n2)
}

/**
 * @brief Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
 *
 * @param set_1 set 1, a tensor of dimensions (n1, 4)
 * @param set_2 set 2, a tensor of dimensions (n2, 4)
 * @return torch::Tensor Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
 */
inline torch::Tensor find_jaccard_overlap(torch::Tensor set_1, torch::Tensor set_2)
{
    // Find intersections
    torch::Tensor intersection = find_intersection(set_1, set_2); // (n1, n2)

    // Find areas of each box in both sets
    torch::Tensor areas_set_1 = (set_1.index({"...", 2}) - set_1.index({"...", 0})) * (set_1.index({"...", 3}) - set_1.index({"...", 1})); // (n1)
    torch::Tensor areas_set_2 = (set_2.index({"...", 2}) - set_2.index({"...", 0})) * (set_2.index({"...", 3}) - set_2.index({"...", 1})); // (n2)

    // Find the union
    torch::Tensor union_ = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection; // (n1, n2)

    // // TODO:: REMOVE from "+" HERE
    // return intersection / (union_ + (torch::rand({1}) * 2));
    // std::cout << union_ << std::endl;
    return intersection / union_;
}
