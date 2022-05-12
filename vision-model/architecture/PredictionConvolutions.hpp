/*
This python implementation is writen by sgrvinod.
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
Copyright(c) sgrvinod,
All rights reserved.
*/

#ifndef PREDICTION_CONV_H
#define PREDICTION_CONV_H

#include "../Interface.hpp"
#include "../utils.hpp"

/**
 * @brief Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
 * @brief  The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes. 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
 * @brief class scores represent the scores of each object class in each of the 8732 bounding boxes located. A high score for 'background' = no object.
 */
class PredictionConvolutionsImpl : public torch::nn::Module
{
public:
    /**
     * @brief Construct a new Prediction Convolutions Impl object
     *
     */
    PredictionConvolutionsImpl(){};

    /**
     * @brief Destroy the Prediction Convolutions Impl object
     *
     */
    ~PredictionConvolutionsImpl(){
        // delete encoder;
    };

    /**
     * @brief Construct a new Prediction Convolutions Impl object
     *
     * @param n_classes_
     */
    PredictionConvolutionsImpl(int n_classes_);

    /**
     * @brief Initialize convolution parameters.
     */
    void init_conv2d();

    /**
     * @brief Forward propagation.
     *
     * @param conv4_3_feats conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
     * @param conv7_feats conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
     * @param conv8_2_feats conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
     * @param conv9_2_feats conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
     * @param conv10_2_feats conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
     * @param conv11_2_feats conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
     * @return PredictionConvolutionsFeatures<torch::Tensor, torch::Tensor> 8732 locations and class scores (i.e. w.r.t each prior box) for each image
     */
    PredictionConvolutionsFeatures<> forward(torch::Tensor conv4_3_feats, torch::Tensor conv7_feats, torch::Tensor conv8_2_feats, torch::Tensor conv9_2_feats, torch::Tensor conv10_2_feats, torch::Tensor conv11_2_feats);

private:
    int n_classes;
    torch::nn::Conv2d loc_conv4_3{nullptr};
    torch::nn::Conv2d loc_conv7{nullptr};
    torch::nn::Conv2d loc_conv8_2{nullptr};
    torch::nn::Conv2d loc_conv9_2{nullptr};
    torch::nn::Conv2d loc_conv10_2{nullptr};
    torch::nn::Conv2d loc_conv11_2{nullptr};

    torch::nn::Conv2d cl_conv4_3{nullptr};
    torch::nn::Conv2d cl_conv7{nullptr};
    torch::nn::Conv2d cl_conv8_2{nullptr};
    torch::nn::Conv2d cl_conv9_2{nullptr};
    torch::nn::Conv2d cl_conv10_2{nullptr};
    torch::nn::Conv2d cl_conv11_2{nullptr};
};
TORCH_MODULE(PredictionConvolutions);

#endif // PREDICTION_CONV_H