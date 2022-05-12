/*
This python implementation is writen by sgrvinod.
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
Copyright(c) sgrvinod,
All rights reserved.
*/

#ifndef AUXILIARY_CONV_H
#define AUXILIARY_CONV_H

#include "../Interface.hpp"
#include "../utils.hpp"

/**
 * @brief Additional convolutions to produce higher-level feature maps.
 *
 */
class AuxiliaryConvolutionsImpl : public torch::nn::Module
{
public:
    /**
     * @brief Construct a new Auxiliary Convolutions Impl object. Additional convolutions to produce higher-level feature maps.
     */
    AuxiliaryConvolutionsImpl();

    /**
     * @brief Destroy the Auxiliary Convolutions Impl object
     *
     */
    ~AuxiliaryConvolutionsImpl()
    {
        // delete encoder;
    }

    /**
     * @brief Initialize convolution parameters.
     */
    void init_conv2d();

    /**
     * @brief Forward propagation.
     *
     * @param conv7_feats lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
     * @return AuxiliaryConvolutionsFeatures<> higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
     */
    AuxiliaryConvolutionsFeatures<> forward(torch::Tensor conv7_feats);

private:
    torch::nn::Conv2d conv8_1{nullptr};
    torch::nn::Conv2d conv8_2{nullptr};
    torch::Tensor conv8_2_feats;

    torch::nn::Conv2d conv9_1{nullptr};
    torch::nn::Conv2d conv9_2{nullptr};
    torch::Tensor conv9_2_feats;

    torch::nn::Conv2d conv10_1{nullptr};
    torch::nn::Conv2d conv10_2{nullptr};
    torch::Tensor conv10_2_feats;

    torch::nn::Conv2d conv11_1{nullptr};
    torch::nn::Conv2d conv11_2{nullptr};
    torch::Tensor conv11_2_feats;
};
TORCH_MODULE(AuxiliaryConvolutions);

#endif // AUXILIARY_CONV_H