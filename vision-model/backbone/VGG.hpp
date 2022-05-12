/*
This python implementation is writen by sgrvinod.
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
Copyright(c) sgrvinod,
All rights reserved.
*/

#ifndef VGG_H
#define VGG_H

#include "../Interface.hpp"
#include "../utils.hpp"

/**
 * @brief VGG base convolutions to produce lower-level feature maps.
 */
class VGGImpl : public torch::nn::Module
{
public:
    /**
     * @brief Construct a new VGGImpl object
     *
     */
    VGGImpl();

    /**
     * @brief Forward propagation.
     *
     * @param x images, a tensor of dimensions (N, 3, 300, 300)
     * @return VGGFeatures<> lower-level feature maps conv4_3 and conv7
     */
    VGGFeatures<> forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::MaxPool2d pool1{nullptr};

    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::MaxPool2d pool2{nullptr};

    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::Conv2d conv3_3{nullptr};
    torch::nn::MaxPool2d pool3{nullptr};

    torch::nn::Conv2d conv4_1{nullptr};
    torch::nn::Conv2d conv4_2{nullptr};
    torch::nn::Conv2d conv4_3{nullptr};
    torch::nn::MaxPool2d pool4{nullptr};
    torch::Tensor conv4_3_feats;

    torch::nn::Conv2d conv5_1{nullptr};
    torch::nn::Conv2d conv5_2{nullptr};
    torch::nn::Conv2d conv5_3{nullptr};
    torch::nn::MaxPool2d pool5{nullptr};

    torch::nn::Conv2d conv6{nullptr};
    torch::nn::Conv2d conv7{nullptr};
    torch::Tensor conv7_feats;
};
TORCH_MODULE(VGG);

#endif // VGG_H