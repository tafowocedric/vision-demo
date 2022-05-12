
#include "AuxiliaryConvolutions.hpp"

AuxiliaryConvolutionsImpl::AuxiliaryConvolutionsImpl()
{
    // Auxiliary/additional convolutions on top of the VGG base
    conv8_1 = torch::nn::Conv2d(conv_options(1024, 256, 1).padding(0));          // stride = 1, by default
    conv8_2 = torch::nn::Conv2d(conv_options(256, 512, 3).stride(2).padding(1)); // dim. reduction because stride > 1

    conv9_1 = torch::nn::Conv2d(conv_options(512, 128, 1).padding(0));
    conv9_2 = torch::nn::Conv2d(conv_options(128, 256, 3).stride(2).padding(1)); // dim. reduction because stride > 1

    conv10_1 = torch::nn::Conv2d(conv_options(256, 128, 1).padding(0));
    conv10_2 = torch::nn::Conv2d(conv_options(128, 256, 3).padding(0)); // dim. reduction because padding = 0

    conv11_1 = torch::nn::Conv2d(conv_options(256, 128, 1).padding(0));
    conv11_2 = torch::nn::Conv2d(conv_options(128, 256, 3).padding(0)); // dim. reduction because padding = 0

    register_module("conv8_1", conv8_1);
    register_module("conv8_2", conv8_2);
    register_module("conv9_1", conv9_1);

    register_module("conv9_2", conv9_2);
    register_module("conv10_1", conv10_1);
    register_module("conv10_2", conv10_2);

    register_module("conv11_1", conv11_1);
    register_module("conv11_2", conv11_2);

    // init_conv2d();
};

void AuxiliaryConvolutionsImpl::init_conv2d()
{
    torch::autograd::GradMode::set_enabled(false);
    for (auto &p : this->named_parameters())
    {
        std::string y = p.key();
        auto z = p.value(); // note that z is a Tensor, same as &p : layers->parameters

        if (y.substr(y.find(".") + 1, y.length()).compare("weight"))
            z.uniform_();

        else if (y.substr(y.find(".") + 1, y.length()).compare("bias"))
            z.normal_();
    }
    torch::autograd::GradMode::set_enabled(true);
};

AuxiliaryConvolutionsFeatures<> AuxiliaryConvolutionsImpl::forward(torch::Tensor conv7_feats)
{
    torch::Tensor x = torch::relu(conv8_1(conv7_feats)); // (N, 256, 19, 19)
    x = torch::relu(conv8_2(x));                         // (N, 512, 10, 10)
    conv8_2_feats = x;                                   // (N, 512, 10, 10)

    x = torch::relu(conv9_1(x)); // (N, 128, 10, 10)
    x = torch::relu(conv9_2(x)); // (N, 256, 5, 5)
    conv9_2_feats = x;           // (N, 256, 5, 5)

    x = torch::relu(conv10_1(x)); // (N, 128, 5, 5)
    x = torch::relu(conv10_2(x)); // (N, 256, 3, 3)
    conv10_2_feats = x;           // (N, 256, 3, 3)

    x = torch::relu(conv11_1(x));              // (N, 128, 3, 3)
    conv11_2_feats = torch::relu(conv11_2(x)); // (N, 256, 1, 1)

    // Higher-level feature maps
    return AuxiliaryConvolutionsFeatures(conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats);
};
