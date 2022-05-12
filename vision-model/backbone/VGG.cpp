
#include "VGG.hpp"

VGGImpl::VGGImpl()
{
    conv1_1 = torch::nn::Conv2d(conv_options(3, 64, 3).padding(1)); // stride = 1, by default
    conv1_2 = torch::nn::Conv2d(conv_options(64, 64, 3).padding(1));
    pool1 = torch::nn::MaxPool2d(maxpool_options(2, 2));

    conv2_1 = torch::nn::Conv2d(conv_options(64, 128, 3).padding(1));
    conv2_2 = torch::nn::Conv2d(conv_options(128, 128, 3).padding(1));
    pool2 = torch::nn::MaxPool2d(maxpool_options(2, 2));

    conv3_1 = torch::nn::Conv2d(conv_options(128, 256, 3).padding(1));
    conv3_2 = torch::nn::Conv2d(conv_options(256, 256, 3).padding(1));
    conv3_3 = torch::nn::Conv2d(conv_options(256, 256, 3).padding(1));
    pool3 = torch::nn::MaxPool2d(maxpool_options(2, 2).ceil_mode(true)); // ceiling (not floor) here for even dims

    conv4_1 = torch::nn::Conv2d(conv_options(256, 512, 3).padding(1));
    conv4_2 = torch::nn::Conv2d(conv_options(512, 512, 3).padding(1));
    conv4_3 = torch::nn::Conv2d(conv_options(512, 512, 3).padding(1));
    pool4 = torch::nn::MaxPool2d(maxpool_options(2, 2));

    conv5_1 = torch::nn::Conv2d(conv_options(512, 512, 3).padding(1));
    conv5_2 = torch::nn::Conv2d(conv_options(512, 512, 3).padding(1));
    conv5_3 = torch::nn::Conv2d(conv_options(512, 512, 3).padding(1));
    pool5 = torch::nn::MaxPool2d(maxpool_options(3, 1).padding(1)); // retains size because stride is 1 (and padding)

    // Replacements for FC6 and FC7 in VGG16
    conv6 = torch::nn::Conv2d(conv_options(512, 1024, 3).padding(6).dilation(6)); // atrous convolution
    conv7 = torch::nn::Conv2d(conv_options(1024, 1024, 1));

    register_module("conv1_1", conv1_1);
    register_module("conv1_2", conv1_2);
    register_module("pool1", pool1);

    register_module("conv2_1", conv2_1);
    register_module("conv2_2", conv2_2);
    register_module("pool2", pool2);

    register_module("conv3_1", conv3_1);
    register_module("conv3_2", conv3_2);
    register_module("conv3_3", conv3_3);
    register_module("pool3", pool3);

    register_module("conv4_1", conv4_1);
    register_module("conv4_2", conv4_2);
    register_module("conv4_3", conv4_3);
    register_module("pool4", pool4);

    register_module("conv5_1", conv5_1);
    register_module("conv5_2", conv5_2);
    register_module("conv5_3", conv5_3);
    register_module("pool5", pool5);

    register_module("conv6", conv6);
    register_module("conv7", conv7);
};

VGGFeatures<> VGGImpl::forward(torch::Tensor x)
{
    x = torch::relu(conv1_1(x)); // (N, 64, 300, 300)
    x = torch::relu(conv1_2(x)); // (N, 64, 300, 300)
    x = pool1(x);                // (N, 64, 150, 150)

    x = torch::relu(conv2_1(x)); // (N, 128, 150, 150)
    x = torch::relu(conv2_2(x)); // (N, 128, 150, 150)
    x = pool2(x);                // (N, 128, 75, 75)

    x = torch::relu(conv3_1(x)); // (N, 256, 75, 75)
    x = torch::relu(conv3_2(x)); // (N, 256, 75, 75)
    x = torch::relu(conv3_3(x)); // (N, 256, 75, 75)
    x = pool3(x);                // (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

    x = torch::relu(conv4_1(x)); // (N, 512, 38, 38)
    x = torch::relu(conv4_2(x)); // (N, 512, 38, 38)
    x = torch::relu(conv4_3(x)); // (N, 512, 38, 38)
    conv4_3_feats = x;           // (N, 512, 38, 38)
    x = pool4(x);                // (N, 512, 19, 19)

    x = torch::relu(conv5_1(x)); // (N, 512, 19, 19)
    x = torch::relu(conv5_2(x)); // (N, 512, 19, 19)
    x = torch::relu(conv5_3(x)); // (N, 512, 19, 19)
    x = pool5(x);                // (N, 512, 19, 19), pool5 does not reduce dimensions

    x = torch::relu(conv6(x)); // (N, 1024, 19, 19)

    conv7_feats = torch::relu(conv7(x)); // (N, 1024, 19, 19)

    return VGGFeatures(conv4_3_feats, conv7_feats);
}
