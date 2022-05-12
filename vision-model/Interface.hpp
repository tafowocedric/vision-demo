#ifndef INTERFACE_H
#define INTERFACE_H

#include <torch.hpp>

template <typename Conv4 = torch::Tensor, typename Conv7 = torch::Tensor>
struct VGGFeatures
{
    VGGFeatures() = default;
    ~VGGFeatures()
    {
        // delete
    }
    VGGFeatures(Conv4 conv4_3_feats_, Conv7 conv7_feats_)
        : conv4_3_feats(std::move(conv4_3_feats_)), conv7_feats(std::move(conv7_feats_)) {}

    Conv4 conv4_3_feats;
    Conv7 conv7_feats;
};

template <typename Conv8 = torch::Tensor, typename Conv9 = torch::Tensor, typename Conv10 = torch::Tensor, typename Conv11 = torch::Tensor>
struct AuxiliaryConvolutionsFeatures
{
    AuxiliaryConvolutionsFeatures() = default;
    ~AuxiliaryConvolutionsFeatures()
    {
        // delete
    }
    AuxiliaryConvolutionsFeatures(Conv8 conv8_2_feats_, Conv9 conv9_2_feats_, Conv10 conv10_2_feats_, Conv11 conv11_2_feats_)
        : conv8_2_feats(std::move(conv8_2_feats_)), conv9_2_feats(std::move(conv9_2_feats_)), conv10_2_feats(std::move(conv10_2_feats_)), conv11_2_feats(std::move(conv11_2_feats_)) {}

    Conv8 conv8_2_feats;
    Conv9 conv9_2_feats;
    Conv10 conv10_2_feats;
    Conv11 conv11_2_feats;
};

template <typename Locations = torch::Tensor, typename ClassScores = torch::Tensor>
struct PredictionConvolutionsFeatures
{
    PredictionConvolutionsFeatures() = default;
    ~PredictionConvolutionsFeatures()
    {
        // delete
    }
    PredictionConvolutionsFeatures(Locations locs_, ClassScores class_score_)
        : locs(std::move(locs_)), class_score(std::move(class_score_)) {}

    Locations locs;
    ClassScores class_score;
};

#endif // INTERFACE_H