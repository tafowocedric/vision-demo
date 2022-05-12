#include "architecture/AuxiliaryConvolutions.hpp"
#include "architecture/PredictionConvolutions.hpp"
#include "backbone/VGG.hpp"

/**
 * @brief The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
 */
class SSD300Impl : public torch::nn::Module
{
public:
    /**
     * @brief Construct a new SSD300Impl object
     *
     * @param n_classes_ number of classes
     * @param device torch device (CPU/CUDA)
     */
    SSD300Impl(int n_classes_, torch::Device device_);

    /**
     * @brief Create a prior boxes object; Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
     *
     * @return torch::Tensor prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
     */
    torch::Tensor create_prior_boxes();

    /**
     * @brief Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects. For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
     *
     * @param predicted_locs predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
     * @param predicted_scores class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
     * @param min_score minimum threshold for a box to be considered a match for a certain class
     * @param max_overlap maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
     * @param top_k if there are a lot of resulting detection across all classes, keep only the top 'k'
     * @return std::tuple<std::vector<torch::Tensor>> detections (boxes, labels, and scores), lists of length batch_size
     */
    std::tuple<std::vector<torch::Tensor>> detect_objects(torch::Tensor predicted_locs, torch::Tensor predicted_scores, float min_score, float max_overlap, int top_k);

    /**
     * @brief Forward propagation.
     *
     * @param x images, a tensor of dimensions (N, 3, 300, 300)
     * @return PredictionConvolutionsFeatures<> 8732 locations and class scores (i.e. w.r.t each prior box) for each image
     */
    PredictionConvolutionsFeatures<> forward(torch::Tensor x);

    /**
     * @brief Get the priors cxcy object
     *
     * @return torch::Tensor
     */
    torch::Tensor get_priors_cxcy();

private:
    int n_classes;
    torch::Device device{NULL};
    torch::Tensor priors_cxcy;

    VGGImpl *base;
    AuxiliaryConvolutionsImpl *aux_conv;
    PredictionConvolutionsImpl *pred_conv;

    torch::Tensor rescale_factors;
};
TORCH_MODULE(SSD300);