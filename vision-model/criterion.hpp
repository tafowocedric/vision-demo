#ifndef CRITERION_H
#define CRITERION_H

#pragma once

#include "utils.hpp"

/**
 * @brief The MultiBox loss, a loss function for object detection.
 *  This is a combination of:
 *  (1) a localization loss for the predicted locations of the boxes, and
 *  (2) a confidence loss for the predicted class scores.
 */
class MultiBoxLoss : public torch::nn::Module
{

public:
    /**
     * @brief Construct a new Multi Box Loss object
     *
     * @param priors_cxcy_ prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
     * @param threshold_ object threshold
     * @param neg_pos_ratio_ negative position ratio
     * @param alpha_
     */
    MultiBoxLoss(torch::Tensor priors_cxcy_, torch::Device device_, float threshold_, int8_t neg_pos_ratio_, float alpha_);
    ~MultiBoxLoss(){
        // delete
    };

    torch::Tensor forward(torch::Tensor predicted_locs, torch::Tensor predicted_scores, torch::Tensor boxes, torch::Tensor labels);

private:
    torch::Tensor priors_cxcy;
    torch::Tensor priors_xy;
    float threshold;
    int8_t neg_pos_ratio;
    float alpha;
    torch::Device device{NULL};

    // loss functions
    torch::nn::L1Loss smooth_l1;
    torch::nn::CrossEntropyLoss cross_entropy;
};

MultiBoxLoss::MultiBoxLoss(torch::Tensor priors_cxcy_, torch::Device device_, float threshold_ = 0.5, int8_t neg_pos_ratio_ = 3, float alpha_ = 1.0) : priors_cxcy(priors_cxcy_), threshold(threshold_), neg_pos_ratio(neg_pos_ratio_), alpha(alpha_), device(device_)
{
    priors_xy = cxcy_to_xy(priors_cxcy);

    torch::nn::L1LossOptions l1Options = torch::nn::L1LossOptions();
    torch::nn::CrossEntropyLossOptions crossEntropyOptions = torch::nn::CrossEntropyLossOptions().reduction(torch::kNone);

    smooth_l1 = torch::nn::L1Loss(l1Options);
    cross_entropy = torch::nn::CrossEntropyLoss(crossEntropyOptions);
}

torch::Tensor MultiBoxLoss::forward(torch::Tensor predicted_locs, torch::Tensor predicted_scores, torch::Tensor boxes, torch::Tensor labels)
{
    auto batch_size = predicted_locs.sizes()[0];
    auto n_priors = priors_cxcy.sizes()[0];
    auto n_classes = predicted_scores.sizes()[2];

    assert(n_priors == predicted_locs.sizes()[1] && n_priors == predicted_scores.sizes()[1]);

    torch::Tensor true_locs = torch::zeros({batch_size, n_priors, 4}, at::kFloat).to(device); // (N, 8732, 4)
    torch::Tensor true_classes = torch::zeros({batch_size, n_priors}, at::kLong).to(device);  // (N, 8732)

    // For each image
    for (size_t i = 0; i < batch_size; i++)
    {
        auto n_objects = boxes[i].sizes()[0];

        torch::Tensor overlap = find_jaccard_overlap(boxes[i], priors_xy); // (n_objects, 8732)

        // For each prior, find the object that has the maximum overlap
        auto [overlap_for_each_prior, object_for_each_prior] = overlap.max(0); // (8732)

        // We don't want a situation where an object is not represented in our positive (non-background) priors -
        // 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
        // 2. All priors with the object may be assigned as background based on the threshold (0.5).

        // To remedy this -
        // First, find the prior that has the maximum overlap for each object.
        auto [_, prior_for_each_object] = overlap.max(1); // (N_o)

        // Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
        // object_for_each_prior[prior_for_each_object] = [n_objects]()
        // {
        //     torch::Tensor range = torch::zeros({n_objects}, at::kLong);
        //     for (int i = 0; i < n_objects; i++)
        //     {
        //         range[i] = torch::tensor({i});
        //     }
        //     return range;
        // }();
        // std::cout << object_for_each_prior.sizes() << std::endl;
        // std::cout << overlap << std::endl;

        //     // To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
        //     overlap_for_each_prior[prior_for_each_object] = 1.;

        //     // Labels for each prior
        //     torch::Tensor label_for_each_prior = labels[i][object_for_each_prior]; // (8732)
        //     // Set priors whose overlaps with objects are less than the threshold to be background (no object)
        //     label_for_each_prior[overlap_for_each_prior < threshold] = 0; // (8732)

        //     // Store
        //     true_classes[i] = label_for_each_prior;

        //     // Encode center-size object coordinates into the form we regressed predicted boxes to
        //     true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), priors_cxcy); // (8732, 4)
    }

    // // Identify priors that are positive (object/non-background)
    // torch::Tensor positive_priors = true_classes.nonzero(); // (N, 8732)

    // // LOCALIZATION LOSS
    // // Localization loss is computed only over positive (non-background) priors
    // torch::Tensor loc_loss = smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]);

    // // Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
    // // So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

    // // CONFIDENCE LOSS

    // // Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
    // // That is, FOR EACH IMAGE,
    // // we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
    // // This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

    // // Number of positive and hard-negative priors per image
    // torch::Tensor n_positives = positive_priors.sum(1);           // (N)
    // torch::Tensor n_hard_negatives = neg_pos_ratio * n_positives; // (N)

    // // First, find the loss for all priors
    // torch::Tensor conf_loss_all = cross_entropy(predicted_scores.view({-1, n_classes}), true_classes.view({-1})); // (N * 8732)
    // conf_loss_all = conf_loss_all.view({batch_size, n_priors});                                                   // (N, 8732)

    // // We already know which priors are positive
    // torch::Tensor conf_loss_pos = conf_loss_all[positive_priors]; // (sum(n_positives))

    // // Next, find which priors are hard-negative
    // // To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
    // torch::Tensor conf_loss_neg = conf_loss_all.clone(); // (N, 8732)
    // conf_loss_neg[positive_priors] = 0.;                 // (N, 8732), positive priors are ignored (never in top n_hard_negatives)

    // auto [conf_loss_neg_s, _] = conf_loss_neg.sort(1, true); // (N, 8732), sorted by decreasing hardness
    // conf_loss_neg = conf_loss_neg_s;

    // torch::Tensor hardness_ranks = [n_priors]()
    // {
    //     torch::Tensor range = torch::zeros({n_priors}, at::kLong);
    //     for (int i = 0; i < n_priors; i++)
    //     {
    //         range[i] = torch::tensor({i});
    //     }
    //     return range;
    // }();
    // hardness_ranks.unsqueeze(0).expand_as(conf_loss_neg).to(device); // (N, 8732)

    // torch::Tensor hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1); // (N, 8732)
    // torch::Tensor conf_loss_hard_neg = conf_loss_neg[hard_negatives];              // (sum(n_hard_negatives))

    // // As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
    // torch::Tensor conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().item<float>(); // (), scalar

    // // TOTAL LOSS
    // return conf_loss + alpha * loc_loss;
    return torch::tensor({0.0});
}

#endif // CRITERION_H