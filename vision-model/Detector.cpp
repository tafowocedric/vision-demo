
#include "Detector.hpp"
#include "utils.hpp"

SSD300Impl::SSD300Impl(int n_classes_, torch::Device device_) : n_classes(n_classes_),
                                                                device(device_)
{
    base = new VGGImpl();
    aux_conv = new AuxiliaryConvolutionsImpl();
    pred_conv = new PredictionConvolutionsImpl(n_classes);

    // Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
    // Rescale factor is initially set at 20, but is learned for each channel during back-prop
    rescale_factors = register_parameter("rescale_factors", torch::randn({1, 512, 1, 1}, at::dtype(torch::kFloat)));
    torch::nn::init::constant_(rescale_factors, 20);

    register_module("base", std::shared_ptr<VGGImpl>(base));
    register_module("aux_conv", std::shared_ptr<AuxiliaryConvolutionsImpl>(aux_conv));
    register_module("pred_conv", std::shared_ptr<PredictionConvolutionsImpl>(pred_conv));

    // Prior boxes
    priors_cxcy = create_prior_boxes();
};

torch::Tensor SSD300Impl::create_prior_boxes()
{
    std::vector<std::pair<std::string, int>> fmap_dims = {{"conv4_3", 38}, {"conv7", 19}, {"conv8_2", 10}, {"conv9_2", 5}, {"conv10_2", 3}, {"conv11_2", 1}};
    std::vector<std::pair<std::string, float>> obj_scales = {{"conv4_3", 0.1}, {"conv7", 0.2}, {"conv8_2", 0.375}, {"conv9_2", 0.55}, {"conv10_2", 0.725}, {"conv11_2", 0.9}};
    std::vector<std::pair<std::string, std::vector<float>>> aspect_ratios = {{"conv4_3", {1., 2., 0.5}}, {"conv7", {1., 2., 3., 0.5, .333}}, {"conv8_2", {1., 2., 3., 0.5, .333}}, {"conv9_2", {1., 2., 3., 0.5, .333}}, {"conv10_2", {1., 2., 0.5}}, {"conv11_2", {1., 2., 0.5}}};

    std::vector<std::string> fmaps;
    for (auto const &item : fmap_dims)
        fmaps.emplace_back(item.first);

    std::vector<std::vector<float>> prior_boxes;
    for (size_t k = 0; k < fmaps.size(); k++)
    {
        for (size_t i = 0; i < fmap_dims.at(k).second; i++)
        {
            for (size_t j = 0; j < fmap_dims.at(k).second; j++)
            {
                float cx = (j + 0.5) / fmap_dims.at(k).second;
                float cy = (j + 0.5) / fmap_dims.at(k).second;

                for (auto &ratio : aspect_ratios.at(k).second)
                {
                    prior_boxes.push_back({cx, cy, obj_scales.at(k).second * sqrt(ratio), obj_scales.at(k).second / sqrt(ratio)});

                    // For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    // scale of the current feature map and the scale of the next feature map
                    if (ratio == 1.)
                    {
                        float additional_scale;
                        try
                        {
                            additional_scale = sqrt(obj_scales.at(k).second * obj_scales.at(k + 1).second);
                            // For the last feature map, there is no "next" feature map
                        }
                        catch (const std::exception &e)
                        {
                            additional_scale = 1.;
                        }

                        prior_boxes.push_back({cx, cy, additional_scale, additional_scale});
                    }
                };
            }
        }
    }
    torch::Tensor tensor_prior_boxes = torch::from_blob(prior_boxes.data(), {(int)prior_boxes.size(), 4}, torch::kFloat32).to(device);
    tensor_prior_boxes.clamp_(0, 1); // (8732, 4)

    return tensor_prior_boxes;
};

std::tuple<std::vector<torch::Tensor>> SSD300Impl::detect_objects(torch::Tensor predicted_locs, torch::Tensor predicted_scores, float min_score, float max_overlap, int top_k)
{
    // TODO:: VERIFY THIS FUNCTION AGAIN

    int batch_size = predicted_locs.sizes()[0];
    int n_priors = priors_cxcy.sizes()[0];
    torch::Tensor predictiond_scores = torch::softmax(predicted_scores, 2); // (N, 8732, n_classes)

    // Lists to store final predicted boxes, labels, and scores for all images
    static std::vector<torch::Tensor> all_images_boxes;
    static std::vector<torch::Tensor> all_images_labels;
    static std::vector<torch::Tensor> all_images_scores;

    assert(n_priors == predicted_locs.sizes()[1] && n_priors == predicted_scores.sizes()[1]);

    for (int i = 0; i < batch_size; i++)
    {
        // Decode object coordinates from the form we regressed predicted boxes to
        torch::Tensor decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy)); // (8732, 4), these are fractional pt. coordinates

        // Lists to store boxes and scores for this image
        std::vector<torch::Tensor> image_boxes_list;
        std::vector<torch::Tensor> image_labels_list;
        std::vector<torch::Tensor> image_scores_list;

        // Check for each class
        for (int c = 1; c < n_classes; c++)
        {
            // Keep only predicted boxes and scores where scores for this class are above the minimum score
            torch::Tensor class_scores = predicted_scores[i].index({"...", c});                 // (8732)
            torch::Tensor score_above_min_score = (class_scores > min_score).to(torch::kUInt8); // torch.uint8 (byte) tensor, for indexing
            torch::Scalar n_above_min_score = score_above_min_score.sum().item();

            if (n_above_min_score.equal(0))
                continue;

            torch::Tensor indices = torch::nonzero(score_above_min_score);
            class_scores = class_scores.index({indices});                                   // (n_qualified), n_min_score <= 8732
            torch::Tensor class_decoded_locs = decoded_locs.index({indices}).view({-1, 4}); // (n_qualified, 4)

            // Sort predicted boxes and scores by scores
            auto [class_scores_s, sort_ind] = torch::sort(class_scores, 0, true);
            class_scores = class_scores_s;
            class_decoded_locs = class_decoded_locs.index({sort_ind}).view({-1, 4});

            // Find the overlap between predicted boxes
            torch::Tensor overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs);

            // Non - Maximum Suppression(NMS)
            // A torch.uint8(byte) tensor to keep track of which predicted boxes to suppress
            //  1 implies suppress, 0 implies don't suppress
            auto suppress = torch::zeros({n_above_min_score.toInt()}, at::dtype(torch::kUInt8)).to(device);

            // Consider each box in order of decreasing scores
            for (int box = 0; box < class_decoded_locs.sizes()[0]; box++)
            {
                if (suppress[box].item().toInt() == 1)
                    continue;

                // Suppress boxes whose overlaps(with this box) are greater than maximum overlap
                // Find such boxes and update suppress indices
                suppress = torch::max(suppress, overlap[box] > max_overlap);

                // The max operation retains previously suppressed boxes, like an 'OR' operation
                // Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0;
            }
            // Store only unsuppressed boxes for this class
            image_boxes_list.push_back(class_decoded_locs.index({1 - suppress}).to(torch::kFloat));
            image_labels_list.push_back(((1 - suppress) * c).to(at::kLong).to(device));
            image_scores_list.push_back(class_scores.index({1 - suppress}).to(torch::kFloat));
        }
        // If no object in any class is found, store a placeholder for 'background'
        if ((int)image_boxes_list.size() == 0)
        {
            image_boxes_list.push_back(torch::tensor({0., 0., 1., 1.}, {torch::kFloat}));
            image_labels_list.push_back(torch::tensor({0}, {torch::kLong}).to(device));
            image_scores_list.push_back(torch::tensor({0.}, {torch::kFloat}));
        }

        // Concatenate into single tensors
        torch::Tensor image_boxes = torch::cat(image_boxes_list, 0);   // (n_objects, 4)
        torch::Tensor image_labels = torch::cat(image_labels_list, 0); // (n_objects)
        torch::Tensor image_scores = torch::cat(image_scores_list, 0); // (n_objects)
        int n_objects = image_scores.sizes()[0];

        // Keep only the top k objects
        if (n_objects > top_k)
        {
            auto [image_scores_s, sort_ind] = torch::sort(image_scores, 0, true);
            image_scores = image_scores_s;

            image_scores = image_scores.index({torch::indexing::Slice({torch::indexing::None, top_k})});                   // (top_k)
            image_boxes = image_boxes.index({sort_ind}).index({torch::indexing::Slice({torch::indexing::None, top_k})});   // (top_k, 4)
            image_labels = image_labels.index({sort_ind}).index({torch::indexing::Slice({torch::indexing::None, top_k})}); // (top_k)
        }

        // Append to lists that store predicted boxes and scores for all images
        all_images_boxes.push_back(image_boxes);
        all_images_labels.push_back(image_labels);
        all_images_scores.push_back(image_scores);
    }

    return (all_images_boxes, all_images_labels, all_images_scores); // lists of length batch_size
}

PredictionConvolutionsFeatures<> SSD300Impl::forward(torch::Tensor x)
{
    // Run VGG base network convolutions (lower level feature map generators)
    VGGFeatures<> base_out = base->forward(x); // (N, 512, 38, 38), (N, 1024, 19, 19)

    // Rescale conv4_3 after L2 norm
    torch::Tensor norm = base_out.conv4_3_feats.pow(2).sum(1, true).sqrt(); // (N, 1, 38, 38)
    base_out.conv4_3_feats = base_out.conv4_3_feats / norm;                 // (N, 512, 38, 38)
    base_out.conv4_3_feats = base_out.conv4_3_feats * rescale_factors;      // (N, 512, 38, 38)
    // (PyTorch autobroadcasts singleton dimensions during arithmetic)

    // Run auxiliary convolutions (higher level feature map generators)
    AuxiliaryConvolutionsFeatures<> aux_conv_out = aux_conv->forward(base_out.conv7_feats); // (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

    // Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
    PredictionConvolutionsFeatures<> pred_conv_out = pred_conv->forward(base_out.conv4_3_feats, base_out.conv7_feats,
                                                                        aux_conv_out.conv8_2_feats, aux_conv_out.conv9_2_feats,
                                                                        aux_conv_out.conv10_2_feats, aux_conv_out.conv11_2_feats);

    return pred_conv_out;
};

torch::Tensor SSD300Impl::get_priors_cxcy()
{
    return priors_cxcy;
}