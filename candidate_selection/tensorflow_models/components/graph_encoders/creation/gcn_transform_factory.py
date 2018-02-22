from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.affine_transform import \
    AffineGcnTransform
from candidate_selection.tensorflow_models.components.graph_encoders.gcn_transforms.relu_transform import ReluTransform


class GcnTransformFactory:

    def get_transforms(self, hypergraph, transform_setting_list, gcn_instructions):
        return [self.build_transform(transform, hypergraph, gcn_instructions) for transform in transform_setting_list]

    def build_transform(self, transform_setting, hypergraph, gcn_instructions):
        if transform_setting["type"] == "affine":
            return AffineGcnTransform(transform_setting["input_dimension"], transform_setting["output_dimension"])

        if transform_setting["type"] == "relu":
            return ReluTransform()