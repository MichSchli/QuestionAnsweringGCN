from models.abstract_tensorflow_model import AbstractTensorflowModel
from models.prediction import Prediction
import tensorflow as tf


class DummyTensorflowModel(AbstractTensorflowModel):

    def compute_entity_scores(self, mode):
        self.graph.initialize_zero_embeddings(dimension=2)
        entity_embeddings = self.graph.get_target_vertex_embeddings() + 0.7

        entity_embeddings = self.mlp.transform(entity_embeddings, mode)

        return entity_embeddings
