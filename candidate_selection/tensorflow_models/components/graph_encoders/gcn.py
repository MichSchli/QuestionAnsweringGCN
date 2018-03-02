from candidate_selection.tensorflow_models.components.abstract_component import AbstractComponent
import tensorflow as tf


class GCN(AbstractComponent):

    propagation_units = []
    update_units = []

    def __init__(self, settings):
        self.propagation_units = []
        self.entity_update_units = []

    def add_layer(self, propagation_unit, update_unit):
        self.propagation_units.append(propagation_unit)
        self.update_units.append(update_unit)


    def get_regularization_term(self):
        reg = tf.reduce_sum([p.get_regularization_term() for p in self.propagation_units])
        reg += tf.reduce_sum([u.get_regularization_term() for u in self.update_units])

        return reg

    def set_gate_key(self, key):
        for p in self.propagation_units:
            p.set_gate_key(key)

    def get_edge_gates(self):
        return [p.get_edge_gates() for p in self.propagation_units]

    def prepare_tensorflow_variables(self, mode="train"):
        for p in self.propagation_units:
            p.prepare_tensorflow_variables(mode=mode)

        for u in self.update_units:
            u.prepare_tensorflow_variables(mode=mode)

    def propagate(self):
        for propagation, update in zip(self.propagation_units, self.update_units):
            entity_context, event_context = propagation.propagate()
            update.update(entity_context, event_context)

