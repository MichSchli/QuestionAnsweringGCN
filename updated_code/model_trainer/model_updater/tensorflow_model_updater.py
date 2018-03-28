import tensorflow as tf

class TensorflowModelUpdater:

    optimize_func = None

    def __init__(self, learning_rate, gradient_clipping):
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping

    def compute_update_graph(self, model):
        if self.optimize_func is None:
            parameters_to_optimize = tf.trainable_variables()
            opt_func = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients = tf.gradients(model.get_loss_graph(), parameters_to_optimize)
            grad_func = tf.clip_by_global_norm(gradients, self.gradient_clipping)[0]
            self.optimize_func = opt_func.apply_gradients(zip(grad_func, parameters_to_optimize))
        return self.optimize_func

    def update(self, model, batch):
        model_loss = model.get_loss_graph()
        model_update = self.compute_update_graph(model)

        model.handle_variable_assignment(batch, "train")
        loss, _ = model.sess.run([model_loss, model_update], feed_dict=model.get_assignment_dict())

        print(loss)

        exit()