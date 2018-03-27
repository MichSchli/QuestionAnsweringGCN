class TensorflowModelUpdater:

    def __init__(self):
        pass

    def update(self, model, batch):
        model_prediction = model.get_loss_graph()
        model.handle_variable_assignment(batch, "train")
        loss = model.sess.run(model_prediction, feed_dict=model.get_assignment_dict())

        print(loss)

        exit()