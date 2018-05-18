from example_batcher.example_batcher import ExampleBatcher


class ExampleBatcherFactory:

    def __init__(self):
        pass

    def get(self, experiment_configuration, mode):
        batch_size = int(experiment_configuration["training"]["batch_size"]) if mode == "train" else int(experiment_configuration["testing"]["batch_size"])
        batcher = ExampleBatcher(batch_size)
        return batcher