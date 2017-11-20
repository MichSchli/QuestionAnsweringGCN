class NameMapper:

    inner = None

    def __init__(self, inner):
        self.inner = inner

    def enrich(self, instances):
        instances = self.inner.enrich(instances)

        for instance in instances:
            instance["neighborhood"].make_name_map()
            yield instance