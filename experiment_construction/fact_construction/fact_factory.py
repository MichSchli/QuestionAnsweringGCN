from experiment_construction.fact_construction.csv_facts import CsvFacts
from experiment_construction.fact_construction.freebase_facts import FreebaseFacts


class FactFactory:

    def construct_facts(self, settings):
        if settings["endpoint"]["facts"] == "freebase":
            facts = FreebaseFacts()
        else:
            facts = CsvFacts(settings["endpoint"]["file"])
        return facts