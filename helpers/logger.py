class Logger:

    log_file = None
    console_verbosity = None
    logger_verbosity = None

    def __init__(self, log_file_name, console_verbosity=1, logger_verbosity=1):
        self.log_file = open(log_file_name, "w")
        self.console_verbosity = console_verbosity
        self.logger_verbosity = logger_verbosity

    def write(self, string, verbosity_priority=3):
        if verbosity_priority <= self.logger_verbosity:
            print(string, file=self.log_file)

        if verbosity_priority <= self.console_verbosity:
            print(string)
