class Logger:

    log_file = None
    console_verbosity = None
    logger_verbosity = None
    log_file_name = None

    def __init__(self, log_file_name, console_verbosity=1, logger_verbosity=1):
        self.log_file = open(log_file_name, "w")
        self.log_file.close()
        self.log_file_name = log_file_name
        self.console_verbosity = console_verbosity
        self.logger_verbosity = logger_verbosity

    def write(self, string, verbosity_priority=3):
        with open(self.log_file_name, "a") as log_file:
            if verbosity_priority <= self.logger_verbosity:
                print(string, file=log_file)

        if verbosity_priority <= self.console_verbosity:
            print(string)
