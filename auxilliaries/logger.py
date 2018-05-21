class Logger:

    """
    Logger designed to write either to a designated log file, to the console, or to the disk.
    """

    log_file_name = None
    logger_settings = None

    def __init__(self, log_file_name, logger_settings):
        self.log_file_name = log_file_name
        self.logger_settings = logger_settings

        self.clear_log_file(log_file_name)

    def clear_log_file(self, log_file_name):
        log_file = open(log_file_name, "w")
        log_file.close()

        for area in self.logger_settings:
            for subject in self.logger_settings[area]:
                for method in self.logger_settings[area][subject].split(","):
                    if method.startswith("disk:"):
                        method = method[5:]
                        disk_file = open(method, "w")
                        disk_file.close()

    def should_log(self, area, subject):
        return area in self.logger_settings and subject in self.logger_settings[area]

    def write_configuration(self, configuration, area, subject):
        for header, items in configuration.items():
            self.write("["+header+"]", area, subject)
            for key, value in items.items():
                self.write("\t" + key + " = " + value, area, subject)

    def write(self, message, area, subject):
        if self.should_log(area, subject):
            methods = self.logger_settings[area][subject].split(",")
            for method in methods:
                self.log_message(message, method)

    def log_message(self, message, method):
        if method == "log":
            with open(self.log_file_name, "a") as log_file:
                print(message, file=log_file)
        elif method == "console":
            print(message)
        elif method.startswith("disk:"):
            method = method[5:]
            with open(method, "a") as disk_file:
                print(message, file=disk_file)
