
class Logger:
    def __init__(self, inputLogFile="log.txt", printOption=True) -> None:
        self.inputLogFile = inputLogFile
        self.printOption = printOption

    def print(self, *args):
        if (self.printOption):
            for arg in args:
                print(arg)
