
class Logger:
    def __init__(self, inputLogFile="log.txt", printOption=True) -> None:
        self.inputLogFile = inputLogFile
        self.printOption = printOption

    def print(self, *args, end=''):
        if (self.printOption):
            for arg in args:
                print(arg, end)

    def printError(self, *args):
        for arg in args:
            print(arg)
