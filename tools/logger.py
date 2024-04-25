from enum import Enum


class PrintOption(Enum):
    NOT = 1
    RESUME = 2
    ALL = 3


class Logger:
    def __init__(self, inputLogFile="log.txt", printOption=PrintOption.ALL) -> None:
        self.inputLogFile = inputLogFile
        self.printOption = printOption

    def print(self, *args, end='', printOption=PrintOption.ALL):
        if (self.printOption == PrintOption.ALL or printOption == PrintOption.RESUME):
            for arg in args:
                print(arg, end)

    def printError(self, *args):
        for arg in args:
            print(arg)
