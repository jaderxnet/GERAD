class ManagerSingleton(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Manager(metaclass=ManagerSingleton):
    mapa = {1: "Fazer", 2: "Fazer", 3: "Fazer", 4: "Fazer", 5: "Fazer"}
    index = 1

    def some_business_logic(self):
        """
        Finally, any singleton should define some business logic, which can be
        executed on its instance.
        """
        if (self.index in self.mapa.keys()):
            if (self.mapa[self.index] == "Fazer"):
                print(id(self), " - Fazendo: ", self.index)
                self.mapa[self.index] = "Fazendo"
            elif (self.mapa[self.index] == "Fazendo"):
                print(id(self), " - Aguardando: ", self.index)
                self.mapa[self.index] = "Feito"
            elif (self.mapa[self.index] == "Feito"):
                print(id(self), " - Feito: ", self.index)
                self.index += 1
#                print("ERRO!!! ", self.index)
        else:
            print("Map AND!")

        # ...


class VideoProcessorSingleton(metaclass=ManagerSingleton):

    def __init__(self, currentStatusItemsList):
        self.statusList = ["Ready", "Processing", "Processed", "Finished"]
        self.currentStatusItemsList = currentStatusItemsList
        self.currentIndex = 0

    def nextStatus(self):
        """
        Change to the next state
        """

        for indexStatus, status in enumerate(self.statusList):
            if (indexStatus + 2) <= len(self.statusList):
                if (self.currentStatusItemsList[self.currentIndex] == status):
                    self.currentStatusItemsList[self.currentIndex] = self.statusList[indexStatus + 1]
                    return 1
        return 0

    def nextItem(self):
        """
        Change the the item if all processing is finished
        """
        if (self.currentIndex + 1 < len(self.currentStatusItemsList) and self.currentIndex != -1):
            self.currentIndex += 1
            return 1
        return 0

    def printItemStatus(self):
        print(self.currentIndex, " - ",
              self.currentStatusItemsList[self.currentIndex])

    def printAllStatus(self):
        print(self.currentStatusItemsList)

    def filterBy(self, status):
        return self.currentStatusItemsList == status


if __name__ == '__main__':
    statusLista = ["Ready", "Ready", "Ready", "Ready"]

    statusSingleton = VideoProcessorSingleton(statusLista)

    statusSingleton.printAllStatus()
    statusSingleton.nextStatus()
    statusSingleton.printItemStatus()
    statusSingleton.nextStatus()
    statusSingleton.printItemStatus()
    statusSingleton.nextStatus()
    statusSingleton.printItemStatus()
    statusSingleton.nextStatus()
    statusSingleton.printItemStatus()
    statusSingleton.nextItem()
    statusSingleton.printItemStatus()
    statusSingleton.nextStatus()
    statusSingleton.printItemStatus()
    statusSingleton.nextItem()
    statusSingleton.nextItem()
    statusSingleton.nextItem()
    statusSingleton.nextItem()
    statusSingleton.nextItem()
    statusSingleton.nextItem()
    statusSingleton.nextStatus()
    statusSingleton.nextStatus()
    statusSingleton.nextStatus()
    statusSingleton.nextStatus()
    statusSingleton.nextStatus()
    statusSingleton.nextStatus()
    statusSingleton.printItemStatus()
    statusSingleton.printAllStatus()
