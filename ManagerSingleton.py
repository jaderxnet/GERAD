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


if __name__ == "__main__":
    # The client code.

    m1 = Manager()
    m1.some_business_logic()
    m2 = Manager()
    m2.some_business_logic()
    m1.some_business_logic()
    m1.some_business_logic()
    m2.some_business_logic()

    if id(m1) == id(m2):
        print("Singleton works, both variables contain the same instance.")
    else:
        print("Singleton failed, variables contain different instances.")
