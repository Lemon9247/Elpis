class Sheep:
    def __init__(self, name):
        self.name = name
        self.hunger = 10
        self.happiness = 5

    def eat(self):
        self.hunger -= 2
        self.happiness += 1

    def play(self):
        self.happiness += 2
        self.hunger += 1