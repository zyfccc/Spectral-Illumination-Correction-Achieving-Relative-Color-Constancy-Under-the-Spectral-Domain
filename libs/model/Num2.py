import math


class Num2:
    def __init__(self, x=0, y=0, json=None):
        if json is not None:
            self.fromJson(json)
        else:
            self.x = x
            self.y = y

    def toJson(self):
        return {
            'x': self.x,
            'y': self.y
        }

    def fromJson(self, json):
        self.x = json['x']
        self.y = json['y']

    def distance(self, num):
        return math.sqrt((self.x - num.x) * (self.x - num.x) + (self.y - num.y) * (self.y - num.y))
