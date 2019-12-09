
class Num3:
    def __init__(self, x=0, y=0, z=0, json=None):
        if json is not None:
            self.fromJson(json)
        else:
            self.x = x
            self.y = y
            self.z = z

    def toJson(self):
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }

    def fromJson(self, json):
        self.x = json['x']
        self.y = json['y']
        self.z = json['z']
