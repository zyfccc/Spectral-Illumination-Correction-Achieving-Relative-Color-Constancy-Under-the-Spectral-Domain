from libs.model.Num2 import Num2
from libs.model.Num3 import Num3


class Annotation:

    def __init__(self, json=None):
        self.id = None
        self.result = None
        self.type = None
        self.rgb = None
        if json is None:
            return
        self.fromJson(json)

    def toJson(self):
        return {
            'id': self.id,
            'result': self.result,
            'rgb': self.rgb.toJson() if self.rgb is not None else None,
            'position': self.position.toJson() if self.position is not None else None,
            'rect': self.rect.toJson() if self.rect is not None else None,
            'type': self.type
        }

    def fromJson(self, json):
        self.id = json.get('id')
        self.result = json.get('result')
        rgb = json.get('rgb')
        self.rgb = Num3(json=rgb) if rgb is not None else None
        position = json.get('position')
        self.position = Num2(json=position) if position is not None else None
        rect = json.get('rect')
        self.rect = Num2(json=rect) if rect is not None else None
        self.type = json.get('type', '0')
