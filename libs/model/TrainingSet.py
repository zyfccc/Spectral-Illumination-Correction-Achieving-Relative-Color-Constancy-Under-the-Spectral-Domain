from libs.model.Annotation import Annotation


class TrainingSet:

    def __init__(self, json=None):
        self.id = None
        self.trueValue = None
        self.imageInfoId = None
        self.imageInfo = None
        self.background = None
        self.bgImageInfoId = None
        self.annotations = []
        self.references = []
        if json is None:
            return
        self.fromJson(json)

    def toJson(self):
        annotations = []
        for element in self.annotations:
            annotations.append(element.toJson())
        references = []
        for element in self.references:
            references.append(element.toJson())
        return {
            'id': self.id,
            'imagePath': self.imagePath,
            'trueValue': self.trueValue,
            'imageInfoId': self.imageInfoId,
            'background': self.background.toJson() if hasattr(self, 'background') and self.background is not None else None,
            'annotations': annotations,
            'references': references,
            'bgImageInfoId': self.bgImageInfoId
        }

    def fromJson(self, json):
        self.id = json.get('id')
        self.trueValue = json.get('trueValue')
        self.imagePath = json.get('imagePath')
        self.imageInfoId = json.get('imageInfoId')
        background = json.get('background')
        self.background = Annotation(
            json=background) if background is not None else None
        self.bgImageInfoId = json.get('bgImageInfoId')

        annotations = json['annotations']
        for element in annotations:
            created = Annotation(element)
            self.annotations.append(created)
        references = json['references']
        for element in references:
            created = Annotation(element)
            self.references.append(created)
