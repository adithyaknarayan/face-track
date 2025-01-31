from dataclasses import dataclass

@dataclass
class FaceMetaData:
    timestamp:float = None
    x:int = None
    y:int = None
    h:int = None
    w:int = None