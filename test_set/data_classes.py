from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class TrainingData():
    duration_ms: List[int] = field(default_factory=list)
    label: List[str] = field(default_factory=list)
    name: List[str] = field(default_factory=list)

    def append_data(self, duration_ms: int, label: str, name: str):
        self.duration_ms.append(duration_ms)
        self.label.append(label)
        self.name.append(name)

    def to_dict(self) -> Dict:
        return {
            "duration_ms": self.duration_ms,
            "label": self.label,
            "name": self.name
        }

@dataclass
class AudioData():
    audio_document_id: str
    identifier: str
    text_document_id: str
    training_data: TrainingData = field(default_factory=TrainingData)

    def append_training_data(self, duration_ms: int, label: str, name: str):
        self.training_data.append_data(
            duration_ms=duration_ms,
            label=label,
            name=name
        )

    def to_dict(self) -> Dict:
        return {
            "audio_document_id": self.audio_document_id,
            "identifier": self.identifier,
            "text_document_id": self.text_document_id,
            "training_data": self.training_data.to_dict()
        }