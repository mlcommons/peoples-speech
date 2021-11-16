
import json
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

    def get_total_duration_ms(self) -> float:
        total_duration_ms = 0
        for duration_ms in self.duration_ms:
            total_duration_ms += duration_ms
        return total_duration_ms

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

    @classmethod
    def from_json_str(cls, json_str: str):
        audio_data_args = json.loads(json_str)
        try:
            training_data = TrainingData(
                duration_ms=audio_data_args["training_data"]["duration_ms"],
                label=audio_data_args["training_data"]["label"],
                name=audio_data_args["training_data"]["name"]
            )
        except:
            training_data = None
        if training_data is None:
            return AudioData(
                identifier=audio_data_args["identifier"],
                audio_document_id=audio_data_args["audio_document_id"],
                text_document_id=audio_data_args["text_document_id"]
            )
        else:
            return AudioData(
                identifier=audio_data_args["identifier"],
                audio_document_id=audio_data_args["audio_document_id"],
                text_document_id=audio_data_args["text_document_id"],
                training_data=training_data
            )

    def append_training_data(self, duration_ms: int, label: str, name: str):
        self.training_data.append_data(
            duration_ms=duration_ms,
            label=label,
            name=name
        )
    
    def get_total_duration_ms(self) -> float:
        return self.training_data.get_total_duration_ms()

    def to_dict(self) -> Dict:
        return {
            "audio_document_id": self.audio_document_id,
            "identifier": self.identifier,
            "text_document_id": self.text_document_id,
            "training_data": self.training_data.to_dict()
        }