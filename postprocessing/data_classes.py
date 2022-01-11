import os
import json
from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class TrainingSample():
    duration_ms: int
    label: str
    name: str

    def to_nemo_line(self, audio_dir=None) -> str:
        if audio_dir is None:
            audio_filepath = self.name
        else:
            audio_filepath = os.path.join(audio_dir, self.name)
        as_nemo_dict = {
            "duration": self.duration_ms / 1000,
            "text": self.label,
            "audio_filepath": audio_filepath
        }
        return json.dumps(as_nemo_dict)
@dataclass
class TrainingData():
    samples: List[TrainingSample] = field(default_factory=list)

    def append_data(self, duration_ms: int, label: str, name: str):
        new_sample = TrainingSample(
            duration_ms=duration_ms,
            label=label,
            name=name
        )
        self.samples.append(new_sample)

    def get_total_duration_ms(self) -> float:
        total_duration_ms = 0
        for sample in self.samples:
            total_duration_ms += sample.duration_ms
        return total_duration_ms

    def to_dict(self) -> Dict:
        as_dict = {"duration_ms": [], "label": [], "name": []}
        for sample in self.samples:
            as_dict["duration_ms"].append(sample.duration_ms)
            as_dict["label"].append(sample.label)
            as_dict["name"].append(sample.name)
        return as_dict
    
    @classmethod
    def from_dict(cls, as_dict: dict):
        if "duration_ms" not in as_dict \
            or "label" not in as_dict \
            or "name" not in as_dict:
            raise ValueError("Dict representation needs keys ['duration_ms', 'label', 'name']")
        else:
            training_data = TrainingData()
            zipped_dict = zip(as_dict["duration_ms"], as_dict["label"], as_dict["name"])
            for duration_ms, label, name in zipped_dict:
                training_data.append_data(
                    duration_ms=duration_ms,
                    label=label,
                    name=name
                )
            return training_data
    
    def to_nemo_lines(self, audio_dir=None) -> List[str]:
        nemo_lines = []
        for sample in self.samples:
            nemo_line = sample.to_nemo_line(audio_dir=audio_dir)
            nemo_lines.append(nemo_line)
        return nemo_lines

@dataclass
class AudioData():
    audio_document_id: str
    identifier: str
    text_document_id: str
    training_data: TrainingData = field(default_factory=TrainingData)

    @classmethod
    def from_json_str(cls, json_str: str):
        audio_data_args = json.loads(json_str)
        training_data_dict = audio_data_args.get("training_data")
        if training_data_dict is None:
            training_data = None
        else:
            training_data = TrainingData.from_dict(training_data_dict)
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
    
    def to_nemo_lines(self, audio_dir=None) -> List[str]:
        return self.training_data.to_nemo_lines(audio_dir=audio_dir)