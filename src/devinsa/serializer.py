from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from types import GenericAlias
from typing import Any, get_args, TypeAlias

from django.core.signing import JSONSerializer

_Dataclass: TypeAlias = type


class DataclassJSONSerializer(JSONSerializer):
    def dumps(self, obj: dict) -> bytes:
        for key, val in obj.items():
            if is_dataclass(val):
                obj[key] = asdict(val)
            elif isinstance(val, list):
                for idx, elem in enumerate(val):
                    if is_dataclass(elem):
                        val[idx] = asdict(elem)
        return super().dumps(obj)

    def loads(self, data: bytes) -> dict:
        obj: dict = super().loads(data)
        for key, val in obj.items():
            if isinstance(val, dict):
                obj[key] = self.to_dataclass(val)
            elif isinstance(val, list):
                for idx, elem in enumerate(val):
                    if isinstance(elem, dict):
                        val[idx] = self.to_dataclass(elem)
        return obj

    @staticmethod
    def to_dataclass(obj: dict[str, Any]) -> dict[str, Any] | _Dataclass:
        for my_dataclass in DataclassJSONSerializer._get_dataclasses():
            temp = DataclassJSONSerializer._dataclass_from_dict(my_dataclass, obj)
            if temp != obj:
                return temp
        return obj

    @staticmethod
    def _get_dataclasses() -> Iterable[_Dataclass]:
        from importlib import import_module
        from inspect import getmembers, isclass
        return (obj for _, obj in getmembers(import_module("app.models.dataclasses"), isclass) if is_dataclass(obj))

    @staticmethod
    def _dataclass_from_dict(cls: type | GenericAlias, data: dict[str, Any] | Any) -> dict[str, Any] | _Dataclass | Any:
        try:
            fieldtypes: dict[str, type | GenericAlias] = cls.__annotations__
            return cls(**{f: DataclassJSONSerializer._dataclass_from_dict(fieldtypes[f], data[f]) for f in data})
        except (KeyError, AttributeError, TypeError):
            if isinstance(data, list):
                typ: type | None = get_args(cls)[0] if len(get_args(cls)) > 0 else None
                return [DataclassJSONSerializer._dataclass_from_dict(typ, f) for f in data]
            return data
