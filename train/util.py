
from dataset.emotic import EmoticItem
from dataset.emobank import EmoBankData
from torch import DeviceObjType
from typing import TypeVar

Data = TypeVar("Data", EmoticItem, EmoBankData)

def to_device(data: Data, device: DeviceObjType) -> Data:
    return type(data)({ k: v.to(device) for k, v in data.items() })