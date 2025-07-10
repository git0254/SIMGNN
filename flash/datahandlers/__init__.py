from .darpa_handler import DARPAHandler
from .atlas_handler import ATLASHandler

__all__ = ["DARPAHandler", "ATLASHandler"]

handler_map = {
    "theia": DARPAHandler,
    "cadets": DARPAHandler,
    "clearscope": DARPAHandler,
    "trace": DARPAHandler,
    "atlas": ATLASHandler}

path_map = {
    "theia": "C:\\Users\\55236\\Desktop\\darpa_atlas\\darpa_data\\theia",
    "cadets": "C:\\Users\\55236\\Desktop\\darpa_atlas\\darpa_data\\cadets",
    "clearScope": "C:\\Users\\55236\\Desktop\\darpa_atlas\\darpa_data\\clearScope",
    "trace": "C:\\Users\\55236\\Desktop\\darpa_atlas\\darpa_data\\trace",
    "atlas": "C:\\Users\\55236\\Desktop\\darpa_atlas\\atlas_data",
    "darpa": "C:\\Users\\55236\\Desktop\\darpa_atlas\\darpa_data"
}

def get_handler(name, train):
    cls = handler_map.get(name.lower())
    base_path = path_map.get(name)
    if base_path is None:
        raise ValueError(f"未配置数据路径: {name}")
    if cls is None:
        raise ValueError(f"未知数据集: {name}")
    return cls(base_path, train)
