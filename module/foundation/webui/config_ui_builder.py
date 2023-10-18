VALUE_TYPE_RAW = "raw"
VALUE_TYPE_INT = "int"

class ConfigUIBuilder():
    def __init__(self) -> None:
        self._configs_elems = []

    def add_elems(self, comp, value_path, get_cb=None, set_cb=None, value_type="raw"):
        self._configs_elems.append({ "comp": comp, "path": value_path, "get": get_cb, "set": set_cb, "type": value_type })

    def get_elems(self):
        return self._configs_elems