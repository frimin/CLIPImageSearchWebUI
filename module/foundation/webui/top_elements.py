class TopElements:
    def __init__(self) -> None:
        self.msg_text = None
        self.top_tab_context = None
        self._shared = {}
        self._delay_bind = []

    def add_shared_component(self, name: str, components, **evts):
        assert name not in self._shared

        self._shared[name] = {
            "comp": components,
            "evts": evts,
        }

    def get_shared_component(self, name):
        return self._shared[name]["comp"]

    def goto_tab(self, obj, tab_ids):
        self.top_tab_context.goto_tab(obj, tab_ids)

    def delay_bind(self, fn, *args):
        self._delay_bind.append((fn, [i for i in args]))

    def do_delay(self):
        try:
            for fn, args in self._delay_bind:
                fn(self, *args)
        finally:
            self._delay_bind = None

