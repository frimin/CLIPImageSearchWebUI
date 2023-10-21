import gradio as gr
from typing import Callable

def create_to_Tab(id):
    return lambda : gr.Tabs.update(selected=id)

class TopTabContext():
    def __init__(self) -> None:
        self._tree = {}
        self._ptr = None
        self._goto_request = []

    #def goto_tab(self, obj, tab_ids):
    #    self._goto_request.append((obj, tab_ids))

    #def bind_all_requests(self):
    #    for obj, tab_ids in self._goto_request:
    #        self._do_goto_tab(obj, tab_ids)

    def goto_tab(self, obj: Callable, tab_ids):
        tab_ids = tab_ids.split(".")
        ptr = self._tree["__tab__"]
        i = 0
        path_len = len(tab_ids)

        outputs=[self._tree["__tab__"]["comp"]]
        updates_fn=[]

        while True:
            if i >= path_len:
                break
            if len(ptr["childs"]) == 1 and "__tab__" in ptr["childs"]:
                outputs.append(ptr["childs"]["__tab__"]["comp"])
                ptr = ptr["childs"]["__tab__"]
            else:
                id = tab_ids[i]
                i+=1
                if id in ptr["childs"]:
                    fn = create_to_Tab(id) 
                    updates_fn.append(fn)
                else:
                    raise ValueError(f"invalid tab path: {'.'.join(tab_ids)}")

                ptr = ptr["childs"][id]

        def on_goto():
            return [i() for i in updates_fn]

        obj(on_goto, [], outputs)

class TopTab():
    class Tabs():
        def __init__(self, context: TopTabContext, **args) -> None:
            self._id = "__tab__"
            self._type = "tab"
            self._context: TopTabContext = context
            self._elem = gr.Tabs(**args)
            self._node = {
                "comp": self._elem,
                "parent": None,
                "childs": {},
            }                    

        def __enter__(self):
            self._elem.__enter__()
            if self._context._ptr is None:
                self._context._tree[self._id] = self._node 
                self._context._ptr = self._node
            else:
                self._node["parent"] = self._context._ptr
                assert self._id not in self._context._ptr["childs"]
                self._context._ptr["childs"][self._id] = self._node
                self._context._ptr = self._node

        def __exit__(self, *args):
            self._elem.__exit__(*args)
            self._context._ptr = self._context._ptr["parent"]

    class TabItem(Tabs):
        def __init__(self, context: TopTabContext, id, label=None, **args) -> None:
            if label is None:
                label = id
            self._id = id
            self._type = "item"
            self._context: TopTabContext = context
            self._elem = gr.Tab(id=id, label=label, **args)
            self._node = {
                "comp": self._elem,
                "parent": None,
                "childs": {},
            }                    

        def __enter__(self):
            self._elem.__enter__()
            if self._context._ptr is None:
                self._context._tree[self._id] = self._node 
                self._context._ptr = self._node
            else:
                self._node["parent"] = self._context._ptr
                assert self._id not in self._context._ptr["childs"]
                self._context._ptr["childs"][self._id] = self._node
                self._context._ptr = self._node

        def __exit__(self, *args):
            self._elem.__exit__(*args)
            self._context._ptr = self._context._ptr["parent"]
