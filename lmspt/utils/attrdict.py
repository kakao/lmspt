"""Minimal attribute-accessible dict — a permissive, dependency-free drop-in
replacement for ``easydict.EasyDict`` (which is LGPL-3.0).

Supports both attribute and item access and recursively wraps nested
dicts / lists, matching the subset of EasyDict behavior this codebase relies on::

    d = AttrDict({"x": 1, "vq/codebook_loss": 2, "nested": {"a": 3}})
    d.x            # 1        (attribute access)
    d["vq/codebook_loss"]     # 2  (item access — key isn't a valid identifier)
    d.nested.a     # 3        (recursive)
"""


class AttrDict(dict):
    def __init__(self, d=None, **kwargs):
        super().__init__()
        for k, v in {**(dict(d) if d else {}), **kwargs}.items():
            self[k] = v

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        elif isinstance(value, (list, tuple)):
            value = type(value)(
                AttrDict(v) if isinstance(v, dict) and not isinstance(v, AttrDict) else v
                for v in value
            )
        super().__setitem__(key, value)

    __setattr__ = __setitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
