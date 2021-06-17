from __future__ import annotations

import attr
from typing import List, Optional
from collections import defaultdict

Namespace = List[str]

class ObjectType:
    @staticmethod
    def from_str(ty_name: str) -> ObjectType:
        return BaseType(ty_name)

@attr.s(auto_attribs=True)
class BaseType(ObjectType):
    name: str

@attr.s(auto_attribs=True)
class TypeCtor(ObjectType):
    ty_ctor: str
    ty_args: List[ObjectType]

@attr.s(auto_attribs=True)
class ObjectField:
    field_name: str
    field_type: ObjectType
    is_binding: bool = False
    use_in_sequal_reduce: bool = True

@attr.s
class ObjectMethod:
    pass

@attr.s(auto_attribs=True)
class ObjectDefinition:
    name: str
    fields: List[ObjectField]
    methods: List[ObjectMethod] = attr.Factory(list)
    inherits_from: str = "ObjectRef"
    namespace: Namespace = attr.Factory(list)
    imports: List[Namespace] = attr.Factory(list)
    final: bool = True
    docs: str = ""

    def ref_name(self) -> str:
        return self.name

    def payload_name(self) -> str:
        return self.name + "Node"

    def parent_payload_name(self) -> str:
        if self.inherits_from != "ObjectRef":
            return self.inherits_from + "Node"
        return "Object"

    def parent_ref_name(self) -> str:
        if self.inherits_from != "ObjectRef":
            return self.inherits_from
        return "ObjectRef"

    def type_key(self) -> str:
        return ".".join(self.namespace + [self.name])

    def ctor_pf(self) -> str:
        # this is a temporary hack, need to think about how to clean up ns handling
        return ".".join(self.namespace[:-1] + [self.name])
