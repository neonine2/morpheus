from alibi.utils.missing_optional_dependency import import_optional

CounterfactualProto, CounterFactualProto = import_optional(
    'alibi.explainers.cfproto',
    names=['CounterfactualProto','CounterFactualProto'])  # TODO: remove in an upcoming release

__all__ = [
    "CounterfactualProto"
]
