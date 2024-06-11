from .registry import Registry

LossRegistry = Registry()
ModelRegistry = Registry()
BackboneRegistry = Registry()
DatasetRegistry = Registry()
TransformRegistry = Registry()
OptimizerRegistry = Registry()
SchedulerRegistry = Registry()

__all__ = [
    'Registry', 'BackboneRegistry', 'DatasetRegistry', 'TransformRegistry',
    'OptimizerRegistry'
]
