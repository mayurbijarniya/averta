from averta.ingest.swegym import SweGymAdapter

ADAPTERS = {adapter.source: adapter for adapter in (SweGymAdapter(),)}

__all__ = ["ADAPTERS", "SweGymAdapter"]
