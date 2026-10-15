# ruff: noqa: B018
"""Names vulture cannot see being used: attributes set on third-party objects and pytest hooks.

Run: vulture app.py teamMaker tests vulture_whitelist.py
"""

from types import SimpleNamespace

_ = SimpleNamespace()

# OR-Tools solver parameters and callback (set/called by the library)
_.max_time_in_seconds
_.num_workers
_.random_seed
_.use_lns_only
_.num_full_subsolvers
_.on_solution_callback

# ruamel.yaml reads this attribute to keep quotes in config.yaml
_.preserve_quotes

# pytest picks these up by name
_.pytestmark
_._clean_caches
_.clear_cache
