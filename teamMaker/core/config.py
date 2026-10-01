"""Configuration loading for Team Maker.

A config file may declare ``extends: other.yaml`` (path relative to the file
itself). The chain is deep-merged on top of DEFAULTS with the child winning.
When ``config/config.yaml`` does not exist, ``config/config.example.yaml`` is
used. Unknown and deprecated keys produce warnings instead of failing.
"""

import copy
import os
import random
import warnings

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LOCAL_CONFIG_NAME = "config.yaml"
EXAMPLE_CONFIG_NAME = "config.example.yaml"

DEFAULT_RANK_VALUES = {
    "Iron 1": 1,
    "Iron 2": 2,
    "Iron 3": 3,
    "Bronze 1": 4,
    "Bronze 2": 5,
    "Bronze 3": 6,
    "Silver 1": 7,
    "Silver 2": 8,
    "Silver 3": 9,
    "Gold 1": 10,
    "Gold 2": 11,
    "Gold 3": 12,
    "Platinum 1": 13,
    "Platinum 2": 14,
    "Platinum 3": 15,
    "Diamond 1": 16,
    "Diamond 2": 17,
    "Diamond 3": 18,
    "Ascendant 1": 19,
    "Ascendant 2": 20,
    "Ascendant 3": 21,
    "Immortal 1": 23,
    "Immortal 2": 25,
    "Immortal 3": 27,
    "Radiant": 30,
}

# Optimizer defaults (consumed by the Phase 4 optimizer).
OPTIMIZER_DEFAULTS = {
    "iterations": 50000000,
    "restarts": 8,
    "time_limit_s": 180,
    "t0": None,
    "t_end": None,
    "target_range": 0.1,
    "target_role_penalty": 0.0,
    "range_weight": 1.0,
    "std_weight": 0.2,
    "role_weight": 0.5,
    "cluster_regions": ["NA"],
    "cluster_max_teams": None,
    "cluster_weight": 2.0,
}

OUTPUT_DEFAULTS = {
    "teams_png": True,
    "team_order": "shuffled",
}

# Defaults for the optional ``groups:`` block (consumed by make_groups).
GROUPS_DEFAULTS = {
    "count": 3,
    "servers": ["London", "Frankfurt", "Frankfurt"],
    "sizes": "auto",
    "region_server_cost": {"Frankfurt": {"NA": 10}, "London": {"MENA": 2}},
    "balance_weight": 1.0,
    "std_weight": 0.2,
    "mode": "auto",
    "manual": None,
    "pinned": {},
    "iterations": 200000,
    "restarts": 4,
    "title": "VCC Groups",
}

DEFAULTS = {
    "players_file": "players.json",
    "mode": "advanced",
    "current_season": 26,
    "current_act": 5,
    "acts_per_season": 6,
    "use_tracker": False,
    "weight_current": 0.7,
    "weight_peak": 0.3,
    "weight_current_tracker": 0.2,
    "weight_peak_tracker": 0.1,
    "use_peak_act": True,
    "peak_act_max_episode": 8,
    "peak_act_max_act": 2,
    "peak_act_decay_rate": 0.9,
    "weight_peak_act": 0.15,
    "use_role_balancing": True,
    "use_region_debuff": False,
    "non_eu_debuff": 0.9,
    "use_new_player_debuff": True,
    "new_player_debuff": 0.95,
    "use_returning_player_stats": True,
    "recent_data_ranked_weight": 0.65,
    "recent_data_previous_weight": 0.35,
    "older_data_ranked_weight": 0.75,
    "older_data_previous_weight": 0.25,
    "latest_season_available": "auto",
    "use_ping_adjustment": True,
    "use_region_ping_estimates": True,
    "region_ping_estimates": {
        "EU": 30,
        "MENA": 80,
        "NA": 110,
        "ASIA": 180,
        "OCE": 200,
        "SA": 170,
    },
    "ping_breakpoints": [[70, 0.0], [80, 0.05], [110, 0.10], [150, 0.15], [200, 0.20]],
    "use_immortal_rr_granularity": True,
    "radiant_rr_threshold": 550,
    "rr_granularity_ranks": ["Immortal 3", "Radiant"],
    "rr_bonus_cap": 2.0,
    "top_tier_compression": {
        "enabled": False,
        "knee_percentile": 90,
        "slope": 0.5,
    },
    "rank_values": DEFAULT_RANK_VALUES,
    "random_seed": None,
    "optimizer": OPTIMIZER_DEFAULTS,
    "output": OUTPUT_DEFAULTS,
    "groups": None,
    "season_rating_distributions": None,
}

# Keys that are still accepted for compatibility but no longer do anything.
DEPRECATED_KEYS = {
    "use_rating_scaled_debuffs": "removed (no code read it)",
    "debuff_min_scale": "removed (no code read it)",
    "debuff_max_scale": "removed (no code read it)",
    "debuff_midpoint_rating": "removed (no code read it)",
    "debuff_scaling_steepness": "removed (no code read it)",
    "debuff_max_penalty": "removed (no code read it)",
    "use_top_tier_compression": "replaced by top_tier_compression.enabled",
    "top_tier_gap_compression": "replaced by top_tier_compression.slope",
    "top_tier_score_reduction": "replaced by top_tier_compression.knee_percentile",
    "compression_full_strength_ranks": "replaced by top_tier_compression",
    "compression_taper_ranks": "replaced by top_tier_compression",
}

# Legacy optimizer keys: old key -> (new dotted key, factor) or None if ignored.
LEGACY_OPTIMIZER_KEYS = {
    "role_balance_weight": ("optimizer.role_weight", 0.1),
    "max_time": ("optimizer.time_limit_s", 1),
    "early_termination_threshold": ("optimizer.target_range", 1),
    "max_restarts": ("optimizer.restarts", 1),
    "annealing_iterations": None,
    "initial_temperature": None,
    "cooling_rate": None,
    "max_no_improvement": None,
    "use_adaptive_cooling": None,
    "use_tabu_search": None,
    "tabu_tenure": None,
    "restart_threshold": None,
}

# Keys the loader consumes itself.
_META_KEYS = {"extends"}


def deep_merge(base, override):
    """Recursively merge two dicts without mutating either.

    Dicts are merged key by key; every other value (lists included) in
    ``override`` replaces the one in ``base``.

    Args:
        base: Base dictionary.
        override: Dictionary whose values win.

    Returns:
        A new merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path):
    """Read a YAML file as UTF-8 and return its mapping (empty if blank).

    Args:
        path: Path of the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If the top level of the file is not a mapping.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


def _load_chain(path, stack, sources):
    """Load ``path`` and everything it extends, returning the merged dict.

    Args:
        path: Config file to load.
        stack: Absolute paths currently being loaded (cycle detection).
        sources: List that receives absolute paths, base-most file first.

    Returns:
        Merged dictionary (without the ``extends`` key).
    """
    abs_path = os.path.abspath(path)
    if abs_path in stack:
        chain = " -> ".join(stack + [abs_path])
        raise ValueError(f"Config 'extends' cycle detected: {chain}")
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")

    raw = load_yaml(abs_path)
    parent = raw.get("extends")
    merged = {}
    if parent:
        parent_path = parent
        if not os.path.isabs(parent_path):
            parent_path = os.path.join(os.path.dirname(abs_path), parent_path)
        merged = _load_chain(parent_path, stack + [abs_path], sources)
    sources.append(abs_path)
    own = {k: v for k, v in raw.items() if k not in _META_KEYS}
    return deep_merge(merged, own)


def _warn(config, message):
    """Record a config warning and emit it via the warnings module."""
    config["_warnings"].append(message)
    warnings.warn(message, UserWarning, stacklevel=3)


def _apply_legacy_and_check_keys(config, user_keys):
    """Map legacy keys, then warn about deprecated and unknown keys.

    Args:
        config: Merged config (modified in place).
        user_keys: Keys that came from the user's files (not DEFAULTS).
    """
    optimizer = config.get("optimizer")
    for key in sorted(user_keys):
        if key in LEGACY_OPTIMIZER_KEYS:
            target = LEGACY_OPTIMIZER_KEYS[key]
            if target is None:
                _warn(
                    config,
                    f"Config key '{key}' is a legacy annealing setting: ignored.",
                )
                continue
            dotted, factor = target
            section, name = dotted.split(".")
            if isinstance(optimizer, dict) and name in config["_user_optimizer_keys"]:
                _warn(
                    config,
                    f"Config key '{key}' is deprecated; ignored because {dotted} is set.",
                )
                continue
            value = config[key] * factor if factor != 1 else config[key]
            optimizer[name] = round(value, 6) if isinstance(value, float) else value
            _warn(
                config,
                f"Config key '{key}' is deprecated; mapped to {dotted} = {optimizer[name]}.",
            )
        elif key in DEPRECATED_KEYS:
            _warn(config, f"Config key '{key}' is deprecated: {DEPRECATED_KEYS[key]}.")
        elif key not in DEFAULTS and not key.startswith("_"):
            _warn(config, f"Unknown config key '{key}' (ignored).")

    config.pop("use_top_tier_compression", None)

    for section, defaults in (
        ("optimizer", OPTIMIZER_DEFAULTS),
        ("output", OUTPUT_DEFAULTS),
        ("groups", GROUPS_DEFAULTS),
        ("top_tier_compression", DEFAULTS["top_tier_compression"]),
    ):
        value = config.get(section)
        if isinstance(value, dict):
            for sub in value:
                if sub not in defaults:
                    _warn(config, f"Unknown config key '{section}.{sub}' (ignored).")


def load_team_config(path=None, config_dir=None):
    """Load the team maker configuration.

    Args:
        path: Explicit config file. When omitted, ``config.yaml`` in the config
            directory is used, falling back to ``config.example.yaml``.
        config_dir: Directory holding the config files (defaults to
            ``teamMaker/config``).

    Returns:
        Merged configuration dictionary. ``_sources`` lists the files that
        were merged (base-most first) and ``_warnings`` the warnings raised.

    Raises:
        FileNotFoundError: If no config file (or extended file) exists.
        ValueError: If an ``extends`` cycle is found.
    """
    config_dir = config_dir or CONFIG_DIR
    fallback_used = False
    if path is None:
        path = os.path.join(config_dir, LOCAL_CONFIG_NAME)
        if not os.path.isfile(path):
            path = os.path.join(config_dir, EXAMPLE_CONFIG_NAME)
            fallback_used = True

    sources = []
    user = _load_chain(path, [], sources)

    config = deep_merge(DEFAULTS, user)
    config["_sources"] = sources
    config["_warnings"] = []
    config["_user_optimizer_keys"] = set(
        (user.get("optimizer") or {}).keys()
        if isinstance(user.get("optimizer"), dict)
        else ()
    )
    if fallback_used:
        print(f"No {LOCAL_CONFIG_NAME} found; using {EXAMPLE_CONFIG_NAME}.")

    # Groups block: fill in defaults only when the section is configured.
    if isinstance(user.get("groups"), dict):
        config["groups"] = deep_merge(GROUPS_DEFAULTS, user["groups"])
    elif user.get("groups") is None:
        config["groups"] = None

    _apply_legacy_and_check_keys(config, set(user.keys()))
    del config["_user_optimizer_keys"]
    return config


def resolve_seed(config):
    """Return the integer seed, drawing and recording one if it is null.

    Args:
        config: Configuration dictionary (``random_seed`` is updated in place
            when a seed had to be drawn; ``_seed_generated`` is set).

    Returns:
        The seed as an int.
    """
    seed = config.get("random_seed")
    if seed is None:
        seed = random.SystemRandom().randrange(1, 1_000_000)
        config["random_seed"] = seed
        config["_seed_generated"] = True
        print(f"random_seed not set; using generated seed {seed}")
    else:
        seed = int(seed)
        config["random_seed"] = seed
        config["_seed_generated"] = False
    return seed
