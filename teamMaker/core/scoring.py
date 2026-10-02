"""Player scoring: one number per player from rank, peak, tracker scores,
VCC history and ping. See compute_player_score_detailed for the steps."""

import math
import os
import json
import time
import re
import warnings

from teamMaker.core.config import DEFAULTS, load_team_config

__all__ = [
    "compute_player_score",
    "compute_player_score_detailed",
    "calculate_previous_season_score",
    "parse_peak_act",
    "calculate_peak_act_weight",
    "rank_to_numeric",
    "rank_to_numeric_with_rr",
    "calculate_ping_adjustment",
    "apply_top_tier_compression",
    "score_players",
    "normalize_season",
    "REGION_ALIASES",
]

# Season distributions loaded from archive/season_distributions.json.
_SEASON_DISTRIBUTIONS_CACHE = None

# Region codes seen in player data that differ from the config keys.
REGION_ALIASES = {"ME": "MENA"}

# Seasons we already warned about (one warning per season per process).
_WARNED_SEASONS = set()

_SEASON_RE = re.compile(r"^[SV]?(\d+)$")


def normalize_region(region):
    """Uppercase/strip a region code and apply REGION_ALIASES."""
    region = str(region or "EU").upper().strip()
    return REGION_ALIASES.get(region, region)


def normalize_season(season):
    """Normalise a season label to the "S<number>" form.

    Args:
        season: Season label such as "S14", "s14", "V14" or "14" (or an int).

    Returns:
        Normalised label (e.g. "S14"); other strings come back uppercased.
    """
    text = str(season if season is not None else "").upper().strip()
    match = _SEASON_RE.match(text)
    if match:
        return f"S{int(match.group(1))}"
    return text


def season_number(season):
    """Return the numeric part of a season label, or 0 if it has none."""
    match = _SEASON_RE.match(normalize_season(season))
    return int(match.group(1)) if match else 0


def _reset_caches():
    """Clear module-level caches (used by tests)."""
    global _SEASON_DISTRIBUTIONS_CACHE
    _SEASON_DISTRIBUTIONS_CACHE = None
    _WARNED_SEASONS.clear()


def canonical_rank(rank_str, rank_values):
    """Return the rank_values key for a rank string, or None if unknown.

    Case and extra spaces are ignored ("diamond  1" -> "Diamond 1"), and a
    bare tier from signup CSVs ("Gold") maps to the middle division.
    """
    text = " ".join(str(rank_str or "").split()).lower()
    if not text:
        return None
    lookup = {k.lower(): k for k in rank_values}
    return lookup.get(text) or lookup.get(f"{text} 2")


def rank_to_numeric(rank_str, rank_values):
    """Convert rank string to numeric value.

    Args:
        rank_str: Rank string (e.g., "Diamond 2"; case-insensitive)
        rank_values: Dict mapping rank strings to numeric values

    Returns:
        Numeric value for the rank, or 0 if not found
    """
    rank = canonical_rank(rank_str, rank_values)
    if rank is None:
        if rank_str and str(rank_str).strip():
            warnings.warn(f"Unknown rank encountered: '{rank_str}'", UserWarning)
        return 0
    return rank_values[rank]


def rank_to_numeric_with_rr(rank_str, rank_values, rr_value=None, config=None):
    """Convert rank string to numeric value with RR granularity.

    The bonus is relative to the configured ``rank_values`` so retuning a
    rank (for example Immortal 3 = 27) moves the whole scale with it.
    - Ranks in ``rr_granularity_ranks`` (default Immortal 3, Radiant) get RR.
    - Radiant: base + max(0, (RR - radiant_rr_threshold) / 100).
    - Every other listed rank: base + min(rr_bonus_cap, RR / 100).

    Args:
        rank_str: Rank string (e.g., "Immortal 3")
        rank_values: Dict mapping rank strings to numeric values
        rr_value: Optional RR value for granular scaling
        config: Optional config dict (rr_granularity_ranks, radiant_rr_threshold,
            rr_bonus_cap)

    Returns:
        Numeric value, with RR-based scaling for the configured ranks
    """
    config = config or {}
    base_value = rank_to_numeric(rank_str, rank_values)
    rank_str = canonical_rank(rank_str, rank_values)

    if rr_value is None or rr_value < 0:
        return base_value
    if rank_str not in config.get("rr_granularity_ranks", DEFAULTS["rr_granularity_ranks"]):
        return base_value

    if rank_str == "Radiant":
        threshold = config.get("radiant_rr_threshold", DEFAULTS["radiant_rr_threshold"])
        return base_value + max(0.0, (rr_value - threshold) / 100.0)
    return base_value + min(config.get("rr_bonus_cap", DEFAULTS["rr_bonus_cap"]), rr_value / 100.0)


def parse_peak_act(peak_act_str, current_season=26, current_act=5, acts_per_season=6):
    """Parse peak act string and calculate acts ago.

    Episode acts (E<ep>A<act>) have 3 acts per episode; season acts
    (S<season>A<act>, or the V alias) have ``acts_per_season`` acts.

    Args:
        peak_act_str: Peak act string (e.g., "E8A2", "S25A3", "V26A4")
        current_season: Current season number (e.g., 26)
        current_act: Current act number within the season
        acts_per_season: Number of acts in a season (6 since season 25)

    Returns:
        Number of acts ago the peak was achieved, or 0 if invalid
    """
    if not peak_act_str:
        return 0

    s = peak_act_str.upper().strip()

    # Episode 9 Act 3 was the last episode format (transition point)
    EPISODE_TO_SEASON_TRANSITION_EP = 9
    EPISODE_TO_SEASON_TRANSITION_ACT = 3
    FIRST_SEASON_NUMBER = 25

    current_season_acts = (current_season - FIRST_SEASON_NUMBER) * acts_per_season + current_act

    try:
        if s.startswith("E"):
            parts = s[1:].split("A")
            if len(parts) == 2:
                episode = int(parts[0])
                act = int(parts[1])
                if (
                    episode < 1
                    or episode > EPISODE_TO_SEASON_TRANSITION_EP
                    or act < 1
                    or act > 3
                ):
                    return 0

                peak_total_acts = (episode - 1) * 3 + act
                transition_total_acts = (
                    EPISODE_TO_SEASON_TRANSITION_EP - 1
                ) * 3 + EPISODE_TO_SEASON_TRANSITION_ACT
                current_total_acts = transition_total_acts + current_season_acts
                return max(0, current_total_acts - peak_total_acts)

        elif s[:1] in ("S", "V"):
            parts = s[1:].split("A")
            if len(parts) == 2:
                season = int(parts[0])
                act = int(parts[1])
                if season < FIRST_SEASON_NUMBER or act < 1 or act > acts_per_season:
                    return 0
                peak_total_season_acts = (
                    season - FIRST_SEASON_NUMBER
                ) * acts_per_season + act
                return max(0, current_season_acts - peak_total_season_acts)

    except (ValueError, IndexError):
        pass
    return 0


def calculate_peak_act_weight(acts_ago, decay_rate):
    """Calculate weight for peak act based on time decay.

    Args:
        acts_ago: Number of acts since peak
        decay_rate: Decay rate per act (e.g., 0.9)

    Returns:
        Share of the peak's lead over the current rank that still counts
    """
    return decay_rate**acts_ago


def _get_season_distributions(config):
    """Get season rating distributions (the archive file is read once).

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary mapping season names to sorted rating distributions
    """
    global _SEASON_DISTRIBUTIONS_CACHE

    # Return cached value if available
    if _SEASON_DISTRIBUTIONS_CACHE is not None:
        return _SEASON_DISTRIBUTIONS_CACHE

    # Try to get from config first
    dist = config.get("season_rating_distributions")
    if dist:
        # Config-supplied distributions are not cached (they may differ per call).
        return {
            normalize_season(k): sorted(v)
            for k, v in dist.items()
            if isinstance(v, list)
        }

    # Load from file and cache
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "archive", "season_distributions.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _SEASON_DISTRIBUTIONS_CACHE = {
                normalize_season(k): sorted(v)
                for k, v in raw.items()
                if isinstance(v, list)
            }
            return _SEASON_DISTRIBUTIONS_CACHE
        except Exception as e:
            print("Could not load archive/season_distributions.json:", e)

    _SEASON_DISTRIBUTIONS_CACHE = {}
    return _SEASON_DISTRIBUTIONS_CACHE


def calculate_ping_adjustment(ping, config):
    """Calculate the ping penalty.

    ``ping_breakpoints`` is a list of [ping_ms, penalty_fraction]; the penalty
    is interpolated linearly between them and flat outside them (default:
    none up to 70 ms, 5% at 80, 10% at 110, 15% at 150, 20% from 200).

    Args:
        ping: Ping in milliseconds (can be None)
        config: Configuration dictionary

    Returns:
        Dictionary with adjustment details including multiplier and breakdown
    """
    breakpoints = config.get("ping_breakpoints", DEFAULTS["ping_breakpoints"])

    result = {
        "enabled": config.get("use_ping_adjustment", DEFAULTS["use_ping_adjustment"]),
        "ping": ping,
        "multiplier": 1.0,
        "penalty_percent": 0.0,
        "source": "unknown",
    }

    if not result["enabled"]:
        return result

    if ping is None:
        result["source"] = "no_data"
        result["multiplier"] = 1.0
        return result

    result["source"] = "actual_ping"

    # Handle values below minimum breakpoint
    if ping <= breakpoints[0][0]:
        penalty = breakpoints[0][1]
        result["penalty_percent"] = penalty * 100
        result["multiplier"] = 1.0 - penalty
        return result

    # Handle values above maximum breakpoint
    if ping >= breakpoints[-1][0]:
        penalty = breakpoints[-1][1]
        result["penalty_percent"] = penalty * 100
        result["multiplier"] = 1.0 - penalty
        return result

    # Piecewise linear interpolation between breakpoints
    for i in range(len(breakpoints) - 1):
        x1, y1 = breakpoints[i]
        x2, y2 = breakpoints[i + 1]

        if x1 <= ping <= x2:
            # Linear interpolation: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            penalty = y1 + (y2 - y1) * (ping - x1) / (x2 - x1)
            result["penalty_percent"] = penalty * 100
            result["multiplier"] = 1.0 - penalty
            result["interpolation"] = f"Between {x1}ms and {x2}ms"
            return result

    # Fallback (should not reach here)
    result["multiplier"] = 1.0
    return result


def _latest_season_available(config):
    """Return the latest season number that has usable data.

    ``latest_season_available: auto`` uses the newest season that has a
    rating distribution.

    Args:
        config: Configuration dictionary

    Returns:
        Season number as an int
    """
    value = config.get("latest_season_available", DEFAULTS["latest_season_available"])
    if isinstance(value, str) and value.strip().lower() == "auto":
        seasons = [season_number(k) for k in _get_season_distributions(config)]
        return max(seasons) if seasons else 0
    return season_number(value)


def calculate_previous_season_score(player_info, config):
    """Calculate score contribution from previous season stats.

    Args:
        player_info: Player information dictionary
        config: Configuration dictionary

    Returns:
        Previous season score component (0 if not applicable)
    """
    if not config.get("use_returning_player_stats", DEFAULTS["use_returning_player_stats"]):
        return 0.0
    if not player_info.get("is_returning_player", False):
        return 0.0

    prev_stats = player_info.get("previous_season_stats", {})
    if not prev_stats:
        return 0.0

    distributions = _get_season_distributions(config)

    def is_known(season):
        """Check the season has a distribution, warning once if it does not."""
        if season in distributions:
            return True
        if season and season not in _WARNED_SEASONS:
            _WARNED_SEASONS.add(season)
            warnings.warn(
                f"No rating distribution for season '{season}'; its stats are skipped "
                "(run scrape_distributions to add it).",
                UserWarning,
            )
        return False

    def convert_zscore_to_score(rating, distribution):
        """Map an adjusted rating to 1-35 points by its z-score in its season.

        Comparing within each season keeps seasons comparable when the
        rating formula changed. Too few values or no spread give 15 points.
        """
        MIN_SAMPLE_SIZE = 10
        MIN_STD_DEV = 0.001

        if not distribution or len(distribution) < MIN_SAMPLE_SIZE:
            return 15.0

        # Calculate mean and standard deviation
        mean = sum(distribution) / len(distribution)
        variance = sum((x - mean) ** 2 for x in distribution) / len(distribution)
        std_dev = math.sqrt(variance)

        if std_dev < MIN_STD_DEV:
            return 15.0

        # Calculate z-score
        z_score = (rating - mean) / std_dev

        # Map z-score to score range (1-35 points)
        # Center around 18, with ~8 point spread per std dev
        base_score = 18.0
        score_per_std = 8.0

        score = base_score + (z_score * score_per_std)

        # Clamp to reasonable range
        return max(1.0, min(35.0, score))

    if isinstance(prev_stats, list):
        season_scores = {}
        for entry in prev_stats:
            season = normalize_season(entry.get("season", ""))
            rating = float(entry.get("adjusted_rating", 0.0))
            if season and rating > 0 and is_known(season):
                season_scores[season] = convert_zscore_to_score(
                    rating, distributions[season]
                )
        if not season_scores:
            return 0.0
        if len(season_scores) == 1:
            return next(iter(season_scores.values()))

        # Weight seasons by recency.
        def calculate_decay_weights(seasons):
            weights = {}
            # Sort seasons by number (S10, S9, S8, etc.)
            sorted_seasons = sorted(
                seasons,
                key=season_number,
                reverse=True,
            )

            # Decay weights: Most recent 60%, next 30%, rest split remaining 10%
            if len(sorted_seasons) >= 1:
                weights[sorted_seasons[0]] = 0.60  # Most recent season
            if len(sorted_seasons) >= 2:
                weights[sorted_seasons[1]] = 0.30  # Second most recent
            if len(sorted_seasons) >= 3:
                # Remaining seasons split the remaining 10%
                remaining_weight = 0.10
                remaining_seasons = sorted_seasons[2:]
                weight_per_season = remaining_weight / len(remaining_seasons)
                for season in remaining_seasons:
                    weights[season] = weight_per_season

            return weights

        weights = calculate_decay_weights(season_scores.keys())
        return sum(season_scores[s] * weights[s] for s in weights)
    else:
        season = normalize_season(prev_stats.get("season", ""))
        rating = float(prev_stats.get("adjusted_rating", 0.0))
        if not season or rating <= 0 or not is_known(season):
            return 0.0
        return convert_zscore_to_score(rating, distributions[season])


def compute_player_score(player_info, config):
    """Compute player score (simplified interface).

    Args:
        player_info: Dictionary containing player data (ranks, stats, etc.)
        config: Configuration dictionary with weights and settings

    Returns:
        Final score as a float
    """
    return compute_player_score_detailed(player_info, config)["final_score"]


def compute_player_score_detailed(player_info, config):
    """Compute player score with detailed breakdown.

    Steps, each switchable in the config:
    - current and peak rank values (RR on top for Immortal 3 / Radiant), with
      older peaks fading toward the current rank
    - tracker scores
    - VCC history blend for returning players
    - ping penalty
    - new player discount

    Args:
        player_info: Dictionary containing player data including:
            - current_rank: String (e.g., "Diamond 2")
            - peak_rank: String
            - current_rr: Int (optional, for Immortal 3+)
            - peak_rr: Int (optional, for Immortal 3+)
            - tracker_current: Float (optional)
            - tracker_peak: Float (optional)
            - peak_rank_act: String (optional, e.g., "E8A2")
            - region: String (optional, e.g., "EU")
            - ping: Int (optional, actual ping)
            - is_returning_player: Bool
            - previous_season_stats: Dict or List
        config: Configuration dictionary with all settings

    Returns:
        Dictionary with detailed breakdown of score components
    """
    rank_values = config.get("rank_values", {})

    # Extract RR values for Immortal 3+ granularity
    current_rr = player_info.get("current_rr")
    peak_rr = player_info.get("peak_rr")

    # Use RR granularity if enabled and RR available
    use_rr = config.get("use_immortal_rr_granularity", DEFAULTS["use_immortal_rr_granularity"])

    if use_rr:
        current_val = rank_to_numeric_with_rr(
            player_info.get("current_rank", ""), rank_values, current_rr, config
        )
        peak_val = rank_to_numeric_with_rr(
            player_info.get("peak_rank", ""), rank_values, peak_rr, config
        )
    else:
        current_val = rank_to_numeric(player_info.get("current_rank", ""), rank_values)
        peak_val = rank_to_numeric(player_info.get("peak_rank", ""), rank_values)

    # No peak rank entered: assume peak == current instead of 0, so players
    # with a blank peak_rank are not penalised against those with one.
    if not player_info.get("peak_rank"):
        peak_val = current_val

    # Older peaks count less: a peak above the current rank fades toward the
    # current rank by peak_act_decay_rate per act since peak_rank_act.
    peak_fade = {"enabled": False}
    if config.get("use_peak_act", DEFAULTS["use_peak_act"]) and player_info.get("peak_rank_act"):
        acts_ago = parse_peak_act(
            str(player_info["peak_rank_act"]).upper().strip(),
            config.get("current_season", DEFAULTS["current_season"]),
            config.get("current_act", DEFAULTS["current_act"]),
            config.get("acts_per_season", DEFAULTS["acts_per_season"]),
        )
        weight = calculate_peak_act_weight(
            acts_ago, config.get("peak_act_decay_rate", DEFAULTS["peak_act_decay_rate"])
        )
        raw_peak = peak_val
        if peak_val > current_val:
            peak_val = current_val + (peak_val - current_val) * weight
        peak_fade = {
            "enabled": True,
            "peak_rank_act": player_info["peak_rank_act"],
            "acts_ago": acts_ago,
            "weight": weight,
            "peak_value_before_fade": raw_peak,
            "peak_value_after_fade": peak_val,
        }

    breakdown = {
        "rank_components": {
            "current_rank": player_info.get("current_rank"),
            "current_rank_value": current_val,
            "current_rank_weighted": config.get("weight_current", DEFAULTS["weight_current"]) * current_val,
            "peak_rank": player_info.get("peak_rank"),
            "peak_rank_value": peak_val,
            "peak_rank_weighted": config.get("weight_peak", DEFAULTS["weight_peak"]) * peak_val,
        },
        "tracker_components": {},
        "advanced_components": {},
        "final_score": 0.0,
    }

    # Add RR details if used
    if use_rr:
        breakdown["rank_components"]["rr_granularity_used"] = True
        breakdown["rank_components"]["current_rr"] = current_rr
        breakdown["rank_components"]["peak_rr"] = peak_rr

    current_score = (
            config.get("weight_current", DEFAULTS["weight_current"]) * current_val
            + config.get("weight_peak", DEFAULTS["weight_peak"]) * peak_val
    )
    breakdown["base_score"] = current_score

    # Tracker (both modes) if enabled
    if config.get("use_tracker", DEFAULTS["use_tracker"]):
        cur_tracker = player_info.get("tracker_current")
        peak_tracker = player_info.get("tracker_peak")
        if cur_tracker is not None and peak_tracker is not None:
            cur_score = config.get("weight_current_tracker", DEFAULTS["weight_current_tracker"]) * math.sqrt(
                max(0, cur_tracker)
            )
            peak_score = config.get("weight_peak_tracker", DEFAULTS["weight_peak_tracker"]) * math.log(
                1 + max(0, peak_tracker)
            )
            consistency_factor = 1.0
            if peak_val > 0:
                consistency_factor += 0.1 * (current_val / peak_val)
            current_score = (
                                    current_score + cur_score + peak_score
                            ) * consistency_factor
            breakdown["tracker_components"] = {
                "enabled": True,
                "current_tracker": cur_tracker,
                "current_tracker_score": cur_score,
                "peak_tracker": peak_tracker,
                "peak_tracker_score": peak_score,
                "consistency_factor": consistency_factor,
                "tracker_total": cur_score + peak_score,
            }
        else:
            breakdown["tracker_components"] = {"enabled": False}
    else:
        breakdown["tracker_components"] = {"enabled": False}

    adv = {"peak_act": peak_fade}

    # VCC history: blended in for returning players, more when it is recent.
    if config.get("use_returning_player_stats", DEFAULTS["use_returning_player_stats"]) and player_info.get(
            "is_returning_player", False
    ):
        prev_score = calculate_previous_season_score(player_info, config)
        if prev_score > 0:
            pre_blend = current_score
            prev_stats = player_info.get("previous_season_stats", {})
            if isinstance(prev_stats, list):
                seasons = [season_number(e.get("season", "")) for e in prev_stats]
                most_recent_season = max(seasons) if seasons else 0
            else:
                most_recent_season = season_number(prev_stats.get("season", ""))

            # Played in the newest archived season: trust the history more.
            if most_recent_season >= _latest_season_available(config):
                ranked_weight = config.get("recent_data_ranked_weight", DEFAULTS["recent_data_ranked_weight"])
                previous_weight = config.get("recent_data_previous_weight", DEFAULTS["recent_data_previous_weight"])
                reason = f"Recent data (S{most_recent_season}) - standard blend ({previous_weight:.0%} historical)"
            else:
                ranked_weight = config.get("older_data_ranked_weight", DEFAULTS["older_data_ranked_weight"])
                previous_weight = config.get("older_data_previous_weight", DEFAULTS["older_data_previous_weight"])
                reason = f"Older data (S{most_recent_season}) - reduced historical weight ({previous_weight:.0%} historical)"

            current_score = current_score * ranked_weight + prev_score * previous_weight
            adv["previous_season"] = {
                "enabled": True,
                "previous_season_score": prev_score,
                "pre_blend_score": pre_blend,
                "post_blend_score": current_score,
                "blend_reason": reason,
                "most_recent_season": most_recent_season,
                "ranked_weight": ranked_weight,
                "previous_weight": previous_weight,
            }
        else:
            adv["previous_season"] = {"enabled": True, "previous_season_score": 0.0}
    else:
        adv["previous_season"] = {"enabled": False}

    # Ping: the actual ping, or an estimate for the player's region.
    if config.get("use_ping_adjustment", DEFAULTS["use_ping_adjustment"]):
        ping = player_info.get("ping")
        region = normalize_region(player_info.get("region", "EU"))
        if ping is None and config.get("use_region_ping_estimates", DEFAULTS["use_region_ping_estimates"]):
            region_estimates = {
                normalize_region(k): v
                for k, v in config.get(
                    "region_ping_estimates", DEFAULTS["region_ping_estimates"]
                ).items()
            }
            ping = region_estimates.get(region, None)
            ping_source = "region_estimate"
        else:
            ping_source = "actual" if ping is not None else "unavailable"

        ping_adj = calculate_ping_adjustment(ping, config)
        ping_adj["region"] = region
        ping_adj["ping_source"] = ping_source
        if ping_adj["multiplier"] < 1.0:
            pre = current_score
            current_score *= ping_adj["multiplier"]
            ping_adj["pre_adjustment_score"] = pre
            ping_adj["post_adjustment_score"] = current_score
        adv["ping_adjustment"] = ping_adj
    else:
        adv["ping_adjustment"] = {"enabled": False}

    # New players (no VCC history): small discount for the uncertainty.
    if config.get("use_new_player_debuff", DEFAULTS["use_new_player_debuff"]):
        if not player_info.get("is_returning_player", False):
            pre = current_score
            mult = config.get("new_player_debuff", DEFAULTS["new_player_debuff"])
            current_score *= mult
            adv["new_player_debuff"] = {
                "enabled": True,
                "is_returning_player": False,
                "debuff_multiplier": mult,
                "pre_debuff_score": pre,
                "post_debuff_score": current_score,
                "reason": "New player uncertainty - ranked data only",
            }
        else:
            adv["new_player_debuff"] = {
                "enabled": True,
                "is_returning_player": True,
                "debuff_applied": False,
            }
    else:
        adv["new_player_debuff"] = {"enabled": False}

    breakdown["advanced_components"] = adv
    breakdown["final_score"] = current_score
    return breakdown


def apply_top_tier_compression(player_scores_dict, config):
    """Compress the top of the score distribution with a monotone soft knee.

    Scores at or below the knee (the ``knee_percentile`` of all scores) are
    unchanged; the part above the knee is scaled by ``slope`` (0-1). A score
    is never raised and the ordering of players is preserved.

    Args:
        player_scores_dict: Dict of {player_name: final_score}
        config: Configuration dictionary (``top_tier_compression`` section)

    Returns:
        Dict of {player_name: compressed_score}
    """
    settings = config.get("top_tier_compression") or {}
    percentile = float(settings.get("knee_percentile", 90))
    slope = min(1.0, max(0.0, float(settings.get("slope", 0.5))))

    if not player_scores_dict:
        return {}
    ordered = sorted(player_scores_dict.values())
    # Linear-interpolated percentile of the scores.
    pos = (len(ordered) - 1) * min(100.0, max(0.0, percentile)) / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    knee = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {
        name: score if score <= knee else knee + slope * (score - knee)
        for name, score in player_scores_dict.items()
    }


def _load_players(players_path):
    """Load player data from JSON file."""
    if not os.path.isfile(players_path):
        raise FileNotFoundError("Players file not found: " + players_path)
    with open(players_path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_players(players, config):
    """Score every player, applying top-tier compression when enabled.

    Args:
        players: Dict of {player_name: player_info}
        config: Configuration dictionary

    Returns:
        Tuple (scores, breakdowns): scores maps name to the score rounded to
        1 decimal (input order); breakdowns maps name to the detailed
        breakdown, whose ``final_score`` is the unrounded value.
    """
    breakdowns = {
        name: compute_player_score_detailed(info, config)
        for name, info in players.items()
    }
    raw = {name: bd["final_score"] for name, bd in breakdowns.items()}

    compression = config.get("top_tier_compression") or {}
    if compression.get("enabled", False):
        compressed = apply_top_tier_compression(raw, config)
        for name, value in compressed.items():
            breakdowns[name]["original_final_score"] = breakdowns[name]["final_score"]
            breakdowns[name]["compressed_final_score"] = value
            breakdowns[name]["final_score"] = value
        raw = compressed

    scores = {name: round(value, 1) for name, value in raw.items()}
    return scores, breakdowns


def _export_scores(players_data, config, out_detailed, out_minimal):
    """Export detailed and minimal score files."""
    minimal, detailed = score_players(players_data, config)

    # Sort both by final score
    detailed = dict(
        sorted(detailed.items(), key=lambda kv: kv[1]["final_score"], reverse=True)
    )
    minimal = dict(sorted(minimal.items(), key=lambda kv: kv[1], reverse=True))

    with open(out_detailed, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
    with open(out_minimal, "w", encoding="utf-8") as f:
        json.dump(minimal, f, indent=2)
    return detailed, minimal


def _print_summary(minimal_scores, top=10):
    """Print top player scores."""
    items = list(minimal_scores.items())
    print("\nTop players:")
    for i, (name, score) in enumerate(items[:top], 1):
        print(f" {i:2d}. {name:25s} {score:.2f}")
    if len(items) > top:
        print(f" ... and {len(items)-top} more")


def main():
    """Main entry point for score calculation."""
    start = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        config = load_team_config()
    except (OSError, ValueError) as e:
        print("Failed to load config:", e)
        return
    players_file = config.get("players_file", DEFAULTS["players_file"])
    if not os.path.isabs(players_file):
        players_path = os.path.join(base_dir, "data", players_file)
    else:
        players_path = players_file
    print("Loading players from:", players_path)
    try:
        players_data = _load_players(players_path)
    except Exception as e:
        print("Failed to load players:", e)
        return
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_detailed = os.path.join(out_dir, "player_scores.json")
    out_minimal = os.path.join(out_dir, "player_scores_minimal.json")
    detailed, minimal = _export_scores(players_data, config, out_detailed, out_minimal)
    _print_summary(minimal)
    print(f"\nExported {len(minimal)} player scores.")
    print("Detailed ->", out_detailed)
    print("Minimal  ->", out_minimal)
    print(f"Done in {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()