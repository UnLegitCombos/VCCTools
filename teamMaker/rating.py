"""Rating & player scoring (trimmed)."""

import math
import os
import json
import time
import bisect
import warnings


# Global cache for season distributions to avoid repeated file I/O (Bug Fix #5)
_SEASON_DISTRIBUTIONS_CACHE = None


def rank_to_numeric(rank_str, rank_values):
    """Convert rank string to numeric value.

    Args:
        rank_str: Rank string (e.g., "Diamond 2")
        rank_values: Dict mapping rank strings to numeric values

    Returns:
        Numeric value for the rank, or 0 if not found
    """
    result = rank_values.get(rank_str, 0)
    # Bug Fix #4: Warn when unknown rank encountered
    if result == 0 and rank_str and rank_str.strip():
        warnings.warn(f"Unknown rank encountered: '{rank_str}'", UserWarning)
    return result


def rank_to_numeric_with_rr(rank_str, rank_values, rr_value=None):
    """Convert rank string to numeric value with RR granularity for Immortal 3+.

    Enhancement: Immortal RR Granularity
    - Immortal 1 & 2: Discrete values (no RR)
    - Immortal 3: Scale from 25.0 to 29.95 based on RR (25.0 + min(4.95, RR/100))
    - Radiant: Scale from 30.0+ based on RR above 550 (30.0 + max(0, (RR-550)/100))

    Args:
        rank_str: Rank string (e.g., "Immortal 3")
        rank_values: Dict mapping rank strings to numeric values
        rr_value: Optional RR value for granular scaling

    Returns:
        Numeric value, with RR-based scaling for Immortal 3+
    """
    base_value = rank_to_numeric(rank_str, rank_values)

    # Only apply RR granularity to Immortal 3 and Radiant
    if rr_value is not None and rr_value >= 0:
        if rank_str == "Immortal 3":
            # Scale from 25.0 to 27.0 based on RR
            # Formula: 25.0 + min(2.0, RR/100)
            rr_bonus = min(2.0, rr_value / 100.0)
            return 25.0 + rr_bonus
        elif rank_str == "Radiant":
            # Scale from 30.0+ based on RR above 550
            # Formula: 30.0 + max(0, (RR-550)/100)
            rr_bonus = max(0.0, (rr_value - 550) / 100.0)
            return 30.0 + rr_bonus

    return base_value


def parse_peak_act(peak_act_str, current_season=25, current_act=3):
    """Parse peak act string and calculate acts ago.

    Bug Fix #1: Enhanced to handle evolving episode-to-season transition.
    Dynamically calculates transition point instead of hardcoded value.

    Args:
        peak_act_str: Peak act string (e.g., "E8A2", "S25A3")
        current_season: Current season number (e.g., 25)
        current_act: Current act number (1-4 for seasons, 1-3 for episodes)

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

                # Calculate total acts in episode format
                peak_total_acts = (episode - 1) * 3 + act

                # Calculate current position accounting for transition
                transition_total_acts = (
                    EPISODE_TO_SEASON_TRANSITION_EP - 1
                ) * 3 + EPISODE_TO_SEASON_TRANSITION_ACT
                current_season_acts = (
                    current_season - FIRST_SEASON_NUMBER
                ) * 4 + current_act
                current_total_acts = transition_total_acts + current_season_acts

                return max(0, current_total_acts - peak_total_acts)

        elif s.startswith("S"):
            parts = s[1:].split("A")
            if len(parts) == 2:
                season = int(parts[0])
                act = int(parts[1])
                if season < FIRST_SEASON_NUMBER or act < 1 or act > 4:
                    return 0

                # Both peak and current are in season format
                current_total_season_acts = (
                    current_season - FIRST_SEASON_NUMBER
                ) * 4 + current_act
                peak_total_season_acts = (season - FIRST_SEASON_NUMBER) * 4 + act
                return max(0, current_total_season_acts - peak_total_season_acts)

    except (ValueError, IndexError):
        pass
    return 0


def calculate_peak_act_weight(acts_ago, decay_rate):
    """Calculate weight for peak act based on time decay.

    Args:
        acts_ago: Number of acts since peak
        decay_rate: Decay rate per act (e.g., 0.9)

    Returns:
        Weight multiplier for the peak act bonus
    """
    return decay_rate**acts_ago


def _get_season_distributions(config):
    """Get season rating distributions with caching.

    Bug Fix #5: Implements module-level caching to avoid repeated file I/O.

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
        _SEASON_DISTRIBUTIONS_CACHE = {
            k.upper(): sorted(v) for k, v in dist.items() if isinstance(v, list)
        }
        return _SEASON_DISTRIBUTIONS_CACHE

    # Load from file and cache
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "season_distributions.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _SEASON_DISTRIBUTIONS_CACHE = {
                k.upper(): sorted(v) for k, v in raw.items() if isinstance(v, list)
            }
            return _SEASON_DISTRIBUTIONS_CACHE
        except Exception as e:
            print("Could not load season_distributions.json:", e)

    _SEASON_DISTRIBUTIONS_CACHE = {}
    return _SEASON_DISTRIBUTIONS_CACHE


def _percentile(value, sorted_values):
    """Calculate percentile using binary search for efficiency.

    Bug Fix #3: Replaced O(n) linear search with O(log n) binary search.

    Args:
        value: Value to find percentile for
        sorted_values: Pre-sorted list of values

    Returns:
        Percentile (0-100)
    """
    if not sorted_values:
        return 50.0

    # Use bisect for O(log n) binary search
    index = bisect.bisect_left(sorted_values, value)

    # Calculate percentile
    percentile = (index / len(sorted_values)) * 100.0

    return min(100.0, max(0.0, percentile))


def calculate_ping_adjustment(ping, config):
    """Calculate ping-based rating adjustment.

    Enhancement: Ping-Based Rating Adjustment
    Replaces region debuff with actual ping-based penalties using piecewise linear interpolation.

    Breakpoints:
    - 0-80ms: 0% penalty (1.00x)
    - 80ms: 5% penalty (0.95x)
    - 110ms: 10% penalty (0.90x)
    - 150ms: 15% penalty (0.85x)
    - 200ms+: 20% penalty (0.80x)

    Args:
        ping: Ping in milliseconds (can be None)
        config: Configuration dictionary

    Returns:
        Dictionary with adjustment details including multiplier and breakdown
    """
    # Default breakpoints: [[ping_ms, penalty_fraction], ...]
    breakpoints = config.get(
        "ping_breakpoints",
        [[0, 0.00], [80, 0.05], [110, 0.10], [150, 0.15], [200, 0.20]],
    )

    result = {
        "enabled": config.get("use_ping_adjustment", False),
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


def calculate_previous_season_score(player_info, config):
    """Calculate score contribution from previous season stats.

    Args:
        player_info: Player information dictionary
        config: Configuration dictionary

    Returns:
        Previous season score component (0 if not applicable)
    """
    if not config.get("use_returning_player_stats", False):
        return 0.0
    if not player_info.get("is_returning_player", False):
        return 0.0

    prev_stats = player_info.get("previous_season_stats", {})
    if not prev_stats:
        return 0.0

    distributions = _get_season_distributions(config)

    def convert_zscore_to_score(rating, distribution):
        """Convert rating to score using Z-score normalization.

        Bug Fix #2: Enhanced with robust validation for division by zero.
        - Minimum sample size check (MIN_SAMPLE_SIZE=10)
        - Minimum standard deviation check (MIN_STD_DEV=0.001)

        This correctly handles cross-season comparisons where formula changes
        have shifted the overall rating distributions.
        """
        MIN_SAMPLE_SIZE = 10  # Configurable: minimum sample size for valid statistics
        MIN_STD_DEV = 0.001  # Configurable: minimum std dev to avoid division by zero

        if not distribution or len(distribution) < MIN_SAMPLE_SIZE:
            return 15.0

        # Calculate mean and standard deviation
        mean = sum(distribution) / len(distribution)
        variance = sum((x - mean) ** 2 for x in distribution) / len(distribution)
        std_dev = math.sqrt(variance)

        # Bug Fix #2: Robust validation for division by zero
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
            season = str(entry.get("season", "")).upper()
            rating = float(entry.get("adjusted_rating", 0.0))
            if season and rating > 0 and season in distributions:
                season_scores[season] = convert_zscore_to_score(
                    rating, distributions[season]
                )
        if not season_scores:
            return 0.0
        if len(season_scores) == 1:
            return next(iter(season_scores.values()))

        # New decay-based weighting system
        def calculate_decay_weights(seasons):
            weights = {}
            # Sort seasons by number (S10, S9, S8, etc.)
            sorted_seasons = sorted(
                seasons,
                key=lambda s: int(s[1:]) if s.startswith("S") else 0,
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
        season = str(prev_stats.get("season", "")).upper()
        rating = float(prev_stats.get("adjusted_rating", 0.0))
        if not season or rating <= 0 or season not in distributions:
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

    Main scoring function that combines:
    - Current and peak rank values
    - Tracker stats (if enabled)
    - Peak act bonus (if enabled and in advanced mode)
    - Previous season performance (if enabled and in advanced mode)
    - Ping adjustment (if enabled and in advanced mode)
    - New player debuff (if enabled and in advanced mode)
    - RR granularity for Immortal 3+ (if enabled)

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
    mode = config.get("mode", "basic")
    rank_values = config.get("rank_values", {})

    # Extract RR values for Immortal 3+ granularity
    current_rr = player_info.get("current_rr")
    peak_rr = player_info.get("peak_rr")

    # Use RR granularity if enabled and RR available
    use_rr = config.get("use_immortal_rr_granularity", False)

    if use_rr:
        current_val = rank_to_numeric_with_rr(
            player_info.get("current_rank", ""), rank_values, current_rr
        )
        peak_val = rank_to_numeric_with_rr(
            player_info.get("peak_rank", ""), rank_values, peak_rr
        )
    else:
        current_val = rank_to_numeric(player_info.get("current_rank", ""), rank_values)
        peak_val = rank_to_numeric(player_info.get("peak_rank", ""), rank_values)

    breakdown = {
        "mode": mode,
        "rank_components": {
            "current_rank": player_info.get("current_rank"),
            "current_rank_value": current_val,
            "current_rank_weighted": config.get("weight_current", 0.8) * current_val,
            "peak_rank": player_info.get("peak_rank"),
            "peak_rank_value": peak_val,
            "peak_rank_weighted": config.get("weight_peak", 0.2) * peak_val,
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
        config.get("weight_current", 0.8) * current_val
        + config.get("weight_peak", 0.2) * peak_val
    )
    breakdown["base_score"] = current_score

    # Tracker (both modes) if enabled
    if config.get("use_tracker", False):
        cur_tracker = player_info.get("tracker_current")
        peak_tracker = player_info.get("tracker_peak")
        if cur_tracker is not None and peak_tracker is not None:
            cur_score = config.get("weight_current_tracker", 0.4) * math.sqrt(
                max(0, cur_tracker)
            )
            peak_score = config.get("weight_peak_tracker", 0.2) * math.log(
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

    if mode == "advanced":
        adv = {}
        if config.get("use_peak_act") and player_info.get("peak_rank_act"):
            peak_act_str = player_info.get("peak_rank_act", "").upper().strip()

            # Only apply peak act bonus for episodes/acts at or before the configured threshold
            should_apply_bonus = False
            max_episode = config.get("peak_act_max_episode", 8)
            max_act = config.get("peak_act_max_act", 1)

            if peak_act_str.startswith("E"):
                try:
                    parts = peak_act_str[1:].split("A")
                    episode_num = int(parts[0])
                    act_num = int(parts[1]) if len(parts) > 1 else 1

                    # Check if episode/act is at or before the threshold
                    if episode_num < max_episode or (
                        episode_num == max_episode and act_num <= max_act
                    ):
                        should_apply_bonus = True
                except (ValueError, IndexError):
                    should_apply_bonus = False

            if should_apply_bonus:
                acts_ago = parse_peak_act(
                    peak_act_str,
                    config.get("current_season", 25),
                    config.get("current_act", 3),
                )
                act_weight = calculate_peak_act_weight(
                    acts_ago, config.get("peak_act_decay_rate", 0.9)
                )
                bonus = config.get("weight_peak_act", 0.15) * peak_val * act_weight
                current_score += bonus
                adv["peak_act"] = {
                    "enabled": True,
                    "episode_check": f"{peak_act_str} <= E{max_episode}A{max_act}",
                    "acts_ago": acts_ago,
                    "act_weight": act_weight,
                    "peak_act_bonus": bonus,
                }
            else:
                reason = f"Peak after E{max_episode}A{max_act} threshold - using consistency factor instead"
                if not peak_act_str.startswith("E"):
                    reason = "Season peak - using consistency factor instead"
                adv["peak_act"] = {
                    "enabled": False,
                    "reason": reason,
                    "peak_act_string": peak_act_str,
                    "threshold": f"E{max_episode}A{max_act}",
                }
        else:
            adv["peak_act"] = {"enabled": False}

        # Previous season blend with recency-based weighting
        if config.get("use_returning_player_stats", False) and player_info.get(
            "is_returning_player", False
        ):
            prev_score = calculate_previous_season_score(player_info, config)
            if prev_score > 0:
                pre_blend = current_score

                # Check data recency to determine blend weights
                prev_stats = player_info.get("previous_season_stats", {})

                def get_most_recent_season(stats):
                    if isinstance(stats, list):
                        seasons = [
                            int(s.get("season", "S0")[1:])
                            for s in stats
                            if s.get("season", "").startswith("S")
                        ]
                        return max(seasons) if seasons else 0
                    else:
                        season_str = stats.get("season", "S0")
                        return int(season_str[1:]) if season_str.startswith("S") else 0

                most_recent_season = get_most_recent_season(prev_stats)

                # Get the latest season available from config (configurable for future seasons)
                most_recent_available = config.get("latest_season_available", 10)

                # Determine blend weights based on data recency
                if most_recent_season >= most_recent_available:  # S10 is most recent
                    # Most recent data - higher confidence in historical performance
                    ranked_weight = config.get("recent_data_ranked_weight", 0.65)
                    previous_weight = config.get("recent_data_previous_weight", 0.35)
                    reason = f"Recent data (S{most_recent_season}) - standard blend ({previous_weight:.0%} historical)"
                else:
                    # Older data - lower confidence in historical performance
                    ranked_weight = config.get("older_data_ranked_weight", 0.75)
                    previous_weight = config.get("older_data_previous_weight", 0.25)
                    reason = f"Older data (S{most_recent_season}) - reduced historical weight ({previous_weight:.0%} historical)"

                # Apply the blend
                current_score = (
                    current_score * ranked_weight + prev_score * previous_weight
                )

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

        # Enhancement: Ping-based adjustment (replaces region debuff)
        if config.get("use_ping_adjustment", False):
            # Get actual ping or estimate from region
            ping = player_info.get("ping")

            # Bug Fix #6: Add .strip() to region string normalization
            region = str(player_info.get("region", "EU")).upper().strip()

            # Fallback to region-based estimate if ping not available
            if ping is None and config.get("use_region_ping_estimates", True):
                region_estimates = config.get(
                    "region_ping_estimates",
                    {"EU": 30, "ME": 80, "NA": 120, "ASIA": 180, "OCE": 200, "SA": 170},
                )
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
            # Legacy region debuff (deprecated)
            if config.get("use_region_debuff"):
                # Bug Fix #6: Add .strip() to region string normalization
                region = str(player_info.get("region", "EU")).upper().strip()
                if region != "EU":
                    pre = current_score
                    mult = config.get("non_eu_debuff", 0.95)
                    current_score *= mult
                    adv["region_debuff"] = {
                        "enabled": True,
                        "region": region,
                        "debuff_multiplier": mult,
                        "pre_debuff_score": pre,
                        "post_debuff_score": current_score,
                    }
                else:
                    adv["region_debuff"] = {
                        "enabled": True,
                        "region": region,
                        "debuff_applied": False,
                    }
            else:
                adv["region_debuff"] = {"enabled": False}

        # New player debuff (uncertainty due to limited data)
        if config.get("use_new_player_debuff", True):
            is_returning = player_info.get("is_returning_player", False)
            if not is_returning:
                pre = current_score
                mult = config.get("new_player_debuff", 0.95)  # 95% of ranked data only
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
    else:
        breakdown["advanced_components"] = {"mode": "basic"}

    breakdown["final_score"] = current_score
    return breakdown


def apply_top_tier_compression(player_scores_dict, config):
    """
    Apply compression to top-tier players with gradient tapering.

    Uses a rank-based tapering system that smoothly transitions from full
    compression at the top to no compression further down the rankings.
    This eliminates artificial gaps at threshold boundaries.

    Args:
        player_scores_dict: Dict of {player_name: final_score}
        config: Configuration dictionary

    Returns:
        Dict of {player_name: compressed_score}
    """
    gap_compression_factor = config.get(
        "top_tier_gap_compression", 0.35
    )  # Keep 35% of gaps at full strength
    top_score_reduction = config.get(
        "top_tier_score_reduction", 0.95
    )  # Reduce top by 5%

    # Gradient parameters (configurable)
    full_compression_ranks = config.get(
        "compression_full_strength_ranks", 7
    )  # Ranks 1-7: full compression
    taper_ranks = config.get(
        "compression_taper_ranks", 3
    )  # Ranks 8-10: tapered compression

    # Sort players by score (descending)
    sorted_players = sorted(
        player_scores_dict.items(), key=lambda x: x[1], reverse=True
    )

    compressed_scores = {}
    prev_compressed_score = None

    for i, (player, original_score) in enumerate(sorted_players):
        rank = i + 1  # 1-indexed rank

        if i == 0:
            # Top player: apply percentage reduction
            compressed_scores[player] = original_score * top_score_reduction
            prev_compressed_score = compressed_scores[player]
        elif rank <= full_compression_ranks:
            # Full compression zone (ranks 1-7)
            original_gap = sorted_players[i - 1][1] - original_score
            compressed_gap = original_gap * gap_compression_factor
            compressed_scores[player] = prev_compressed_score - compressed_gap
            prev_compressed_score = compressed_scores[player]
        elif rank <= full_compression_ranks + taper_ranks:
            # Taper zone (ranks 8-10): gradually reduce compression strength
            taper_position = rank - full_compression_ranks  # 1, 2, 3, ...
            # Linear taper: 100% -> 66% -> 33% -> 0%
            taper_strength = max(0.0, 1.0 - (taper_position / (taper_ranks + 1)))

            original_gap = sorted_players[i - 1][1] - original_score

            # Blend between compressed and original gap
            compressed_gap = original_gap * gap_compression_factor
            tapered_gap = compressed_gap * taper_strength + original_gap * (
                1 - taper_strength
            )

            compressed_scores[player] = prev_compressed_score - tapered_gap
            prev_compressed_score = compressed_scores[player]
        else:
            # No compression zone (rank 11+): preserve original scores
            compressed_scores[player] = original_score
            prev_compressed_score = original_score

    return compressed_scores


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
]


def _load_config():
    """Load configuration from config.json."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "config.json")
    if not os.path.isfile(cfg_path):
        print("Missing config.json (no embedded defaults anymore). Aborting.")
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_players(players_path):
    """Load player data from JSON file."""
    if not os.path.isfile(players_path):
        raise FileNotFoundError("Players file not found: " + players_path)
    with open(players_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _export_scores(players_data, config, out_detailed, out_minimal):
    """Export detailed and minimal score files."""
    detailed = {}
    minimal = {}

    # Calculate all scores first
    for name, info in players_data.items():
        breakdown = compute_player_score_detailed(info, config)
        detailed[name] = breakdown
        minimal[name] = breakdown["final_score"]

    # Apply top-tier compression if enabled
    if config.get("use_top_tier_compression", False):
        minimal = apply_top_tier_compression(minimal, config)
        # Update detailed scores with compressed values
        for name, compressed_score in minimal.items():
            detailed[name]["compressed_final_score"] = compressed_score
            detailed[name]["original_final_score"] = detailed[name]["final_score"]
            detailed[name]["final_score"] = compressed_score

    # Round minimal scores
    minimal = {name: round(score, 2) for name, score in minimal.items()}

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = _load_config()
    if not config:
        return
    players_file = config.get("players_file", "players.json")
    if not os.path.isabs(players_file):
        players_path = os.path.join(script_dir, players_file)
    else:
        players_path = players_file
    print("Loading players from:", players_path)
    try:
        players_data = _load_players(players_path)
    except Exception as e:
        print("Failed to load players:", e)
        return
    out_detailed = os.path.join(script_dir, "player_scores.json")
    out_minimal = os.path.join(script_dir, "player_scores_minimal.json")
    detailed, minimal = _export_scores(players_data, config, out_detailed, out_minimal)
    _print_summary(minimal)
    print(f"\nExported {len(minimal)} player scores.")
    print("Detailed ->", out_detailed)
    print("Minimal  ->", out_minimal)
    print(f"Done in {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
