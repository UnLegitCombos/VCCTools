"""Rating & player scoring (trimmed)."""

import math
import os
import json
import time


def rank_to_numeric(rank_str, rank_values):
    return rank_values.get(rank_str, 0)


def parse_peak_act(peak_act_str, current_season=25, current_act=3):
    if not peak_act_str:
        return 0

    s = peak_act_str.upper().strip()
    try:
        if s.startswith("E"):
            parts = s[1:].split("A")
            if len(parts) == 2:
                episode = int(parts[0])
                act = int(parts[1])
                if episode < 1 or episode > 9 or act < 1 or act > 3:
                    return 0
                peak_total_acts = (episode - 1) * 3 + act
                current_total_acts = 9 * 3 + (current_season - 25) * 4 + current_act
                return max(0, current_total_acts - peak_total_acts)
        elif s.startswith("S"):
            parts = s[1:].split("A")
            if len(parts) == 2:
                season = int(parts[0])
                act = int(parts[1])
                if season < 25 or act < 1 or act > 4:
                    return 0
                current_total_season_acts = (current_season - 25) * 4 + current_act
                peak_total_season_acts = (season - 25) * 4 + act
                return max(0, current_total_season_acts - peak_total_season_acts)
    except (ValueError, IndexError):
        pass
    return 0


def calculate_peak_act_weight(acts_ago, decay_rate):
    return decay_rate**acts_ago


def _get_season_distributions(config):
    dist = config.get("season_rating_distributions")
    if dist:
        return {k.upper(): sorted(v) for k, v in dist.items() if isinstance(v, list)}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "season_distributions.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {k.upper(): sorted(v) for k, v in raw.items() if isinstance(v, list)}
        except Exception as e:
            print("Could not load season_distributions.json:", e)
    return {}


def _percentile(value, sorted_values):
    if not sorted_values:
        return 50.0
    count = 0
    for v in sorted_values:
        if value > v:
            count += 1
        else:
            break
    return min(100.0, max(0.0, (count / len(sorted_values)) * 100))


def calculate_previous_season_score(player_info, config):
    if not config.get("use_returning_player_stats", False):
        return 0.0
    if not player_info.get("is_returning_player", False):
        return 0.0

    prev_stats = player_info.get("previous_season_stats", {})
    if not prev_stats:
        return 0.0

    distributions = _get_season_distributions(config)

    def convert_zscore_to_score(rating, distribution):
        """
        Use Z-Score method to preserve actual performance differences.
        This correctly handles cross-season comparisons where formula changes
        have shifted the overall rating distributions.
        """
        if not distribution or len(distribution) < 2:
            return 15.0

        # Calculate mean and standard deviation
        mean = sum(distribution) / len(distribution)
        variance = sum((x - mean) ** 2 for x in distribution) / len(distribution)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
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
    return compute_player_score_detailed(player_info, config)["final_score"]


def compute_player_score_detailed(player_info, config):
    mode = config.get("mode", "basic")
    rank_values = config.get("rank_values", {})
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

        # Region debuff
        if config.get("use_region_debuff"):
            region = str(player_info.get("region", "EU")).upper()
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


__all__ = [
    "compute_player_score",
    "compute_player_score_detailed",
    "calculate_previous_season_score",
    "parse_peak_act",
    "calculate_peak_act_weight",
]


def _load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "config.json")
    if not os.path.isfile(cfg_path):
        print("Missing config.json (no embedded defaults anymore). Aborting.")
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_players(players_path):
    if not os.path.isfile(players_path):
        raise FileNotFoundError("Players file not found: " + players_path)
    with open(players_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _export_scores(players_data, config, out_detailed, out_minimal):
    detailed = {}
    minimal = {}
    for name, info in players_data.items():
        breakdown = compute_player_score_detailed(info, config)
        detailed[name] = breakdown
        minimal[name] = round(breakdown["final_score"], 2)
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
    items = list(minimal_scores.items())
    print("\nTop players:")
    for i, (name, score) in enumerate(items[:top], 1):
        print(f" {i:2d}. {name:25s} {score:.2f}")
    if len(items) > top:
        print(f" ... and {len(items)-top} more")


def main():
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
