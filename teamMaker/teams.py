import json
import math
import time
import os
import random
import copy
from tqdm import tqdm
from collections import defaultdict
from rating import (
    compute_player_score,
    compute_player_score_detailed,
    apply_top_tier_compression,
)

"""
Team Formation Optimizer for Valorant

This script creates balanced teams of 5 players from groups of players
that must stay together. It uses simulated annealing to optimize
team balance based on player ranks and optional tracker scores.

Features two modes:
- BASIC MODE: Original functionality with rank and tracker score balancing
- ADVANCED MODE: Additional features including:
  * Peak rank act weighting (older peak ranks matter less)
  * Role balancing for team compositions  
  * Region-based adjustments (non-EU players get slight debuff)
  * Previous season stats integration (percentile-based S8/S9 data)

Player scores are exported to player_scores.json by default.

Usage:
    python teams.py  # Create balanced teams and export player scores
    
Configuration:
    Set "mode": "basic" or "advanced" in config.json to choose mode
"""


def load_config():
    """
    Load configuration from config.json file

    Returns:
        dict: Configuration settings
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

            # Set random seed if provided in config
            if "random_seed" in config:
                random.seed(config["random_seed"])
                print(f"Using random seed: {config['random_seed']}")

            return config
    except FileNotFoundError:
        print("Config file not found. Using default configuration.")
        return {
            "players_file": "playersexample.json",
            "mode": "basic",  # "basic" or "advanced"
            # Basic mode settings (existing functionality)
            "use_tracker": True,
            "weight_current": 0.8,
            "weight_peak": 0.2,
            "weight_current_tracker": 0.4,
            "weight_peak_tracker": 0.2,
            # Advanced mode settings
            "use_peak_act": True,
            "weight_peak_act": 0.15,
            "peak_act_decay_rate": 0.9,  # Exponential decay per act
            "use_role_balancing": True,
            "role_balance_weight": 2.0,  # Penalty for unbalanced role compositions
            "use_region_debuff": True,
            "non_eu_debuff": 0.95,  # Multiplier for non-EU players
            "use_returning_player_stats": False,  # For previous season percentile scoring
            "returning_player_ranked_weight": 0.7,  # Weight for ranked stats for returning players (vs previous season)
            "previous_season_max_score": 10.0,  # Deprecated - kept for backwards compatibility
            "rank_values": {
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
                "Immortal 1": 22,
                "Immortal 2": 23,
                "Immortal 3": 24,
                "Radiant": 28,
            },
            # Role values for advanced mode
            "role_values": {
                "duelist": 1,
                "initiator": 1,
                "controller": 1,
                "sentinel": 1,
                "flex": 0.8,  # Slightly lower value for flexibility
            },
            "max_time": 1200,
            "early_termination_threshold": 0.5,
            "annealing_iterations": 500000,
            "initial_temperature": 200.0,
            "cooling_rate": 0.997,
            "max_no_improvement": 100000,
            "random_seed": None,
        }


# Legacy helpers removed; act weighting & parsing now handled inside rating module


def get_role_balance_score(team_roles, team_secondary_roles=None):
    """
    Calculate role balance score for a team

    Ideal composition: 1 of each main role (duelist, initiator, controller, sentinel) + 1 repeat
    Preferred repeats: double duelist > double initiator > flex > controller/sentinel

    Args:
        team_roles: List of primary role strings for team members
        team_secondary_roles: Optional list of secondary role strings (or None values)

    Returns:
        float: Balance score (higher is more balanced)
    """
    if not team_roles or len(team_roles) != 5:
        return 0.0

    # Count roles (primary roles get 1.0 weight, secondary roles get 0.5 weight)
    role_counts = {}
    for role in team_roles:
        role_counts[role] = role_counts.get(role, 0) + 1.0

    # Add secondary role contributions (0.5 weight each)
    if team_secondary_roles:
        for secondary_role in team_secondary_roles:
            if secondary_role:  # Skip None values
                role_counts[secondary_role] = role_counts.get(secondary_role, 0) + 0.5

    # Main roles that should ideally all be present
    main_roles = ["duelist", "initiator", "controller", "sentinel"]

    # Calculate balance score
    balance_score = 0.0

    # High bonus for having each main role covered (core requirement)
    # Now considers both primary and secondary roles
    for role in main_roles:
        if role in role_counts and role_counts[role] >= 1.0:
            balance_score += 1.0
        elif role in role_counts and role_counts[role] >= 0.5:
            # Partial credit for secondary role coverage only
            balance_score += 0.5

    # Additional bonuses/penalties for the 5th player (duplicate/flex)
    # Apply bonuses even if missing ONE role (e.g., double duelist without initiator is still decent)
    main_roles_covered = sum(
        1 for role in main_roles if role in role_counts and role_counts[role] >= 1
    )

    if main_roles_covered >= 3:
        # Bonus for having good distribution (3+ different roles)
        if main_roles_covered == 4:
            balance_score += 0.5  # Perfect coverage

        # Bonus for preferred duplicate roles
        if role_counts.get("duelist", 0) == 2:
            balance_score += 1.0  # Double duelist is highly preferred
        elif role_counts.get("initiator", 0) == 2:
            balance_score += 0.8  # Double initiator is also preferred
        elif role_counts.get("flex", 0) >= 1:
            balance_score += 0.5  # Flex is good (secondary role)
        elif (
            role_counts.get("controller", 0) == 2 or role_counts.get("sentinel", 0) == 2
        ):
            balance_score += 0.2  # Double smokes/sentinel is acceptable but not ideal

    # Strong penalty for having 3+ of the same role (except duelist/initiator which get lighter penalty)
    for role, count in role_counts.items():
        if count >= 3:
            if role in ["duelist", "initiator"]:
                balance_score -= (
                    count - 2
                ) * 1.0  # Lighter penalty for duelist/initiator
            else:
                balance_score -= (
                    count - 2
                ) * 2.0  # Heavy penalty for triple+ controller/sentinel/flex

    # Moderate penalty for missing main roles (reduced from 1.5 to 0.8)
    if main_roles_covered < 4:
        balance_score -= (4 - main_roles_covered) * 0.8

    return max(0.0, balance_score)


## Player scoring / previous season logic removed; delegated to rating module.


def build_groups(players_data, config):
    """
    Build groups from player data

    Args:
        players_data: Dict containing player information
        config: Dict containing weight configuration

    Returns:
        tuple: (list of group objects for optimization, list of excluded groups, list of subs)
    """
    group_map = defaultdict(list)
    subs = []

    for player_name, pinfo in players_data.items():
        group_id = pinfo.get("group_id", 0)
        # If group_id is None or null, player is a sub
        if group_id is None:
            subs.append(player_name)
        else:
            group_map[group_id].append(player_name)

    group_list = []
    excluded_groups = []

    for g_id, members in group_map.items():
        group_size = len(members)
        # Use precomputed final score (from rating._export_scores) if available to ensure
        # exact parity with rating.py; otherwise fall back to compute_player_score.
        total_score = 0.0
        for m in members:
            p = players_data.get(m, {})
            if "_computed_final_score" in p:
                total_score += float(p["_computed_final_score"])
            else:
                total_score += compute_player_score(p, config)

        # Collect roles for advanced mode (primary AND secondary roles)
        roles = []
        secondary_roles = []
        if config.get("mode", "basic") == "advanced":
            for m in members:
                player_role = players_data[m].get("role", "flex")
                # Handle both single role and list of roles
                if isinstance(player_role, list):
                    primary_role = player_role[0] if player_role else "flex"
                    roles.append(primary_role)
                    # Track secondary role if present
                    secondary_role = player_role[1] if len(player_role) > 1 else None
                    secondary_roles.append(secondary_role)
                else:
                    roles.append(player_role)
                    secondary_roles.append(None)

        # Check if any members are returning players
        has_returning_player = any(
            players_data[m].get("is_returning_player", False) for m in members
        )

        group_obj = {
            "group_id": g_id,
            "members": members,
            "sum_score": total_score,
            "size": group_size,
            "roles": roles,
            "secondary_roles": (
                secondary_roles if config.get("mode", "basic") == "advanced" else []
            ),
            "has_returning_player": has_returning_player,
        }

        # Separate groups based on size - only 1-3 player groups can be optimized
        if group_size in (1, 2, 3):
            group_list.append(group_obj)
        else:
            excluded_groups.append(group_obj)
            print(
                f"ℹ️  Group {g_id} ({group_size} players) excluded from optimization - scores calculated"
            )

    return group_list, excluded_groups, subs


def find_valid_subset(groups):
    """
    Find a subset whose total size is a multiple of 5,
    maximizing total player count while prioritizing groups with returning players.

    Args:
        groups: List of group objects

    Returns:
        list: List of indices of the chosen subset
    """
    n = len(groups)
    total_size = sum(g["size"] for g in groups)

    # If the total size is divisible by 5, we can use all groups
    if total_size % 5 == 0:
        return list(range(n))

    # Separate groups into returning player groups and others
    returning_groups = []
    other_groups = []

    for i, group in enumerate(groups):
        if group.get("has_returning_player", False):
            returning_groups.append((i, group))
        else:
            other_groups.append((i, group))

    print(f"Found {len(returning_groups)} groups with returning players")

    # Quick optimization: if we need to remove players, prioritize keeping returning players
    remainder = total_size % 5
    players_to_remove = remainder if remainder <= 2 else 5 - remainder

    # First try to find a solution that keeps ALL returning player groups
    if returning_groups:
        returning_indices = [idx for idx, _ in returning_groups]
        returning_size = sum(groups[idx]["size"] for idx in returning_indices)
        remaining_size_needed = total_size - returning_size

        # Check if we can find a subset from other groups that complements returning groups
        other_indices = [idx for idx, _ in other_groups]
        other_sizes = [groups[idx]["size"] for idx in other_indices]

        # We need the total to be divisible by 5
        target_other_size = remaining_size_needed
        while (returning_size + target_other_size) % 5 != 0:
            target_other_size -= 1
            if target_other_size < 0:
                break

        if target_other_size >= 0:
            # Try to find a subset of other groups with the target size
            subset_indices = find_subset_with_sum(other_sizes, target_other_size)
            if subset_indices is not None:
                selected_other_indices = [other_indices[i] for i in subset_indices]
                final_indices = returning_indices + selected_other_indices
                final_size = sum(groups[i]["size"] for i in final_indices)
                if final_size % 5 == 0:
                    print(
                        f"✅ Found solution keeping ALL returning player groups! Total: {final_size} players"
                    )
                    return final_indices

    # If we can't keep all returning players, fall back to the original algorithm
    # but still prioritize returning players when possible
    print("⚠️  Cannot keep all returning player groups, using fallback algorithm...")

    # Sort groups by priority: returning players first, then by size (ascending)
    def group_priority(indexed_group):
        idx, group = indexed_group
        has_returning = group.get("has_returning_player", False)
        size = group["size"]
        # Returning player groups get negative priority (higher priority)
        # Smaller groups get higher priority for removal
        return (not has_returning, size)

    groups_with_indices = [(i, groups[i]) for i in range(len(groups))]
    groups_with_indices.sort(key=group_priority)

    # Try removing the lowest priority groups first (non-returning, smallest)
    removed_size = 0
    removed_indices = set()

    for idx, group in groups_with_indices:
        if removed_size + group["size"] <= players_to_remove:
            # Only remove if it's not a returning player group, unless we have no choice
            if (
                not group.get("has_returning_player", False)
                or len(removed_indices) == 0
            ):
                removed_indices.add(idx)
                removed_size += group["size"]
                if (total_size - removed_size) % 5 == 0:
                    remaining_returning = sum(
                        1
                        for i in range(n)
                        if i not in removed_indices
                        and groups[i].get("has_returning_player", False)
                    )
                    print(
                        f"✅ Found solution keeping {remaining_returning} returning player groups"
                    )
                    return [i for i in range(n) if i not in removed_indices]

    # If simple approach didn't work, fall back to more systematic search
    print("Using optimized subset search...")

    # Group sizes for DP
    sizes = [g["size"] for g in groups]

    # Find the largest subset that sums to a multiple of 5
    # We'll check target sizes in descending order
    for target in range(total_size - (total_size % 5), 0, -5):
        if target < total_size - 10:  # Don't remove more than 10 players
            break

        subset = find_subset_with_sum(sizes, target)
        if subset:
            remaining_returning = sum(
                1 for i in subset if groups[i].get("has_returning_player", False)
            )
            print(
                f"Found solution with {remaining_returning} returning player groups via DP"
            )
            return subset

    # Final fallback: just remove smallest groups until divisible by 5
    # But still try to preserve returning players
    print("Using final fallback: removing smallest non-returning groups first...")
    remaining_indices = list(range(n))
    current_total = total_size

    # Sort by priority: non-returning players first, then by size
    groups_by_priority = sorted(
        enumerate(groups),
        key=lambda x: (x[1].get("has_returning_player", False), x[1]["size"]),
    )

    for idx, group in groups_by_priority:
        if current_total % 5 == 0:
            break
        if idx in remaining_indices:
            remaining_indices.remove(idx)
            current_total -= group["size"]

    remaining_returning = sum(
        1 for i in remaining_indices if groups[i].get("has_returning_player", False)
    )
    print(f"Final fallback kept {remaining_returning} returning player groups")
    return remaining_indices


def find_subset_with_sum(sizes, target):
    """
    Find a subset of group indices that sum to target using dynamic programming.
    Returns the subset indices if found, None otherwise.
    """
    n = len(sizes)
    if target > sum(sizes):
        return None

    # DP table: dp[i][s] = True if we can make sum s using first i elements
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    parent = [[None] * (target + 1) for _ in range(n + 1)]

    # Base case
    for i in range(n + 1):
        dp[i][0] = True

    # Fill DP table
    for i in range(1, n + 1):
        for s in range(target + 1):
            # Don't take current element
            dp[i][s] = dp[i - 1][s]
            if dp[i][s]:
                parent[i][s] = (i - 1, s)

            # Take current element if possible
            if s >= sizes[i - 1] and dp[i - 1][s - sizes[i - 1]]:
                dp[i][s] = True
                parent[i][s] = (i - 1, s - sizes[i - 1])

    if not dp[n][target]:
        return None

    # Reconstruct solution
    subset = []
    i, s = n, target
    while i > 0 and s > 0:
        prev_i, prev_s = parent[i][s]
        if prev_s != s:  # We took element i-1
            subset.append(i - 1)
        i, s = prev_i, prev_s

    return subset


def export_player_scores(players_data, config, output_file=None):
    """
    Calculate and export the score for each player to a JSON file with detailed breakdown

    Args:
        players_data: Dict containing player information
        config: Dict containing weight configuration
        output_file: Optional filename for the output file (default: player_scores.json)

    Returns:
        dict: Dictionary mapping player names to their scores
    """
    if output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "player_scores.json")

    player_scores = {}

    # Calculate score for each player with detailed breakdown
    for player_name, player_info in players_data.items():
        score_breakdown = compute_player_score_detailed(player_info, config)
        player_scores[player_name] = {
            "final_score": score_breakdown["final_score"],
            "current_rank": player_info.get("current_rank", "Unknown"),
            "peak_rank": player_info.get("peak_rank", "Unknown"),
            "tracker_current": player_info.get("tracker_current", 0),
            "tracker_peak": player_info.get("tracker_peak", 0),
            "breakdown": score_breakdown,
        }

    # Apply top-tier compression if enabled
    if config.get("use_top_tier_compression", False):
        minimal = {name: data["final_score"] for name, data in player_scores.items()}
        compressed = apply_top_tier_compression(minimal, config)
        for name, compressed_score in compressed.items():
            if name in player_scores:
                player_scores[name]["final_score"] = round(compressed_score, 2)

    # Sort by score (descending)
    sorted_scores = {
        k: v
        for k, v in sorted(
            player_scores.items(), key=lambda item: item[1]["final_score"], reverse=True
        )
    }

    # Write to file
    with open(output_file, "w") as f:
        json.dump(sorted_scores, f, indent=2)

    print(f"Player scores exported to {output_file}")
    return sorted_scores


def export_minimal_player_scores(players_data, config, output_file=None):
    """
    Calculate and export minimal player scores (just names and final scores) to a JSON file

    Args:
        players_data: Dict containing player information
        config: Dict containing weight configuration
        output_file: Optional filename for the output file (default: player_scores_minimal.json)

    Returns:
        dict: Dictionary mapping player names to their final scores
    """
    if output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "player_scores_minimal.json")

    minimal_scores = {}

    # Calculate score for each player (just final score)
    for player_name, player_info in players_data.items():
        final_score = compute_player_score(player_info, config)
        minimal_scores[player_name] = final_score

    # Apply top-tier compression if enabled
    if config.get("use_top_tier_compression", False):
        minimal_scores = apply_top_tier_compression(minimal_scores, config)

    # Round and sort by score (descending)
    rounded_scores = {k: round(v, 2) for k, v in minimal_scores.items()}
    sorted_minimal_scores = {
        k: v
        for k, v in sorted(
            rounded_scores.items(), key=lambda item: item[1], reverse=True
        )
    }

    # Write to file
    with open(output_file, "w") as f:
        json.dump(sorted_minimal_scores, f, indent=2)

    print(f"Minimal player scores exported to {output_file}")
    return sorted_minimal_scores


def evaluate_teams(teams, config=None):
    """
    Evaluate the quality of team assignments using multiple metrics.

    Args:
        teams: A list of team objects with team_score and groups
        config: Configuration dict (for advanced mode features)

    Returns:
        tuple: (score_range, standard_deviation, role_balance_penalty)
    """
    team_scores = [team["team_score"] for team in teams]

    if not team_scores:
        return float("inf"), float("inf"), float("inf")

    # Calculate score range (difference between highest and lowest team score)
    score_range = max(team_scores) - min(team_scores)

    # Calculate standard deviation
    mean_score = sum(team_scores) / len(team_scores)
    variance = sum((score - mean_score) ** 2 for score in team_scores) / len(
        team_scores
    )
    std_dev = math.sqrt(variance)

    # Role balance penalty for advanced mode
    role_balance_penalty = 0.0
    if (
        config
        and config.get("mode", "basic") == "advanced"
        and config.get("use_role_balancing", False)
    ):
        total_role_penalty = 0.0
        for team in teams:
            # Collect all primary and secondary roles in this team
            team_roles = []
            team_secondary_roles = []
            for group in team["groups"]:
                team_roles.extend(group.get("roles", []))
                team_secondary_roles.extend(group.get("secondary_roles", []))

            # Calculate role balance score (higher is better)
            balance_score = get_role_balance_score(team_roles, team_secondary_roles)
            # Convert to penalty (lower is better)
            role_penalty = max(0, 5.0 - balance_score)  # Max penalty of 5
            total_role_penalty += role_penalty

        role_balance_penalty = total_role_penalty * config.get(
            "role_balance_weight", 2.0
        )

    return score_range, std_dev, role_balance_penalty


def is_valid_assignment(teams, total_players):
    """Check if all teams have exactly 5 players"""
    for team in teams:
        team_size = sum(g["size"] for g in team["groups"])
        if team_size != 5:
            return False

    # Make sure all players are assigned
    assigned_players = sum(sum(g["size"] for g in team["groups"]) for team in teams)
    if assigned_players != total_players:
        return False

    return True


def create_initial_assignment(groups, num_teams):
    """
    Create an initial valid assignment of groups to teams.
    Uses a greedy approach to ensure team sizes are all 5.

    Args:
        groups: List of group objects
        num_teams: Number of teams to create

    Returns:
        list: List of team objects with group assignments
    """
    # Sort groups by size (descending) to place larger groups first
    sorted_groups = sorted(groups, key=lambda g: (-g["size"], -g["sum_score"]))

    # Initialize teams
    teams = [{"team_score": 0.0, "groups": []} for _ in range(num_teams)]
    team_sizes = [0] * num_teams

    # First pass: try to get close to 5 players per team
    for group in sorted_groups:
        # Find team with smallest current size that can accommodate this group
        valid_teams = [
            (i, sz) for i, sz in enumerate(team_sizes) if sz + group["size"] <= 5
        ]
        if not valid_teams:
            continue  # Skip this group if it can't be placed yet

        # Place into team with smallest current size
        target_team = min(valid_teams, key=lambda x: x[1])[0]
        teams[target_team]["groups"].append(group)
        teams[target_team]["team_score"] += group["sum_score"]
        team_sizes[target_team] += group["size"]
        sorted_groups.remove(group)

    # Second pass: fill any remaining slots
    for group in sorted_groups:
        # Find team with smallest size that can accommodate this group
        valid_teams = [
            (i, sz) for i, sz in enumerate(team_sizes) if sz + group["size"] <= 5
        ]
        if not valid_teams:
            continue  # Skip if can't place

        target_team = min(valid_teams, key=lambda x: x[1])[0]
        teams[target_team]["groups"].append(group)
        teams[target_team]["team_score"] += group["sum_score"]
        team_sizes[target_team] += group["size"]

    # Verify all teams have 5 players
    if all(sz == 5 for sz in team_sizes):
        return teams

    # If not valid, try a different approach - bin packing
    return bin_packing_assignment(groups, num_teams)


def bin_packing_assignment(groups, num_teams):
    """
    Alternative team assignment using bin packing approach.

    Args:
        groups: List of group objects
        num_teams: Number of teams to create

    Returns:
        list: List of team objects with group assignments
    """
    # Sort groups by size (descending)
    sorted_groups = sorted(groups, key=lambda g: -g["size"])

    # Initialize teams
    teams = [{"team_score": 0.0, "groups": []} for _ in range(num_teams)]
    team_sizes = [0] * num_teams

    # First place the largest groups
    for group in sorted_groups:
        # Find team with smallest current size that can fit this group
        valid_teams = [
            (i, sz) for i, sz in enumerate(team_sizes) if sz + group["size"] <= 5
        ]
        if not valid_teams:
            # If no team can fit, we have a problem with our groups
            return None

        # Place in team with most remaining space
        target_team = min(valid_teams, key=lambda x: x[1])[0]
        teams[target_team]["groups"].append(group)
        teams[target_team]["team_score"] += group["sum_score"]
        team_sizes[target_team] += group["size"]

    # Check if all teams have 5 players
    if not all(sz == 5 for sz in team_sizes):
        print(
            "Warning: Could not create a valid initial assignment with all teams having 5 players."
        )
        return None

    return teams


def simulated_annealing_team_balancer(groups, config):
    """
    Balance teams using simulated annealing.

    Args:
        groups: List of group objects
        config: Configuration dictionary

    Returns:
        list: List of balanced team objects
    """
    total_players = sum(g["size"] for g in groups)
    num_teams = total_players // 5

    if num_teams == 0 or total_players % 5 != 0:
        print("Error: Total player count must be a multiple of 5")
        return None

    # Create initial assignment
    current_solution = create_initial_assignment(groups, num_teams)
    if not current_solution:
        # Try a different approach if initial assignment failed
        print("Trying alternative initial assignment method...")
        return greedy_team_assignment(groups, num_teams)

    # Check if current solution is valid
    if not is_valid_assignment(current_solution, total_players):
        print("Error: Could not create a valid initial assignment")
        return None

    # Evaluate initial solution
    current_range, current_stdev, current_role_penalty = evaluate_teams(
        current_solution, config
    )
    best_solution = current_solution
    best_range = current_range
    best_stdev = current_stdev
    best_role_penalty = current_role_penalty
    temperature = config.get("initial_temperature", 100.0)
    cooling_rate = config.get("cooling_rate", 0.99)
    max_iterations = config.get("annealing_iterations", 100000)

    progress_bar = tqdm(total=max_iterations, desc="Optimizing teams")

    # Track iterations without improvement for early stopping
    iterations_without_improvement = 0
    max_no_improvement = config.get("max_no_improvement", 10000)

    start_time = time.time()
    max_time = config.get("max_time", 120)  # 2 minutes default

    for iteration in range(max_iterations):
        progress_bar.update(1)

        # Check time limit
        if time.time() - start_time > max_time:
            progress_bar.close()
            print(f"\nReached time limit of {max_time} seconds. Stopping optimization.")
            break

        # Early stopping if no improvement for a while
        if iterations_without_improvement > max_no_improvement:
            progress_bar.close()
            print("\nNo improvement for many iterations. Stopping early.")
            break

        # Create a neighbor solution by swapping groups between teams
        neighbor = create_neighbor_solution(current_solution)

        # Skip invalid neighbors
        if not neighbor or not is_valid_assignment(neighbor, total_players):
            continue

        # Evaluate the neighbor
        neighbor_range, neighbor_stdev, neighbor_role_penalty = evaluate_teams(
            neighbor, config
        )

        # Calculate acceptance probability
        # We want to minimize range, standard deviation, and role penalty
        # with range being the primary concern

        # Energy difference (lower is better)
        energy_diff = (
            (neighbor_range - current_range)
            + 0.2 * (neighbor_stdev - current_stdev)
            + 0.1 * (neighbor_role_penalty - current_role_penalty)
        )

        # Accept if better, or with probability based on temperature if worse
        if energy_diff <= 0 or random.random() < math.exp(-energy_diff / temperature):
            current_solution = neighbor
            current_range = neighbor_range
            current_stdev = neighbor_stdev
            current_role_penalty = neighbor_role_penalty

            # Update best solution if this is better
            if (
                neighbor_range < best_range
                or (neighbor_range == best_range and neighbor_stdev < best_stdev)
                or (
                    neighbor_range == best_range
                    and neighbor_stdev == best_stdev
                    and neighbor_role_penalty < best_role_penalty
                )
            ):
                best_solution = neighbor
                best_range = neighbor_range
                best_stdev = neighbor_stdev
                best_role_penalty = neighbor_role_penalty
                iterations_without_improvement = 0

                # If we find a very good solution, print it and exit early
                # The threshold can be configured in config.json
                early_termination_threshold = config.get(
                    "early_termination_threshold", 2.0
                )
                if best_range < early_termination_threshold:
                    progress_bar.close()
                    print(
                        f"\nFound excellent solution with range: {best_range:.2f}, stdev: {best_stdev:.2f}"
                    )
                    print(
                        f"Early termination as solution is below threshold: {early_termination_threshold}"
                    )
                    break
            else:
                iterations_without_improvement += 1
        else:
            iterations_without_improvement += 1

        # Cool the temperature
        temperature *= cooling_rate

    progress_bar.close()

    print(f"Best solution found - Range: {best_range:.2f}, StdDev: {best_stdev:.2f}")
    if config.get("mode", "basic") == "advanced" and config.get(
        "use_role_balancing", False
    ):
        print(f"Role balance penalty: {best_role_penalty:.2f}")
    return best_solution


def create_neighbor_solution(current_solution):
    """
    Create a neighbor solution by making a small change to current solution.

    Args:
        current_solution: Current team assignment

    Returns:
        list: New team assignment after modification
    """
    # Deep copy the current solution to avoid modifying it
    neighbor = copy.deepcopy(current_solution)

    # Choose the type of neighbor move
    move_type = random.choices(
        ["swap_groups", "move_group"],
        weights=[0.7, 0.3],  # Weights for different move types
        k=1,
    )[0]

    if move_type == "swap_groups":
        # Swap two random groups between two random teams
        if len(neighbor) <= 1:
            return None

        team1_idx = random.randrange(len(neighbor))
        team2_idx = random.randrange(len(neighbor))

        # Ensure teams are different
        while team1_idx == team2_idx:
            team2_idx = random.randrange(len(neighbor))

        team1 = neighbor[team1_idx]
        team2 = neighbor[team2_idx]

        # Skip if either team is empty
        if not team1["groups"] or not team2["groups"]:
            return None

        # Choose random groups from each team
        group1_idx = random.randrange(len(team1["groups"]))
        group2_idx = random.randrange(len(team2["groups"]))

        group1 = team1["groups"][group1_idx]
        group2 = team2["groups"][group2_idx]

        # Check if the swap would keep team sizes valid
        team1_size = sum(g["size"] for g in team1["groups"])
        team2_size = sum(g["size"] for g in team2["groups"])

        new_team1_size = team1_size - group1["size"] + group2["size"]
        new_team2_size = team2_size - group2["size"] + group1["size"]

        if new_team1_size != 5 or new_team2_size != 5:
            return None  # Invalid swap

        # Execute the swap
        # Update team scores
        team1["team_score"] = (
            team1["team_score"] - group1["sum_score"] + group2["sum_score"]
        )
        team2["team_score"] = (
            team2["team_score"] - group2["sum_score"] + group1["sum_score"]
        )

        # Swap the groups
        team1["groups"][group1_idx], team2["groups"][group2_idx] = (
            team2["groups"][group2_idx],
            team1["groups"][group1_idx],
        )

    elif move_type == "move_group":
        # Move a single group to another team, with another balancing group move
        if len(neighbor) <= 1:
            return None

        team1_idx = random.randrange(len(neighbor))
        team2_idx = random.randrange(len(neighbor))

        # Ensure teams are different
        while team1_idx == team2_idx:
            team2_idx = random.randrange(len(neighbor))

        team1 = neighbor[team1_idx]
        team2 = neighbor[team2_idx]

        # Skip if source team is empty
        if not team1["groups"]:
            return None

        # Choose a random group from source team
        group1_idx = random.randrange(len(team1["groups"]))
        group1 = team1["groups"][group1_idx]

        # Check if we can move this group while maintaining valid team sizes
        # Calculate current team sizes
        team1_size = sum(g["size"] for g in team1["groups"])
        team2_size = sum(g["size"] for g in team2["groups"])

        # Calculate new sizes after move
        new_team1_size = team1_size - group1["size"]
        new_team2_size = team2_size + group1["size"]

        # If move would make teams invalid, skip
        if new_team1_size < 0 or new_team2_size > 5:
            return None

        # We need to find a compensating move to keep team sizes at 5
        # Find groups from other teams that could be moved to team1
        needed_size = 5 - new_team1_size

        # Find a group from any team (except team2) with the needed size
        potential_donor_teams = [
            (t_idx, t)
            for t_idx, t in enumerate(neighbor)
            if t_idx != team1_idx and t_idx != team2_idx
        ]

        if not potential_donor_teams:
            return None

        # Try to find a donor group with the exact needed size
        valid_moves = []
        for donor_idx, donor_team in potential_donor_teams:
            for g_idx, g in enumerate(donor_team["groups"]):
                if g["size"] == needed_size:
                    valid_moves.append((donor_idx, g_idx, g))

        if not valid_moves:
            return None  # No valid compensating move found

        # Choose a random valid move
        donor_idx, donor_g_idx, donor_group = random.choice(valid_moves)
        donor_team = neighbor[donor_idx]

        # Execute the moves
        # Move original group from team1 to team2
        team2["groups"].append(group1)
        team2["team_score"] += group1["sum_score"]
        team1["groups"].pop(group1_idx)
        team1["team_score"] -= group1["sum_score"]

        # Move donor group to team1
        team1["groups"].append(donor_group)
        team1["team_score"] += donor_group["sum_score"]
        donor_team["groups"].pop(donor_g_idx)
        donor_team["team_score"] -= donor_group["sum_score"]

    return neighbor


def greedy_team_assignment(groups, num_teams):
    """
    Create balanced teams using a greedy approach.

    Args:
        groups: List of group objects
        num_teams: Number of teams to create

    Returns:
        list: List of team objects
    """
    # Sort groups by score (descending)
    sorted_groups = sorted(groups, key=lambda g: (-g["sum_score"], -g["size"]))

    # Initialize empty teams
    teams = [{"team_score": 0.0, "groups": [], "size": 0} for _ in range(num_teams)]

    # Distribute groups using the greedy algorithm
    for g in sorted_groups:
        # Find team with lowest current score that can fit this group
        valid_teams = [
            (i, t) for i, t in enumerate(teams) if t["size"] + g["size"] <= 5
        ]

        if not valid_teams:
            print("Warning: Could not fit all groups into teams!")
            return None

        # Place in team with lowest score
        target_idx = min(valid_teams, key=lambda x: x[1]["team_score"])[0]

        # Add group to team
        teams[target_idx]["groups"].append(g)
        teams[target_idx]["team_score"] += g["sum_score"]
        teams[target_idx]["size"] += g["size"]

    # Check if assignment is valid
    if not all(t["size"] == 5 for t in teams):
        print("Warning: Not all teams have 5 players in greedy assignment")
        return None

    # Clean up internal size field
    for team in teams:
        del team["size"]

    return teams


def reconstruct_assignment(teams, groups):
    """
    Convert team list back to assignment index array

    Args:
        teams: List of team objects
        groups: Original list of groups

    Returns:
        list: Assignment array mapping group index to team index
    """
    assignment = [-1] * len(groups)

    for team_idx, team in enumerate(teams):
        for team_group in team["groups"]:
            # Find matching original group
            for group_idx, group in enumerate(groups):
                if group["group_id"] == team_group["group_id"]:
                    assignment[group_idx] = team_idx
                    break

    return assignment


def calculate_statistics(teams, config=None):
    """
    Calculate statistics for team scores

    Args:
        teams: List of team objects
        config: Configuration dict (for advanced mode features)

    Returns:
        tuple: (min score, max score, range, standard deviation, avg_role_balance)
    """
    scores = [team["team_score"] for team in teams]

    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    # Calculate standard deviation
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    std_dev = math.sqrt(variance)

    # Calculate average role balance for advanced mode
    avg_role_balance = 0.0
    if (
        config
        and config.get("mode", "basic") == "advanced"
        and config.get("use_role_balancing", False)
    ):
        total_balance = 0.0
        for team in teams:
            team_roles = []
            team_secondary_roles = []
            for group in team["groups"]:
                team_roles.extend(group.get("roles", []))
                team_secondary_roles.extend(group.get("secondary_roles", []))
            balance_score = get_role_balance_score(team_roles, team_secondary_roles)
            total_balance += balance_score
        avg_role_balance = total_balance / len(teams) if teams else 0.0

    return min_score, max_score, score_range, std_dev, avg_role_balance


## Previous season percentile utilities removed (now centralized in rating.py with optional config distributions)


def main():
    """
    Main program logic
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load configuration
    config = load_config()
    mode = config.get("mode", "basic")

    print(f"Running in {mode.upper()} mode")
    if mode == "advanced":
        print("Advanced features enabled:")
        if config.get("use_peak_act", False):
            print("  - Peak rank act weighting")
        if config.get("use_role_balancing", False):
            print("  - Role balancing")
        if config.get("use_region_debuff", False):
            print("  - Region-based adjustments")
        if config.get("use_returning_player_stats", False):
            print("  - Returning player stats (percentile-based)")

    # Determine players file path from config
    players_file = config.get("players_file", "players.json")
    players_path = (
        os.path.join(script_dir, players_file)
        if not os.path.isabs(players_file)
        else players_file
    )
    print(f"Loading player data from {players_path}")

    # Load player data
    try:
        with open(players_path, "r") as f:
            players_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Player file '{players_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Player file '{players_path}' is not valid JSON.")
        return

    # Always export player scores by default (both detailed and minimal)
    detailed_output_file = os.path.join(script_dir, "player_scores.json")
    minimal_output_file = os.path.join(script_dir, "player_scores_minimal.json")

    # Use rating._export_scores to compute scores (ensures identical logic to rating.py)
    try:
        from rating import _export_scores

        detailed_scores, minimal_scores = _export_scores(
            players_data, config, detailed_output_file, minimal_output_file
        )
        # Attach compressed minimal scores into players_data for consistent group sums
        for name, score in minimal_scores.items():
            if name in players_data:
                players_data[name]["_computed_final_score"] = score
    except Exception:
        # Fallback to existing exporters if _export_scores unavailable
        export_player_scores(players_data, config, detailed_output_file)
        export_minimal_player_scores(players_data, config, minimal_output_file)

    # Build groups (build_groups will prefer precomputed '_computed_final_score' when present)
    groups, excluded_groups, subs = build_groups(players_data, config)
    total_players = sum(g["size"] for g in groups)
    excluded_players = sum(g["size"] for g in excluded_groups)

    print(f"Total players for optimization = {total_players} in {len(groups)} groups.")
    if excluded_groups:
        print(
            f"Excluded players = {excluded_players} in {len(excluded_groups)} groups (scores calculated but not optimized)."
        )

    # Find the best subset of players
    best_subset = find_valid_subset(groups)
    if not best_subset:
        print("No valid subset of groups can form complete 5-person teams.")
        # Still display excluded groups even if no optimization possible
        display_excluded_groups(excluded_groups, config, players_data)
        return

    chosen_groups = [groups[i] for i in best_subset]
    best_size = sum(g["size"] for g in chosen_groups)
    num_teams = best_size // 5

    # Count returning players in chosen groups
    chosen_returning_groups = sum(
        1 for grp in chosen_groups if grp.get("has_returning_player", False)
    )
    total_returning_groups = sum(
        1 for grp in groups if grp.get("has_returning_player", False)
    )

    print(f"Found optimal subset with {best_size} players (divisible by 5).")
    print(f"Creating {num_teams} balanced teams...")
    print(
        f"✅ Included {chosen_returning_groups}/{total_returning_groups} groups with returning players"
    )

    # Print some basic stats about the groups
    print("\nGroup score distribution:")
    scores = [g["sum_score"] for g in chosen_groups]
    print(f"  Min group score: {min(scores):.2f}")
    print(f"  Max group score: {max(scores):.2f}")
    print(f"  Avg group score: {sum(scores)/len(scores):.2f}")
    print(f"  Total score: {sum(scores):.2f}")
    print(f"  Ideal team score: {sum(scores)/num_teams:.2f}")

    # Time the team formation process
    start_time = time.time()

    # Find the best team arrangement using simulated annealing
    best_teams = simulated_annealing_team_balancer(chosen_groups, config)

    end_time = time.time()
    print(f"Team formation completed in {end_time - start_time:.2f} seconds.")

    if not best_teams:
        print("No valid arrangement found!")
        return

    # Identify leftover groups
    leftover_groups = [g for i, g in enumerate(groups) if i not in best_subset]
    leftover_players = sum(g["size"] for g in leftover_groups)

    # Calculate statistics
    min_score, max_score, score_range, std_dev, avg_role_balance = calculate_statistics(
        best_teams, config
    )

    # Print results
    print("\n==== Best Team Arrangement ====")
    print(f"Using {best_size} players to form {len(best_teams)} teams of 5.")
    print(f"Score difference across teams = {score_range:.2f}")
    print(f"Standard deviation of team scores = {std_dev:.2f}")

    if mode == "advanced" and config.get("use_role_balancing", False):
        print(f"Average role balance score = {avg_role_balance:.2f}/5.0")

    # Print team compositions
    team_scores = []
    for i, team in enumerate(best_teams, start=1):
        team_scores.append(team["team_score"])
        print(f"\nTEAM {i}: total score = {team['team_score']:.2f}")

        if mode == "advanced" and config.get("use_role_balancing", False):
            # Show role composition including secondary roles
            team_roles = []
            team_secondary_roles = []
            for group in team["groups"]:
                team_roles.extend(group.get("roles", []))
                team_secondary_roles.extend(group.get("secondary_roles", []))
            role_balance = get_role_balance_score(team_roles, team_secondary_roles)
            role_counts = {}
            for role in team_roles:
                role_counts[role] = role_counts.get(role, 0) + 1
            role_summary = ", ".join(
                [f"{role}:{count}" for role, count in sorted(role_counts.items())]
            )
            print(f"  Roles: {role_summary} (balance: {role_balance:.1f}/5.0)")

        for grp in sorted(team["groups"], key=lambda g: g["sum_score"], reverse=True):
            # Show detailed role info for each player in advanced mode
            if mode == "advanced":
                member_roles = []
                for member in grp["members"]:
                    role_display = get_player_role_display(member, players_data)
                    member_roles.append(f"{member}({role_display})")
                members_str = ", ".join(member_roles)
            else:
                members_str = ", ".join(grp["members"])

            print(
                f"  Group {grp['group_id']} (size={grp['size']}, score={grp['sum_score']:.2f}) => {members_str}"
            )

    # Print score comparison
    print("\nTeam score comparison:")
    team_indices = list(range(len(team_scores)))
    team_indices.sort(key=lambda i: team_scores[i])

    mean_score = sum(team_scores) / len(team_scores)
    for i in team_indices:
        diff_from_avg = team_scores[i] - mean_score
        print(f"  Team {i+1}: {team_scores[i]:.2f} ({diff_from_avg:+.2f} from avg)")

    # Print leftover players
    if leftover_groups:
        # Count returning players in leftover groups
        leftover_returning_count = sum(
            1 for grp in leftover_groups if grp.get("has_returning_player", False)
        )
        returning_status = (
            f" (⚠️  {leftover_returning_count} groups contain returning players)"
            if leftover_returning_count > 0
            else ""
        )

        print(
            f"\nLeftover groups (subs) totaling {leftover_players} players{returning_status}:"
        )
        for grp in sorted(leftover_groups, key=lambda g: g["sum_score"], reverse=True):
            # Show detailed role info for each player in advanced mode
            if mode == "advanced":
                member_roles = []
                for member in grp["members"]:
                    role_display = get_player_role_display(member, players_data)
                    # Mark returning players with a star
                    if players_data[member].get("is_returning_player", False):
                        member_roles.append(f"{member}({role_display})⭐")
                    else:
                        member_roles.append(f"{member}({role_display})")
                members_str = ", ".join(member_roles)
            else:
                members_str = ", ".join(
                    [
                        (
                            f"{member}⭐"
                            if players_data[member].get("is_returning_player", False)
                            else member
                        )
                        for member in grp["members"]
                    ]
                )

            # Add returning player indicator to group info
            returning_indicator = (
                " 🔄" if grp.get("has_returning_player", False) else ""
            )
            print(
                f"  Group {grp['group_id']} (size={grp['size']}, score={grp['sum_score']:.2f}){returning_indicator} => {members_str}"
            )
    else:
        print("\nNo leftover groups - all players assigned to teams!")

    # Display excluded groups information
    display_excluded_groups(excluded_groups, config, players_data)

    # Final summary about returning player inclusion
    all_returning_groups = sum(
        1 for grp in groups if grp.get("has_returning_player", False)
    )
    chosen_returning_groups = sum(
        1 for grp in chosen_groups if grp.get("has_returning_player", False)
    )
    leftover_returning_groups = sum(
        1 for grp in leftover_groups if grp.get("has_returning_player", False)
    )
    excluded_returning_groups = sum(
        1 for grp in excluded_groups if grp.get("has_returning_player", False)
    )

    print("\n" + "=" * 60)
    print("📊 RETURNING PLAYER SUMMARY")
    print("=" * 60)
    print(f"Total groups with returning players: {all_returning_groups}")
    print(f"  ✅ Included in teams: {chosen_returning_groups}")
    if leftover_returning_groups > 0:
        print(f"  ⚠️  Left as subs: {leftover_returning_groups}")
    if excluded_returning_groups > 0:
        print(f"  🚫 Excluded (5+ stacks): {excluded_returning_groups}")

    if chosen_returning_groups == all_returning_groups - excluded_returning_groups:
        print(
            "🎉 SUCCESS: All eligible returning player groups were included in teams!"
        )
    elif leftover_returning_groups == 0:
        print("✅ Good: No returning player groups left as subs")
    else:
        print(
            f"⚠️  Warning: {leftover_returning_groups} returning player groups left as subs"
        )
    print("=" * 60)

    # Display subs (players with null group_id)
    if subs:
        print("\n" + "=" * 60)
        print("📋 AVAILABLE SUBS (Players with null group_id)")
        print("=" * 60)
        print(f"Total subs available: {len(subs)}\n")

        # Sort subs by score (descending)
        subs_with_scores = []
        for sub_name in subs:
            sub_info = players_data[sub_name]
            sub_score = compute_player_score(sub_info, config)
            subs_with_scores.append((sub_name, sub_score, sub_info))

        subs_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Display each sub with their info
        for sub_name, sub_score, sub_info in subs_with_scores:
            current_rank = sub_info.get("current_rank", "N/A")
            peak_rank = sub_info.get("peak_rank", "N/A")
            role = sub_info.get("role", ["flex"])
            if isinstance(role, list):
                role_str = "/".join(role)
            else:
                role_str = role
            region = sub_info.get("region", "EU")
            returning = " 🔄" if sub_info.get("is_returning_player", False) else ""

            print(
                f"  {sub_name:<20} | Score: {sub_score:>6.2f} | {current_rank:<15} -> {peak_rank:<15} | Role: {role_str:<20} | {region}{returning}"
            )

        print("=" * 60)


def display_excluded_groups(excluded_groups, config, players_data):
    """
    Display information about excluded groups (e.g., 5-stacks)

    Args:
        excluded_groups: List of group objects that were excluded from optimization
        config: Dict containing weight configuration
        players_data: Dict containing all player data
    """
    if not excluded_groups:
        return

    print(f"\n🚫 Excluded Groups (not participating in team optimization):")
    print("=" * 60)

    # Convert groups to team-like objects for reusing calculate_statistics
    pseudo_teams = []
    for group in excluded_groups:
        pseudo_teams.append(
            {
                "team_score": group["sum_score"],
                "groups": [group],  # Wrap the group in a list to match team structure
            }
        )

    # Calculate statistics using the same function as optimization
    min_score, max_score, score_range, std_dev, avg_role_balance = calculate_statistics(
        pseudo_teams, config
    )

    # Display statistics if we have multiple groups
    if len(excluded_groups) > 1:
        print(f"Score difference across groups = {score_range:.2f}")
        print(f"Standard deviation of group scores = {std_dev:.2f}")

        mode = config.get("mode", "basic")
        if mode == "advanced" and config.get("use_role_balancing", False):
            print(f"Average role balance score = {avg_role_balance:.2f}/5.0")

    # Display individual group information
    mode = config.get("mode", "basic")

    for i, group in enumerate(excluded_groups, start=1):
        # Add returning player indicator to group info
        returning_indicator = " 🔄" if group.get("has_returning_player", False) else ""

        print(
            f"\nGroup {group['group_id']} ({group['size']} players) - Total Score: {group['sum_score']:.2f}{returning_indicator}"
        )

        # Show role composition for advanced mode
        if mode == "advanced" and config.get("use_role_balancing", False):
            team_roles = group.get("roles", [])
            team_secondary_roles = group.get("secondary_roles", [])
            if team_roles and len(team_roles) == 5:  # Only show for complete teams
                role_balance = get_role_balance_score(team_roles, team_secondary_roles)
                role_counts = {}
                for role in team_roles:
                    role_counts[role] = role_counts.get(role, 0) + 1
                role_summary = ", ".join(
                    [f"{role}:{count}" for role, count in sorted(role_counts.items())]
                )
                print(f"  Roles: {role_summary} (balance: {role_balance:.1f}/5.0)")

        # Show detailed role info for each player in advanced mode
        if mode == "advanced":
            member_roles = []
            for member in group["members"]:
                role_display = get_player_role_display(member, players_data)
                # Mark returning players with a star
                if players_data[member].get("is_returning_player", False):
                    member_roles.append(f"{member}({role_display})⭐")
                else:
                    member_roles.append(f"{member}({role_display})")
            members_str = ", ".join(member_roles)
        else:
            members_str = ", ".join(
                [
                    (
                        f"{member}⭐"
                        if players_data[member].get("is_returning_player", False)
                        else member
                    )
                    for member in group["members"]
                ]
            )

        print(f"  Members: {members_str}")

    # Show score comparison using the same logic as optimization
    if len(excluded_groups) > 1:
        print("\nGroup score comparison:")
        group_scores = [group["sum_score"] for group in excluded_groups]
        group_indices = list(range(len(group_scores)))
        group_indices.sort(key=lambda i: group_scores[i])

        mean_score = sum(group_scores) / len(group_scores)
        for i in group_indices:
            group = excluded_groups[i]
            diff_from_avg = group_scores[i] - mean_score
            print(
                f"  Group {group['group_id']}: {group_scores[i]:.2f} ({diff_from_avg:+.2f} from avg)"
            )


def get_player_role_display(player_name, players_data):
    """
    Get formatted role display string for a player

    Args:
        player_name: Name of the player
        players_data: Dictionary containing all player data

    Returns:
        str: Formatted role string (e.g., "duelist/sentinel" for multi-role)
    """
    player_info = players_data.get(player_name, {})
    player_role = player_info.get("role", "flex")

    if isinstance(player_role, list):
        return "/".join(player_role)
    else:
        return player_role


if __name__ == "__main__":
    main()
