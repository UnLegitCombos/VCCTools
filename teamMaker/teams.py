import json
import math
import time
import os
import random
from tqdm import tqdm
from collections import defaultdict

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


def rank_to_numeric(rank_str, rank_values):
    """Convert rank string to numeric value"""
    return rank_values.get(rank_str, 0)


def parse_peak_act(peak_act_str):
    """
    Parse peak act string (e.g., "E9A3" or "S25A3") and return episodes/acts ago

    Episodes (E1-E9): 3 acts per episode
    Seasons (S25+): 4 acts per season, starting after E9A3

    Args:
        peak_act_str: String like "E9A3" (Episode 9 Act 3) or "S25A3" (Season 25 Act 3)

    Returns:
        int: Number of acts ago (0 = current act)
    """
    if not peak_act_str:
        return 0

    try:
        # Current season/episode reference (adjust these as needed)
        # Update these values based on current Valorant act
        current_season = 25  # Current season number
        current_act = 3  # Current act within current season

        peak_act_str = peak_act_str.upper().strip()

        if peak_act_str.startswith("E"):
            # Episode format: E1A1 to E9A3
            parts = peak_act_str[1:].split("A")
            if len(parts) == 2:
                episode = int(parts[0])
                act = int(parts[1])

                # Episodes go from E1 to E9, each with 3 acts
                if episode < 1 or episode > 9 or act < 1 or act > 3:
                    return 0  # Invalid episode/act

                # Calculate total acts from start of E1A1 to peak
                peak_total_acts = (episode - 1) * 3 + act

                # Calculate total acts from start of E1A1 to current season
                # E9A3 is the last episode act, then S25 starts
                current_total_acts = 9 * 3 + (current_season - 25) * 4 + current_act

                acts_ago = current_total_acts - peak_total_acts
                return max(0, acts_ago)

        elif peak_act_str.startswith("S"):
            # Season format: S25A1 onwards (4 acts per season)
            parts = peak_act_str[1:].split("A")
            if len(parts) == 2:
                season = int(parts[0])
                act = int(parts[1])

                # Seasons start from S25, each with 4 acts
                if season < 25 or act < 1 or act > 4:
                    return 0  # Invalid season/act

                # Calculate acts ago within the season system
                current_total_season_acts = (current_season - 25) * 4 + current_act
                peak_total_season_acts = (season - 25) * 4 + act

                acts_ago = current_total_season_acts - peak_total_season_acts
                return max(0, acts_ago)

    except (ValueError, IndexError):
        pass

    return 0  # Default to current act if parsing fails


def calculate_peak_act_weight(acts_ago, decay_rate):
    """
    Calculate weight for peak rank based on how many acts ago it was achieved

    Args:
        acts_ago: Number of acts since peak was achieved
        decay_rate: Exponential decay rate per act

    Returns:
        float: Weight multiplier (1.0 for current act, decreasing exponentially)
    """
    return decay_rate**acts_ago


def get_role_balance_score(team_roles):
    """
    Calculate role balance score for a team

    Args:
        team_roles: List of role strings for team members

    Returns:
        float: Balance score (higher is more balanced)
    """
    if not team_roles or len(team_roles) != 5:
        return 0.0

    # Count roles
    role_counts = {}
    for role in team_roles:
        role_counts[role] = role_counts.get(role, 0) + 1

    # Ideal composition: 1 of each main role (duelist, initiator, controller, sentinel)
    # with one flex or duplicate
    main_roles = ["duelist", "initiator", "controller", "sentinel"]

    # Calculate balance score for display
    balance_score = 0.0

    # Bonus for having each main role covered
    for role in main_roles:
        if role in role_counts and role_counts[role] >= 1:
            balance_score += 1.0

    # Penalty for having too many of the same role
    for role, count in role_counts.items():
        if count > 2:  # More than 2 of the same role is bad
            balance_score -= (count - 2) * 0.5

    # Bonus for balanced distribution
    if len(role_counts) >= 4:  # At least 4 different roles
        balance_score += 0.5

    return max(0.0, balance_score)


def compute_player_score(player_info, config):
    """
    Compute a player's score based on ranks and optionally tracker values
    Supports both basic and advanced modes

    Args:
        player_info: Dict containing player data
        config: Dict containing weight configuration

    Returns:
        float: Player's calculated score
    """
    mode = config.get("mode", "basic")
    current_val = rank_to_numeric(player_info["current_rank"], config["rank_values"])
    peak_val = rank_to_numeric(player_info["peak_rank"], config["rank_values"])

    # Base score from ranks
    score = config["weight_current"] * current_val + config["weight_peak"] * peak_val

    # Add tracker scores if enabled (both modes)
    if (
        config["use_tracker"]
        and "tracker_current" in player_info
        and "tracker_peak" in player_info
    ):
        current_tracker = player_info["tracker_current"]
        peak_tracker = player_info["tracker_peak"]

        # Apply non-linear scaling to tracker values
        current_tracker_score = config["weight_current_tracker"] * math.sqrt(
            max(0, current_tracker)
        )
        peak_tracker_score = config["weight_peak_tracker"] * math.log(
            1 + max(0, peak_tracker)
        )

        # Reward consistency (how close current rank is to peak rank)
        consistency_factor = 1.0
        if peak_val > 0:
            rank_ratio = current_val / peak_val
            consistency_factor = 1.0 + (0.1 * rank_ratio)

        # Combine all factors
        score = (
            score + current_tracker_score + peak_tracker_score
        ) * consistency_factor

    # Advanced mode features
    if mode == "advanced":
        # Peak rank act weighting
        if config.get("use_peak_act", False) and "peak_rank_act" in player_info:
            acts_ago = parse_peak_act(player_info["peak_rank_act"])
            act_weight = calculate_peak_act_weight(
                acts_ago, config.get("peak_act_decay_rate", 0.9)
            )
            peak_act_bonus = config.get("weight_peak_act", 0.15) * peak_val * act_weight
            score += peak_act_bonus

        # Previous season stats handling - replaces part of ranked stats for returning players
        if config.get("use_returning_player_stats", False) and player_info.get(
            "is_returning_player", False
        ):
            previous_season_score = calculate_previous_season_score(player_info, config)
            if previous_season_score > 0:
                # Weight configuration for returning players
                ranked_weight = config.get(
                    "returning_player_ranked_weight", 0.7
                )  # 70% ranked
                previous_weight = 1.0 - ranked_weight  # 30% previous season

                # Blend current score (ranked-based) with previous season score
                score = score * ranked_weight + previous_season_score * previous_weight

        # Region debuff for non-EU players (ping penalty only)
        if config.get("use_region_debuff", False):
            region = player_info.get("region", "EU").upper()
            if region != "EU":
                score *= config.get("non_eu_debuff", 0.95)

    return score


def build_groups(players_data, config):
    """
    Build groups from player data

    Args:
        players_data: Dict containing player information
        config: Dict containing weight configuration

    Returns:
        tuple: (list of group objects for optimization, list of excluded groups)
    """
    group_map = defaultdict(list)
    for player_name, pinfo in players_data.items():
        group_id = pinfo.get("group_id", 0)
        group_map[group_id].append(player_name)

    group_list = []
    excluded_groups = []

    for g_id, members in group_map.items():
        group_size = len(members)
        total_score = sum(
            compute_player_score(players_data[m], config) for m in members
        )

        # Collect roles for advanced mode (primary role only)
        roles = []
        if config.get("mode", "basic") == "advanced":
            for m in members:
                player_role = players_data[m].get("role", "flex")
                # Handle both single role and list of roles - use primary (first) role only
                if isinstance(player_role, list):
                    primary_role = player_role[0] if player_role else "flex"
                    roles.append(primary_role)
                else:
                    roles.append(player_role)

        group_obj = {
            "group_id": g_id,
            "members": members,
            "sum_score": total_score,
            "size": group_size,
            "roles": roles,
        }

        # Separate groups based on size - only 1-3 player groups can be optimized
        if group_size in (1, 2, 3):
            group_list.append(group_obj)
        else:
            excluded_groups.append(group_obj)
            print(
                f"ℹ️  Group {g_id} ({group_size} players) excluded from optimization - scores calculated"
            )

    return group_list, excluded_groups


def find_valid_subset(groups):
    """
    Find a subset whose total size is a multiple of 5,
    maximizing total player count.

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

    # Quick optimization: if we need to remove players, find the smallest groups to remove
    remainder = total_size % 5
    players_to_remove = remainder if remainder <= 2 else 5 - remainder

    # Sort groups by size (ascending) to prioritize removing smallest groups
    groups_with_indices = [(i, g["size"]) for i, g in enumerate(groups)]
    groups_with_indices.sort(key=lambda x: x[1])

    # Try removing the smallest groups first
    removed_size = 0
    removed_indices = set()

    for idx, size in groups_with_indices:
        if removed_size + size <= players_to_remove:
            removed_indices.add(idx)
            removed_size += size
            if (total_size - removed_size) % 5 == 0:
                # Found a valid solution
                return [i for i in range(n) if i not in removed_indices]

    # If simple approach didn't work, fall back to more systematic search
    # But limit the search space by using dynamic programming approach
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
            return subset

    # Final fallback: just remove smallest groups until divisible by 5
    print("Using fallback: removing smallest groups...")
    remaining_indices = list(range(n))
    current_total = total_size

    groups_by_size = sorted(enumerate(groups), key=lambda x: x[1]["size"])

    for idx, group in groups_by_size:
        if current_total % 5 == 0:
            break
        remaining_indices.remove(idx)
        current_total -= group["size"]

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
    Calculate and export the score for each player to a JSON file

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

    # Calculate score for each player
    for player_name, player_info in players_data.items():
        score = compute_player_score(player_info, config)
        player_scores[player_name] = {
            "score": round(score, 2),
            "current_rank": player_info.get("current_rank", "Unknown"),
            "peak_rank": player_info.get("peak_rank", "Unknown"),
            "tracker_current": player_info.get("tracker_current", 0),
            "tracker_peak": player_info.get("tracker_peak", 0),
        }

    # Sort by score (descending)
    sorted_scores = {
        k: v
        for k, v in sorted(
            player_scores.items(), key=lambda item: item[1]["score"], reverse=True
        )
    }

    # Write to file
    with open(output_file, "w") as f:
        json.dump(sorted_scores, f, indent=2)

    print(f"Player scores exported to {output_file}")
    return sorted_scores


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
            # Collect all roles in this team
            team_roles = []
            for group in team["groups"]:
                team_roles.extend(group.get("roles", []))

            # Calculate role balance score (higher is better)
            balance_score = get_role_balance_score(team_roles)
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
    import copy

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
            for group in team["groups"]:
                team_roles.extend(group.get("roles", []))
            balance_score = get_role_balance_score(team_roles)
            total_balance += balance_score
        avg_role_balance = total_balance / len(teams) if teams else 0.0

    return min_score, max_score, score_range, std_dev, avg_role_balance


def get_season_rating_percentiles():
    """
    Get the rating distributions for previous seasons to calculate percentiles

    Returns:
        dict: Season data with sorted rating lists
    """
    # S8 adjusted ratings (sorted ascending for percentile calculation)
    s8_ratings = [
        0.38,
        0.38,
        0.40,
        0.42,
        0.50,
        0.50,
        0.50,
        0.51,
        0.58,
        0.61,
        0.63,
        0.64,
        0.68,
        0.68,
        0.70,
        0.71,
        0.74,
        0.76,
        0.77,
        0.78,
        0.80,
        0.81,
        0.82,
        0.85,
        0.88,
        0.91,
        0.92,
        0.97,
        0.98,
        0.99,
        1.00,
        1.02,
        1.02,
        1.02,
        1.04,
        1.05,
        1.05,
        1.08,
        1.09,
        1.11,
        1.17,
        1.22,
        1.24,
        1.30,
        1.42,
        1.42,
        1.46,
        1.49,
        1.50,
    ]

    # S9 adjusted ratings (sorted ascending for percentile calculation)
    s9_ratings = [
        0.561,
        0.608,
        0.656,
        0.668,
        0.732,
        0.742,
        0.755,
        0.781,
        0.795,
        0.807,
        0.848,
        0.849,
        0.875,
        0.878,
        0.896,
        0.909,
        0.923,
        0.940,
        0.952,
        0.960,
        0.967,
        0.979,
        0.989,
        0.989,
        0.998,
        1.005,
        1.006,
        1.006,
        1.055,
        1.078,
        1.088,
        1.148,
        1.179,
        1.189,
        1.264,
    ]

    return {"S8": sorted(s8_ratings), "S9": sorted(s9_ratings)}


def calculate_rating_percentile(player_rating, season_ratings):
    """
    Calculate the percentile of a player's rating within their season

    Args:
        player_rating: Player's rating in that season
        season_ratings: Sorted list of all ratings from that season

    Returns:
        float: Percentile (0-100)
    """
    if not season_ratings:
        return 50.0  # Default to median if no data

    # Count how many players this player performed better than
    better_than_count = 0
    for rating in season_ratings:
        if player_rating > rating:
            better_than_count += 1
        else:
            break  # Since list is sorted, we can break early

    # Calculate percentile
    total_players = len(season_ratings)
    percentile = (better_than_count / total_players) * 100

    return min(100.0, max(0.0, percentile))


def calculate_previous_season_score(player_info, config):
    """
    Calculate score from previous season stats using percentile approach
    Returns a score comparable to ranked-based scoring (not additive)

    Args:
        player_info: Dict containing player data including previous season stats
        config: Dict containing configuration

    Returns:
        float: Previous season score (comparable to ranked score scale)
    """
    if not config.get("use_returning_player_stats", False):
        return 0.0

    if not player_info.get("is_returning_player", False):
        return 0.0

    # Check if player has previous season data
    previous_stats = player_info.get("previous_season_stats", {})
    if not previous_stats:
        return 0.0

    # Handle both single season object and list of seasons
    if isinstance(previous_stats, list):
        # Multiple seasons - take the best percentile
        best_score = 0.0
        for stats in previous_stats:
            season = stats.get("season", "").upper()
            rating = stats.get("adjusted_rating", 0.0)

            if not season or rating <= 0:
                continue

            # Get season rating distributions
            season_data = get_season_rating_percentiles()

            if season not in season_data:
                continue  # Unknown season

            # Calculate percentile
            percentile = calculate_rating_percentile(rating, season_data[season])

            # Convert percentile to score on same scale as ranked scores
            # Scale percentile (0-100) to rank value scale (1-25)
            # This makes previous season score comparable to ranked score
            season_score = (
                1 + (percentile / 100.0) * 24
            )  # Scale to 1-25 like rank values
            best_score = max(best_score, season_score)

        return best_score
    else:
        # Single season object (backwards compatibility)
        season = previous_stats.get("season", "").upper()
        rating = previous_stats.get("adjusted_rating", 0.0)

        if not season or rating <= 0:
            return 0.0

        # Get season rating distributions
        season_data = get_season_rating_percentiles()

        if season not in season_data:
            return 0.0  # Unknown season

        # Calculate percentile
        percentile = calculate_rating_percentile(rating, season_data[season])

        # Convert percentile to score on same scale as ranked scores
        # Scale percentile (0-100) to rank value scale (1-25)
        season_score = 1 + (percentile / 100.0) * 24  # Scale to 1-25 like rank values

        return season_score


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

    # Always export player scores by default
    output_file = os.path.join(script_dir, "player_scores.json")
    export_player_scores(players_data, config, output_file)

    # Build groups
    groups, excluded_groups = build_groups(players_data, config)
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
        return

    chosen_groups = [groups[i] for i in best_subset]
    best_size = sum(g["size"] for g in chosen_groups)
    num_teams = best_size // 5
    print(f"Found optimal subset with {best_size} players (divisible by 5).")
    print(f"Creating {num_teams} balanced teams...")

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
            # Show role composition
            team_roles = []
            for group in team["groups"]:
                team_roles.extend(group.get("roles", []))
            role_balance = get_role_balance_score(team_roles)
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
        print(f"\nLeftover groups (subs) totaling {leftover_players} players:")
        for grp in sorted(leftover_groups, key=lambda g: g["sum_score"], reverse=True):
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
    else:
        print("\nNo leftover groups - all players assigned to teams!")

    # Display excluded groups information
    display_excluded_groups(excluded_groups, config, players_data)


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

    for group in excluded_groups:
        mode = config.get("mode", "basic")

        print(
            f"\nGroup {group['group_id']} ({group['size']} players) - Total Score: {group['sum_score']:.2f}"
        )

        # Show detailed role info for each player in advanced mode
        if mode == "advanced":
            member_roles = []
            for member in group["members"]:
                role_display = get_player_role_display(member, players_data)
                member_roles.append(f"{member}({role_display})")
            members_str = ", ".join(member_roles)
        else:
            members_str = ", ".join(group["members"])

        print(f"  Members: {members_str}")


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
