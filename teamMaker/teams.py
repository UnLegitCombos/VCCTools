import json
import math
import time
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count


"""
Team Formation Optimizer

This script optimizes the formation of balanced teams from groups of players.
It takes into account player ranks and uses configuration from config.json.

Usage:
    python team_formation.py
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
            return json.load(f)
    except FileNotFoundError:
        print("Config file not found. Using default configuration.")
        return {
            "players_file": "players.json",
            "use_tracker": True,
            "weight_current": 0.8,
            "weight_peak": 0.2,
            "weight_current_tracker": 0.1,
            "weight_peak_tracker": 0.05,
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
                "Radiant": 25,
            },
        }


def rank_to_numeric(rank_str, rank_values):
    """Convert rank string to numeric value"""
    return rank_values.get(rank_str, 0)


def compute_player_score(player_info, config):
    """
    Compute a player's score based on ranks and optionally tracker values

    Args:
        player_info: Dict containing player data
        config: Dict containing weight configuration

    Returns:
        float: Player's calculated score
    """
    current_val = rank_to_numeric(player_info["current_rank"], config["rank_values"])
    peak_val = rank_to_numeric(player_info["peak_rank"], config["rank_values"])

    # Base score from ranks
    score = config["weight_current"] * current_val + config["weight_peak"] * peak_val

    # Add tracker scores if enabled
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

    return score


def build_groups(players_data, config):
    """
    Build groups from player data

    Args:
        players_data: Dict containing player information
        config: Dict containing weight configuration

    Returns:
        list: List of group objects
    """
    group_map = defaultdict(list)
    for player_name, pinfo in players_data.items():
        group_map[pinfo["group_id"]].append(player_name)

    group_list = []
    for g_id, members in group_map.items():
        group_size = len(members)
        if group_size not in (1, 2, 3):
            raise ValueError(
                f"Group {g_id} has {group_size} players. Allowed sizes: 1, 2, or 3."
            )
        total_score = sum(
            compute_player_score(players_data[m], config) for m in members
        )
        group_list.append(
            {
                "group_id": g_id,
                "members": members,
                "sum_score": total_score,
                "size": group_size,
            }
        )
    return group_list


def find_valid_subsets(groups):
    """
    Find all subsets whose total size is a multiple of 5,
    returning only those that maximize total player count

    Args:
        groups: List of group objects

    Returns:
        list: List of subset indices that maximize player count in multiples of 5
    """
    n = len(groups)
    best_size = 0
    valid_subsets = []

    def dfs(idx, current_indices, current_size):
        nonlocal best_size, valid_subsets
        if idx == n:
            if current_size % 5 == 0 and current_size > 0:
                if current_size > best_size:
                    best_size = current_size
                    valid_subsets = [current_indices[:]]
                elif current_size == best_size:
                    valid_subsets.append(current_indices[:])
            return
        # skip group idx
        dfs(idx + 1, current_indices, current_size)
        # take group idx
        current_indices.append(idx)
        dfs(idx + 1, current_indices, current_size + groups[idx]["size"])
        current_indices.pop()

    dfs(0, [], 0)

    # Filter to only keep subsets that use maximum possible players
    if best_size == 0:
        return []

    max_subsets = []
    for s in valid_subsets:
        sum_size = sum(groups[i]["size"] for i in s)
        if sum_size == best_size:
            max_subsets.append(s)
    return max_subsets


def arrangement_backtracking(groups_subset):
    """
    Full backtracking to place groups into balanced teams

    Args:
        groups_subset: List of group objects to arrange into teams

    Returns:
        tuple: (best score difference, best assignment)
    """
    total_players = sum(g["size"] for g in groups_subset)
    num_teams = total_players // 5

    best_diff = math.inf
    best_assignment = None

    team_scores = [0.0] * num_teams
    team_sizes = [0] * num_teams
    assignment = [-1] * len(groups_subset)

    def backtrack(i):
        nonlocal best_diff, best_assignment
        if i == len(groups_subset):
            # All groups assigned, evaluate team balance
            if min(team_sizes) == max(team_sizes) == 5:  # All teams complete
                diff = max(team_scores) - min(team_scores)
                if diff < best_diff:
                    best_diff = diff
                    best_assignment = assignment[:]
            return

        grp = groups_subset[i]
        for t in range(num_teams):
            # If group fits in team
            if team_sizes[t] + grp["size"] <= 5:
                # Place group in team
                team_sizes[t] += grp["size"]
                team_scores[t] += grp["sum_score"]
                assignment[i] = t

                # Pruning: if current diff already exceeds best_diff, skip
                current_max = max(
                    score for score, size in zip(team_scores, team_sizes) if size > 0
                )
                current_min = min(
                    score for score, size in zip(team_scores, team_sizes) if size > 0
                )
                if current_max - current_min < best_diff:
                    backtrack(i + 1)

                # Undo assignment and try next team
                team_sizes[t] -= grp["size"]
                team_scores[t] -= grp["sum_score"]
                assignment[i] = -1

    backtrack(0)
    return best_diff, best_assignment


def evaluate_subset(args):
    """
    Parallel worker function to evaluate a subset

    Args:
        args: Tuple containing (subset_indices, all_groups)

    Returns:
        tuple: (score difference, assignment, subset indices)
    """
    subset_indices, all_groups = args
    groups_subset = [all_groups[i] for i in subset_indices]

    diff, assignment = arrangement_backtracking(groups_subset)
    return diff, assignment, subset_indices


def reconstruct_teams(groups_subset, assignment):
    """
    Reconstruct teams from assignment

    Args:
        groups_subset: List of group objects
        assignment: List mapping group index to team index

    Returns:
        list: List of team objects
    """
    total_players = sum(g["size"] for g in groups_subset)
    num_teams = total_players // 5

    teams = []
    for _ in range(num_teams):
        teams.append({"team_score": 0.0, "groups": []})

    for i, grp in enumerate(groups_subset):
        t = assignment[i]
        teams[t]["team_score"] += grp["sum_score"]
        teams[t]["groups"].append(grp)

    return teams


def calculate_statistics(teams):
    """
    Calculate statistics for team scores

    Args:
        teams: List of team objects

    Returns:
        tuple: (min score, max score, range, standard deviation)
    """
    scores = [team["team_score"] for team in teams]

    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    # Calculate standard deviation
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    std_dev = math.sqrt(variance)

    return min_score, max_score, score_range, std_dev


def main():
    """
    Main program logic
    """
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))

    config = load_config()
    players_file = config.get("players_file", "players.json")

    # Resolve players file path relative to script directory
    players_path = os.path.join(script_dir, players_file)
    print(f"Loading player data from {players_path}")

    # Load player data
    with open(players_path, "r") as f:
        players_data = json.load(f)

    # Build groups
    groups = build_groups(players_data, config)
    total_players = sum(g["size"] for g in groups)
    print(f"Total players = {total_players} in {len(groups)} groups.")

    # Find valid subsets
    max_subsets = find_valid_subsets(groups)
    if not max_subsets:
        print("No valid subset of groups can form a complete 5-person team.")
        return

    best_size = sum(groups[i]["size"] for i in max_subsets[0])
    print(
        f"Found {len(max_subsets)} subsets that use {best_size} players (multiples of 5)."
    )

    # Evaluate subsets in parallel
    tasks = [(subset_indices, groups) for subset_indices in max_subsets]

    num_workers = min(cpu_count(), 8)
    print(f"Starting parallel processing with {num_workers} workers...")

    start_time = time.time()
    with Pool(processes=num_workers) as pool:
        results = pool.map(evaluate_subset, tasks)
    end_time = time.time()
    print(f"Parallel evaluation completed in {end_time - start_time:.2f} seconds.")

    # Find best arrangement
    overall_best_diff = math.inf
    overall_best_assignment = None
    overall_best_subset = None

    for diff, assignment, subset_indices in results:
        if assignment is not None and diff < overall_best_diff:
            overall_best_diff = diff
            overall_best_assignment = assignment
            overall_best_subset = subset_indices

    if overall_best_subset is None:
        print("No valid arrangement found!")
        return

    # Reconstruct teams
    chosen_groups = [groups[i] for i in overall_best_subset]
    teams = reconstruct_teams(chosen_groups, overall_best_assignment)

    # Identify leftover groups
    leftover_groups = [g for i, g in enumerate(groups) if i not in overall_best_subset]
    leftover_players = sum(g["size"] for g in leftover_groups)

    # Calculate statistics
    min_score, max_score, score_range, std_dev = calculate_statistics(teams)

    # Print results
    print("\n==== Best Team Arrangement ====")
    print(f"Using {best_size} players to form {len(teams)} teams of 5.")
    print(f"Score difference across teams = {score_range:.2f}")
    print(f"Standard deviation of team scores = {std_dev:.2f}")

    # Print team compositions
    for i, team in enumerate(teams, start=1):
        print(f"\nTEAM {i}: total score = {team['team_score']:.2f}")
        for grp in sorted(team["groups"], key=lambda g: g["sum_score"], reverse=True):
            print(
                f"  Group {grp['group_id']} (size={grp['size']}, score={grp['sum_score']:.2f}) => {', '.join(grp['members'])}"
            )

    # Print leftover players
    if leftover_groups:
        print(f"\nLeftover groups (subs) totaling {leftover_players} players:")
        for grp in sorted(leftover_groups, key=lambda g: g["sum_score"], reverse=True):
            print(
                f"  Group {grp['group_id']} (size={grp['size']}, score={grp['sum_score']:.2f}) => {', '.join(grp['members'])}"
            )
    else:
        print("\nNo leftover groups - all players assigned to teams!")


if __name__ == "__main__":
    main()
