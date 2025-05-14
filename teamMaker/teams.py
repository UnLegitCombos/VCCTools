import json
import math
import time
import os
from tqdm import tqdm
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
            "max_states": 100000000,
            "max_time": 1200,
            "target_team_score": 157.0,
            "num_random_restarts": 10,  # Default number of random restarts
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


def arrangement_backtracking_with_progress(groups_subset):
    """
    Full backtracking to place groups into balanced teams with a progress bar.
    Uses random restarts to escape local minima and increase the chance of finding the most balanced arrangement.
    Each random restart shuffles the group order and runs the backtracking search again, keeping the best result found.
    The number of restarts is configurable in config.json as 'num_random_restarts'.

    Args:
        groups_subset: List of group objects to arrange into teams

    Returns:
        tuple: (best score difference, best assignment)
    """
    config = load_config()
    num_random_restarts = config.get("num_random_restarts", 10)
    total_players = sum(g["size"] for g in groups_subset)
    num_teams = total_players // 5

    best_overall_diff = math.inf
    best_overall_stdev = math.inf
    best_overall_assignment = None

    for restart in range(num_random_restarts):
        best_diff = math.inf
        best_assignment = None
        best_stdev = math.inf

        # Shuffle group order for diversity (except first run)
        if restart == 0:
            sorted_groups = sorted(
                enumerate(groups_subset),
                key=lambda x: (x[1]["size"], x[1]["sum_score"]),
                reverse=True,
            )
        else:
            import random

            sorted_groups = list(enumerate(groups_subset))
            random.shuffle(sorted_groups)

        orig_to_sorted = {
            orig_idx: sorted_idx
            for sorted_idx, (orig_idx, _) in enumerate(sorted_groups)
        }
        groups_shuffled = [g for _, g in sorted_groups]

        team_scores = [0.0] * num_teams
        team_sizes = [0] * num_teams
        assignment = [-1] * len(groups_shuffled)

        max_states = config["max_states"]
        max_time = config["max_time"]

        progress_bar = tqdm(
            total=100, desc=f"Restart {restart+1}/{num_random_restarts}"
        )
        progress_update_interval = max(1, max_states // 100)
        visited_count = 0
        total_score = sum(g["sum_score"] for g in groups_shuffled)
        ideal_team_score = total_score / num_teams
        all_valid_arrangements = []
        start_time = time.time()

        def backtrack(i):
            nonlocal best_diff, best_assignment, visited_count
            nonlocal best_stdev
            visited_count += 1
            if visited_count % progress_update_interval == 0:
                progress_bar.update(1)
            if visited_count >= max_states or (time.time() - start_time > max_time):
                return
            if i == len(groups_shuffled):
                if all(size == 5 for size in team_sizes):
                    diff = max(team_scores) - min(team_scores)
                    avg = sum(team_scores) / len(team_scores)
                    stdev = math.sqrt(
                        sum((s - avg) ** 2 for s in team_scores) / len(team_scores)
                    )
                    all_valid_arrangements.append((diff, stdev, assignment[:]))
                    if (diff < best_diff) or (diff == best_diff and stdev < best_stdev):
                        best_diff = diff
                        best_stdev = stdev
                        best_assignment = assignment[:]
                return False
            grp = groups_shuffled[i]
            grp_score = grp["sum_score"]
            grp_size = grp["size"]
            team_options = []
            for t in range(num_teams):
                if team_sizes[t] + grp_size <= 5:
                    new_score = team_scores[t] + grp_score
                    distance = abs(new_score - ideal_team_score)
                    team_options.append((t, distance))
            team_options.sort(key=lambda x: x[1])
            for t, _ in team_options:
                team_sizes[t] += grp_size
                team_scores[t] += grp_score
                assignment[i] = t
                backtrack(i + 1)
                team_sizes[t] -= grp_size
                team_scores[t] -= grp_score
                assignment[i] = -1
            return False

        backtrack(0)
        progress_bar.close()
        if best_assignment is None and all_valid_arrangements:
            best_diff, best_stdev, best_assignment = min(
                all_valid_arrangements, key=lambda x: (x[0], x[1])
            )
        if best_assignment and (
            (best_diff < best_overall_diff)
            or (best_diff == best_overall_diff and best_stdev < best_overall_stdev)
        ):
            best_overall_diff = best_diff
            best_overall_stdev = best_stdev
            best_overall_assignment = best_assignment
    # Convert back to original indices
    if best_overall_assignment:
        orig_assignment = [-1] * len(groups_subset)
        for sorted_idx, team in enumerate(best_overall_assignment):
            orig_idx = next(o for o, s in orig_to_sorted.items() if s == sorted_idx)
            orig_assignment[orig_idx] = team
        return best_overall_diff, orig_assignment
    return best_overall_diff, best_overall_assignment


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
    diff, assignment = arrangement_backtracking_with_progress(
        groups_subset, num_random_restarts=10
    )
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
