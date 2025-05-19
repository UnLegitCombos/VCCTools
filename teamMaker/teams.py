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

Usage:
    python teams.py
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
            "max_time": 120,  # 2 minutes max
            "annealing_iterations": 100000,
            "initial_temperature": 100.0,
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
        group_id = pinfo.get("group_id", 0)
        group_map[group_id].append(player_name)

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

    # Otherwise, we need to find the largest subset that is divisible by 5
    best_size = 0
    best_subset = []

    def dfs(idx, current_size, current_subset):
        nonlocal best_size, best_subset

        # Base case: we've examined all groups
        if idx == n:
            if current_size % 5 == 0 and current_size > best_size:
                best_size = current_size
                best_subset = current_subset[:]
            return

        # Skip this group
        dfs(idx + 1, current_size, current_subset)

        # Take this group
        current_subset.append(idx)
        dfs(idx + 1, current_size + groups[idx]["size"], current_subset)
        current_subset.pop()

    dfs(0, 0, [])
    return best_subset


def evaluate_teams(teams):
    """
    Evaluate the quality of team assignments using multiple metrics.

    Args:
        teams: A list of team objects with team_score and groups

    Returns:
        tuple: (score_range, standard_deviation)
    """
    team_scores = [team["team_score"] for team in teams]

    if not team_scores:
        return float("inf"), float("inf")

    # Calculate score range (difference between highest and lowest team score)
    score_range = max(team_scores) - min(team_scores)

    # Calculate standard deviation
    mean_score = sum(team_scores) / len(team_scores)
    variance = sum((score - mean_score) ** 2 for score in team_scores) / len(
        team_scores
    )
    std_dev = math.sqrt(variance)

    return score_range, std_dev


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
    current_range, current_stdev = evaluate_teams(current_solution)
    best_solution = current_solution
    best_range = current_range
    best_stdev = current_stdev  # Simulated annealing parameters
    temperature = config.get("initial_temperature", 100.0)
    cooling_rate = config.get("cooling_rate", 0.99)
    max_iterations = config.get("annealing_iterations", 100000)

    progress_bar = tqdm(
        total=max_iterations, desc="Optimizing teams"
    )  # Track iterations without improvement for early stopping
    iterations_without_improvement = 0
    max_no_improvement = config.get("max_no_improvement", 10000)

    start_time = time.time()
    max_time = config.get("max_time", 120)  # 2 minutes default

    for iteration in range(max_iterations):
        progress_bar.update(1)  # Check time limit
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
        neighbor_range, neighbor_stdev = evaluate_teams(neighbor)

        # Calculate acceptance probability
        # We want to minimize both range and standard deviation,
        # with range being the primary concern

        # Energy difference (lower is better)
        energy_diff = (neighbor_range - current_range) + 0.2 * (
            neighbor_stdev - current_stdev
        )

        # Accept if better, or with probability based on temperature if worse
        if energy_diff <= 0 or random.random() < math.exp(-energy_diff / temperature):
            current_solution = neighbor
            current_range = neighbor_range
            current_stdev = neighbor_stdev

            # Update best solution if this is better
            if neighbor_range < best_range or (
                neighbor_range == best_range and neighbor_stdev < best_stdev
            ):
                best_solution = neighbor
                best_range = neighbor_range
                best_stdev = neighbor_stdev
                iterations_without_improvement = (
                    0  # If we find a very good solution, print it and exit early
                )
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
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load configuration
    config = load_config()

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

    # Build groups
    groups = build_groups(players_data, config)
    total_players = sum(g["size"] for g in groups)
    print(f"Total players = {total_players} in {len(groups)} groups.")

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
    min_score, max_score, score_range, std_dev = calculate_statistics(best_teams)

    # Print results
    print("\n==== Best Team Arrangement ====")
    print(f"Using {best_size} players to form {len(best_teams)} teams of 5.")
    print(f"Score difference across teams = {score_range:.2f}")
    print(f"Standard deviation of team scores = {std_dev:.2f}")

    # Print team compositions
    team_scores = []
    for i, team in enumerate(best_teams, start=1):
        team_scores.append(team["team_score"])
        print(f"\nTEAM {i}: total score = {team['team_score']:.2f}")
        for grp in sorted(team["groups"], key=lambda g: g["sum_score"], reverse=True):
            print(
                f"  Group {grp['group_id']} (size={grp['size']}, score={grp['sum_score']:.2f}) => {', '.join(grp['members'])}"
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
            print(
                f"  Group {grp['group_id']} (size={grp['size']}, score={grp['sum_score']:.2f}) => {', '.join(grp['members'])}"
            )
    else:
        print("\nNo leftover groups - all players assigned to teams!")


if __name__ == "__main__":
    main()
