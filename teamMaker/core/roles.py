"""Role coverage scoring for Team Maker.

A team wants one player on each of the four main roles (the fifth player is a
free pick). The role penalty of a team is the minimum-cost assignment of
distinct players to the main roles:

- 0 when the player's primary role (first listed) is the role,
- 0.5 when the role is another listed role, the player is flex, or the player
  has no usable role information,
- 1 otherwise.

Costs are kept as integer halves internally so sums are exact.
"""

import itertools

MAIN_ROLES = ("duelist", "initiator", "controller", "sentinel")
FLEX = "flex"

COST_PRIMARY = 0
COST_SECONDARY = 1  # half a penalty point
COST_OTHER = 2  # one penalty point

_CACHE_LIMIT = 300000

_SIG_IDS = {}
_ID_SIGS = []
_TEAM_COST_CACHE = {}
_PERMUTATIONS = {}


def normalize_roles(role):
    """Return the roles of a player as a lowercase list (primary first).

    Args:
        role: A role string, a list of role strings, or None.

    Returns:
        List of lowercase role names (empty if nothing usable).
    """
    if role is None:
        return []
    if isinstance(role, str):
        role = [role]
    return [str(r).strip().lower() for r in role if r and str(r).strip()]


def role_signature(role):
    """Return the cost vector (in halves) of a player for each main role.

    Args:
        role: Role string or list of role strings, primary first.

    Returns:
        Tuple of four ints in the order of MAIN_ROLES.
    """
    roles = normalize_roles(role)
    known = [r for r in roles if r in MAIN_ROLES]
    if not known or FLEX in roles:
        # Unknown or flex: acceptable everywhere at a small cost.
        return tuple(COST_SECONDARY for _ in MAIN_ROLES)
    primary = roles[0]
    sig = []
    for main in MAIN_ROLES:
        if main == primary:
            sig.append(COST_PRIMARY)
        elif main in roles:
            sig.append(COST_SECONDARY)
        else:
            sig.append(COST_OTHER)
    return tuple(sig)


def signature_id(sig):
    """Intern a role signature and return its small integer id."""
    sid = _SIG_IDS.get(sig)
    if sid is None:
        sid = len(_ID_SIGS)
        _SIG_IDS[sig] = sid
        _ID_SIGS.append(sig)
    return sid


def _permutations(n):
    """Return all injective role -> player index tuples for n players."""
    perms = _PERMUTATIONS.get(n)
    if perms is None:
        perms = list(itertools.permutations(range(n), min(n, len(MAIN_ROLES))))
        _PERMUTATIONS[n] = perms
    return perms


def _solve(sigs):
    """Brute-force the minimum-cost assignment.

    Args:
        sigs: List of role signatures, one per player.

    Returns:
        Tuple (cost_halves, assignment) where assignment[i] is the player
        index holding MAIN_ROLES[i] (None for a role nobody can hold).
    """
    n = len(sigs)
    best_cost = None
    best_perm = None
    for perm in _permutations(n):
        cost = 0
        for role_index, player in enumerate(perm):
            cost += sigs[player][role_index]
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
    if best_perm is None:
        return COST_OTHER * len(MAIN_ROLES), ()
    missing = len(MAIN_ROLES) - len(best_perm)
    return (best_cost or 0) + missing * COST_OTHER, best_perm


def team_cost_halves(sig_ids):
    """Cached role cost (in halves) of a team, keyed by its signature ids.

    Args:
        sig_ids: Sorted tuple of signature ids (see signature_id).

    Returns:
        Integer penalty in halves (2 = one penalty point).
    """
    cost = _TEAM_COST_CACHE.get(sig_ids)
    if cost is None:
        if len(_TEAM_COST_CACHE) >= _CACHE_LIMIT:
            _TEAM_COST_CACHE.clear()
        cost = _solve([_ID_SIGS[i] for i in sig_ids])[0]
        _TEAM_COST_CACHE[sig_ids] = cost
    return cost


def best_role_assignment(player_roles):
    """Assign the main roles to the players of one team at minimum cost.

    Args:
        player_roles: One role value per player (string or list, primary
            first).

    Returns:
        Tuple (penalty, assigned_roles): penalty is the total cost as a float
        (0.5 steps); assigned_roles has one entry per player, either a main
        role or, for the player not needed for a main role, their primary role
        (or "flex").
    """
    sigs = [role_signature(r) for r in player_roles]
    cost, perm = _solve(sigs)
    assigned: list = [None] * len(player_roles)
    for role_index, player in enumerate(perm):
        assigned[player] = MAIN_ROLES[role_index]
    for i, value in enumerate(assigned):
        if value is None:
            roles = normalize_roles(player_roles[i])
            assigned[i] = roles[0] if roles else FLEX
    return cost / 2.0, assigned
