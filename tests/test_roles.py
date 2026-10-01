from teamMaker.core import roles


def test_perfect_team_has_zero_penalty():
    penalty, assigned = roles.best_role_assignment(
        [["duelist"], ["initiator"], ["controller"], ["sentinel"], ["duelist"]]
    )
    assert penalty == 0.0
    assert set(roles.MAIN_ROLES) <= set(assigned)


def test_secondary_role_costs_half():
    penalty, _ = roles.best_role_assignment(
        [["duelist"], ["initiator"], ["controller"], ["duelist", "sentinel"], ["duelist"]]
    )
    assert penalty == 0.5


def test_missing_role_costs_one():
    penalty, _ = roles.best_role_assignment(
        [["duelist"], ["duelist"], ["controller"], ["sentinel"], ["duelist"]]
    )
    assert penalty == 1.0


def test_flex_and_unknown_cost_half():
    penalty, _ = roles.best_role_assignment(
        [["duelist"], ["initiator"], ["controller"], ["flex"], ["duelist"]]
    )
    assert penalty == 0.5
    penalty, _ = roles.best_role_assignment(
        [["duelist"], ["initiator"], ["controller"], [], ["duelist"]]
    )
    assert penalty == 0.5


def test_roles_are_assigned_to_distinct_players():
    penalty, assigned = roles.best_role_assignment([["sentinel"]] * 5)
    assert penalty == 3.0
    assert set(roles.MAIN_ROLES) <= set(assigned)
    assert len(assigned) == 5


def test_string_role_and_case():
    assert roles.role_signature("Duelist") == roles.role_signature(["duelist"])


def test_cache_matches_direct_solve():
    players = [["duelist", "sentinel"], ["initiator"], ["controller", "duelist"], ["flex"], ["sentinel"]]
    ids = tuple(sorted(roles.signature_id(roles.role_signature(r)) for r in players))
    assert roles.team_cost_halves(ids) / 2.0 == roles.best_role_assignment(players)[0]
    assert roles.team_cost_halves(ids) == roles.team_cost_halves(ids)


def test_penalty_independent_of_order():
    players = [["duelist"], ["duelist"], ["controller", "initiator"], ["sentinel"], ["flex"]]
    base = roles.best_role_assignment(players)[0]
    assert roles.best_role_assignment(list(reversed(players)))[0] == base
