from app.services.strategy_rules import (
    STRATEGY_RULES,
    get_strategy_rule,
)


def test_all_transform_strategies_exist():
    expected = {
        "COMPRESS",
        "STRUCTURE",
        "REFINE",
        "CREATIVE_EXPAND",
    }

    assert set(STRATEGY_RULES) == expected


def test_compress_rule_prevents_added_content():
    rule = get_strategy_rule("COMPRESS")

    assert "Remove filler" in rule
    assert "Do not add content" in rule
    assert "Preserve" in rule


def test_structure_rule_prevents_invention():
    rule = get_strategy_rule("STRUCTURE")

    assert "minimal restructuring" in rule
    assert "Do not add requirements" in rule
    assert "Do not add" in rule
    assert "categorization" in rule
    assert "Preserve" in rule


def test_refine_rule_is_light_touch():
    rule = get_strategy_rule("REFINE")

    assert "minor wording" in rule
    assert "Preserve the existing sentence structure" in rule
    assert "Do not convert prose into lists" in rule
    assert "Do not add, remove, or reorganize content" in rule


def test_creative_rule_prevents_invented_constraints():
    rule = get_strategy_rule("CREATIVE_EXPAND")

    assert "minimum missing context" in rule
    assert "Do not invent requirements" in rule
    assert "evaluation criteria" in rule