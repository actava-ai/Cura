from cura_eval.mcq import decide, extract_letter


def test_boxed_wins_over_prose():
    assert extract_letter("I think B. Actually \\boxed{C}", "J") == "C"


def test_answer_marker():
    assert extract_letter("Final answer: D", "J") == "D"
    assert extract_letter("answer is (E)", "J") == "E"


def test_standalone_letter_last_occurrence():
    assert extract_letter("Could be A or maybe B", "J") == "B"


def test_max_letter_bounds_the_range():
    # a 5-option question must not accept G leaking from prose
    assert extract_letter("The answer is G", "E") is None
    assert extract_letter("A then G", "E") == "A"


def test_think_blocks_are_stripped():
    assert extract_letter("<think>surely C</think> \\boxed{A}", "J") == "A"
    assert extract_letter("<thinking>B B B</thinking>", "J") is None


def test_decide_exact_match():
    assert decide("\\boxed{B}", "b", "J")
    assert not decide("\\boxed{B}", "A", "J")
    assert not decide("", "A", "J")
