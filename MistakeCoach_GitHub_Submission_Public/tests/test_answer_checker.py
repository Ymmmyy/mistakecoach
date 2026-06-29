from src.tutor.answer_checker import check_answer


def test_numeric_answer_correct():
    result = check_answer("87.5", "87.5", "numeric")
    assert result.correct


def test_numeric_answer_with_text():
    result = check_answer("The answer is 87.50 points", "87.5", "numeric")
    assert result.correct


def test_multiple_choice_answer_correct():
    result = check_answer("A)", "A", "multiple_choice")
    assert result.correct


def test_text_answer_incorrect():
    result = check_answer("median", "mean", "text")
    assert not result.correct
