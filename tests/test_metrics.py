import pytest

from evalh import metrics


class TestFactCoverage:
    def test_all_facts_present(self) -> None:
        assert metrics.fact_coverage("It costs 6 USD per month.", [["6 USD", "$6"], ["per month"]]) == 1.0

    def test_alternative_phrasing_counts(self) -> None:
        assert (
            metrics.fact_coverage("Only $6, billed monthly.", [["6 USD", "$6"], ["per month", "monthly"]])
            == 1.0
        )

    def test_partial(self) -> None:
        assert metrics.fact_coverage("It costs 8 USD per month.", [["6 USD"], ["per month"]]) == 0.5

    def test_no_substring_false_positive(self) -> None:
        # "16 USD" must not satisfy "6 USD"
        assert metrics.fact_coverage("It costs 16 USD.", [["6 USD"]]) == 0.0

    def test_case_and_punctuation_insensitive(self) -> None:
        assert (
            metrics.fact_coverage("Encrypted with aes-256, and TLS 1.3!", [["AES-256"], ["tls 1.3"]]) == 1.0
        )

    def test_no_facts_required(self) -> None:
        assert metrics.fact_coverage("anything", []) == 1.0


def test_forbidden_hits() -> None:
    assert metrics.forbidden_hits("You get a full refund.", ["full refund", "sms"]) == ["full refund"]
    assert metrics.forbidden_hits("Use an authenticator app.", ["sms"]) == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("I don't know, that is not covered by the documentation.", True),
        ("I can only help with questions about using Inkwell Notes.", True),
        ("I cannot answer that.", True),
        ("The Plus plan costs 6 USD per month.", False),
        ("You do not need to know the price in advance.", False),
    ],
)
def test_is_refusal(answer: str, expected: bool) -> None:
    assert metrics.is_refusal(answer) is expected


def test_citation_validity() -> None:
    assert metrics.citation_validity("See [refunds].", ["refunds", "billing-plans"]) == 1.0
    assert metrics.citation_validity("See [refunds] and [sharing].", ["refunds"]) == 0.5
    assert metrics.citation_validity("No citations here.", ["refunds"]) == 1.0


class TestGroundedness:
    CONTEXT = "Refunds go back to the original payment method within 5 to 10 business days."

    def test_supported_sentence(self) -> None:
        answer = "Refunds go back to the original payment method within 5 to 10 business days [refunds]."
        assert metrics.groundedness(answer, self.CONTEXT) == 1.0

    def test_invented_sentence_lowers_score(self) -> None:
        answer = (
            "Refunds go back to the original payment method within 5 to 10 business days. "
            "Premium customers receive instant cryptocurrency reimbursement."
        )
        assert metrics.groundedness(answer, self.CONTEXT) == 0.5

    def test_empty_answer(self) -> None:
        assert metrics.groundedness("", self.CONTEXT) == 1.0


class TestPii:
    def test_email(self) -> None:
        assert metrics.pii_findings("Mail jane.doe@example.com for help") == ["email:jane.doe@example.com"]

    def test_luhn_valid_card(self) -> None:
        assert metrics.pii_findings("Card 4111 1111 1111 1111 on file") == ["card:1111"]

    def test_luhn_invalid_number_is_not_a_card(self) -> None:
        assert not any(f.startswith("card:") for f in metrics.pii_findings("Order 4111 1111 1111 1112"))

    def test_phone(self) -> None:
        assert metrics.pii_findings("Call +1 415 555 0134 today") == ["phone:0134"]

    def test_clean_text(self) -> None:
        assert metrics.pii_findings("Refunds take 5 to 10 business days and TLS 1.3 is used.") == []


def test_recall_and_reciprocal_rank() -> None:
    assert metrics.recall_at_k(["a", "b", "c"], ["b"]) == 1.0
    assert metrics.recall_at_k(["a", "c"], ["b", "c"]) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert metrics.reciprocal_rank(["a"], ["z"]) == 0.0
    assert metrics.recall_at_k(["a"], []) == 1.0
