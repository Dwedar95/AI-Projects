import json

import pytest

from llmops_gate.cost import PricingTable
from llmops_gate.datasets import INTENTS, intent_keywords
from llmops_gate.evaluation import (
    OnnxClassifierPredictor,
    PromptLLMPredictor,
    evaluate,
    macro_f1,
)
from llmops_gate.providers import Completion, KeywordMockLLM

PRICING = PricingTable.example()


def _template(descriptions: dict[str, str]) -> str:
    labels = "\n".join(f"- {k}: {v}" for k, v in descriptions.items())
    return (
        f"Classify the ticket.\nLabels:\n{labels}\nAnswer with the label only.\nTicket: {{ticket}}"
    )


FULL = {k: ", ".join(v) for k, v in intent_keywords().items()}


def test_macro_f1_on_a_hand_computed_case():
    truth = ["a", "a", "b", "b"]
    predicted = ["a", "b", "b", "b"]
    scores = macro_f1(truth, predicted, ["a", "b"])
    assert scores["a"] == pytest.approx(2 / 3) and scores["b"] == pytest.approx(0.8)
    assert scores["macro"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert macro_f1(["a"], ["b"], ["a"])["a"] == 0.0


def test_onnx_model_metadata_sidecar(good_model):
    meta = json.loads((good_model.parent / "intent.onnx.json").read_text())
    assert meta["labels"] == INTENTS and meta["embed_dim"] == 256 and meta["train_size"] == 2000


def test_onnx_classifier_evaluation(good_model, benchmark):
    predictor = OnnxClassifierPredictor(good_model, INTENTS, 256)
    result = evaluate(predictor, benchmark, PRICING)
    assert result.n_examples == 400 and not result.latency_simulated
    assert result.metrics["accuracy"] > 0.8 and result.metrics["macro_f1"] > 0.8
    assert 0 < result.metrics["latency_p50_ms"] <= result.metrics["latency_p95_ms"] < 50
    assert set(result.per_class_f1) == set(INTENTS)
    assert len(result.errors) == 5 and result.errors[0]["expected"] != result.errors[0]["predicted"]
    single = predictor.predict(["my invoice shows a wrong charge on my card payment"])[0]
    assert single.label == "billing" and 0.2 < single.score <= 1.0


def test_model_trained_on_little_data_is_measurably_worse(good_model, weak_model, benchmark):
    good = evaluate(OnnxClassifierPredictor(good_model, INTENTS, 256), benchmark, PRICING)
    weak = evaluate(OnnxClassifierPredictor(weak_model, INTENTS, 256), benchmark, PRICING)
    assert weak.metrics["accuracy"] < good.metrics["accuracy"] - 0.05


def test_prompt_content_changes_mock_accuracy_and_cost(benchmark):
    def run(descriptions):
        predictor = PromptLLMPredictor(
            _template(descriptions), KeywordMockLLM(), "small-llm", INTENTS
        )
        return evaluate(predictor, benchmark, PRICING)

    full = run(FULL)
    short = run({k: ", ".join(v[:3]) for k, v in intent_keywords().items()})
    assert full.latency_simulated
    assert full.metrics["accuracy"] > short.metrics["accuracy"] + 0.1
    assert full.metrics["cost_per_1k_requests_usd"] > short.metrics["cost_per_1k_requests_usd"]
    assert full.metrics["latency_p95_ms"] > short.metrics["latency_p95_ms"]


def test_label_missing_from_prompt_is_never_predicted(benchmark):
    without_returns = {k: v for k, v in FULL.items() if k != "returns"}
    predictor = PromptLLMPredictor(
        _template(without_returns), KeywordMockLLM(), "small-llm", INTENTS
    )
    result = evaluate(predictor, benchmark, PRICING)
    assert result.per_class_f1["returns"] == 0.0


def test_out_of_vocabulary_llm_answers_become_unknown():
    class Rambling:
        def complete(self, prompt: str, model: str) -> Completion:
            return Completion(
                text="I think this is about money.",
                input_tokens=10,
                output_tokens=7,
                latency_ms=5.0,
                model=model,
            )

    predictor = PromptLLMPredictor("Ticket: {ticket}", Rambling(), "large-llm", INTENTS)
    prediction = predictor.predict(["refund please"])[0]
    assert prediction.label == "unknown" and prediction.score == 0.0
    result = evaluate(predictor, [{"text": "refund please", "label": "returns"}], PRICING)
    assert result.metrics["accuracy"] == 0.0
    assert result.metrics["cost_per_1k_requests_usd"] == pytest.approx(0.155)


def test_empty_benchmark_is_rejected(good_model):
    with pytest.raises(ValueError, match="empty"):
        evaluate(OnnxClassifierPredictor(good_model, INTENTS, 256), [], PRICING)
