from resolvelm.evaluation.demos import demo_incidents
from resolvelm.inference.prompting import request_to_prompt
from resolvelm.inference.repair import parse_analysis_with_retry
from resolvelm.inference.rule_backend import RuleBasedBackend
from resolvelm.schemas.api import AnalyzeRequest


def test_all_demos_produce_schema_valid_json() -> None:
    backend = RuleBasedBackend()
    for demo in demo_incidents():
        req = AnalyzeRequest.model_validate(demo["request"])
        raw = backend.generate_sync(request_to_prompt(req))
        parsed = parse_analysis_with_retry(raw)
        assert parsed.ok, demo["demo_id"]
