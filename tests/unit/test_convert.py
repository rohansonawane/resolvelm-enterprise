from resolvelm.data.convert import incidents_to_dpo, incidents_to_sft
from resolvelm.data.generator import REJECTION_CATEGORIES, generate_corpus
from resolvelm.data.split import grouped_temporal_split


def test_sft_messages_have_three_roles() -> None:
    records = grouped_temporal_split(generate_corpus(6, 2))
    train = [r for r in records if r.split == "train"]
    rows = incidents_to_sft(train)
    assert rows
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert row["messages"][-1]["content"].startswith("{")


def test_dpo_skips_test_split() -> None:
    records = grouped_temporal_split(generate_corpus(8, 2))
    test = [r for r in records if r.split == "test"]
    assert incidents_to_dpo(test, pairs_per_record=2) == []
    train = [r for r in records if r.split == "train"]
    pairs = incidents_to_dpo(train, pairs_per_record=2)
    assert pairs
    assert all(p["split"] != "test" for p in pairs)
    assert all(p["rejection_category"] in REJECTION_CATEGORIES for p in pairs)
    assert all(p["label_origin"] == "synthetic_preference" for p in pairs)
    assert all(p["chosen"] != p["rejected"] for p in pairs)


def test_chat_template_not_invented_without_tokenizer() -> None:
    records = grouped_temporal_split(generate_corpus(4, 1))
    train = [r for r in records if r.split == "train"]
    row = incidents_to_sft(train)[0]
    assert "<|im_start|>" not in row["text"]
    assert "System:" in row["text"]
