import json

from futureshow.tool import tool_polymarket_trade as trade


def _seed_positions(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": "2026-10-24T00:00:00Z",
        "id": 1,
        "positions": {"CASH": 10.0, "market:Yes": 2.0, "market:No": 3.0},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path.read_text(encoding="utf-8")


def test_settle_does_not_clear_closed_market_without_winner(monkeypatch, tmp_path):
    position_file = tmp_path / "position.jsonl"
    original = _seed_positions(position_file)
    monkeypatch.setattr(trade, "_position_file", lambda signature: position_file)
    monkeypatch.setattr(
        trade,
        "get_market_info_fn",
        lambda slug: {
            "closed": True,
            "status": "closed",
            "tokens": [
                {"outcome": "Yes", "winner": False},
                {"outcome": "No", "winner": False},
            ],
        },
    )

    result = trade.settle("market", "agent")

    assert result == {"error": "market resolution unavailable"}
    assert position_file.read_text(encoding="utf-8") == original


def test_settle_pays_winning_outcome_from_resolution(monkeypatch, tmp_path):
    position_file = tmp_path / "position.jsonl"
    _seed_positions(position_file)
    monkeypatch.setattr(trade, "_position_file", lambda signature: position_file)
    monkeypatch.setattr(
        trade,
        "get_market_info_fn",
        lambda slug: {
            "closed": True,
            "status": "resolved",
            "tokens": [
                {"outcome": "Yes", "winner": True},
                {"outcome": "No", "winner": False},
            ],
        },
    )

    result = trade.settle("market", "agent")

    assert result["ok"] is True
    assert result["realized"] == 2.0
    assert result["positions"] == {"CASH": 12.0, "market:Yes": 0.0, "market:No": 0.0}
