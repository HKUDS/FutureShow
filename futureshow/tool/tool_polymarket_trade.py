from typing import Dict, Any, Optional, List
import os
import json
import threading
from pathlib import Path
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv
from agents import function_tool
from futureshow.utils.general_tools import get_config_value, write_config_value
from datetime import datetime, timezone
from futureshow.tool.tool_polymarket_data import get_market_info_fn
load_dotenv()

BASE_URL = "https://clob.polymarket.com"

# ---- liquidity overlay config ----
OVERLAY_TOP_N = 5
OVERLAY_MIN_IMPACT = 0.10
OVERLAY_DECAY_SEC = 600
OVERLAY_DECAY_FACTOR = 0.9

_liquidity_lock = threading.Lock()


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [value]
    return [value]


def get_token_id_for_outcome(market: Dict[str, Any], outcome: str) -> str:
    outcomes = _to_list(market.get("outcomes"))
    tokens = _to_list(
        market.get("clobTokenIds")
        or market.get("clob_token_ids")
        or market.get("tokenIds")
        or market.get("clobTokens")
    )

    if (not tokens) and isinstance(market.get("tokens"), list):
        extracted_tokens: List[Any] = []
        for token in market["tokens"]:
            if isinstance(token, dict):
                extracted_tokens.append(token.get("token_id") or token.get("id"))
            else:
                extracted_tokens.append(token)
        tokens = extracted_tokens

    if len(tokens) < len(outcomes) and tokens:
        # Some markets may expose fewer token ids than outcomes; try to extend with None
        tokens = list(tokens) + [None] * (len(outcomes) - len(tokens))

    desired = str(outcome).lower()
    for idx, name in enumerate(outcomes):
        if str(name).lower() == desired:
            token_value = tokens[idx] if idx < len(tokens) else None
            if token_value is None and isinstance(market.get("tokens"), list):
                token_obj = market["tokens"][idx]
                if isinstance(token_obj, dict):
                    token_value = token_obj.get("token_id") or token_obj.get("id")
            if token_value is not None:
                return str(token_value)
            break
    raise ValueError("outcome not found")


def _resolve_market_and_token(market_slug: str, outcome: str) -> tuple[Dict[str, Any], str]:
    market = get_market_info_fn(market_slug)
    if not isinstance(market, dict) or ("error" in market):
        raise ValueError("market_slug not found")
    token_id = get_token_id_for_outcome(market, outcome)
    return market, token_id


def _get_order_book(token_id: str, signature: str) -> Dict[str, Any]:
    clob_host = os.getenv("CLOB_HOST", BASE_URL)
    response = requests.get(f"{clob_host}/book", params={"token_id": token_id}, timeout=10)
    response.raise_for_status()
    real_book = response.json()

    with _liquidity_lock:
        overlay = _load_liquidity(signature)
        book, updated_overlay = _apply_overlay_to_book(real_book, overlay, token_id)
        if updated_overlay is not overlay:
            overlay = updated_overlay
        _save_liquidity(signature, overlay)
    return book


def _sorted_levels(levels: Optional[List[Dict[str, Any]]], reverse: bool = False) -> List[Dict[str, Any]]:
    if not levels:
        return []

    def _key(level: Dict[str, Any]) -> float:
        try:
            return float(level["price"])
        except (KeyError, TypeError, ValueError):
            return float("inf") if not reverse else float("-inf")

    return sorted(levels, key=_key, reverse=reverse)


def _liquidity_file(signature: str) -> Path:
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "data" / "agent_data" / signature / "position" / "liquidity.json"


def _now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _load_liquidity(signature: str) -> Dict[str, Any]:
    f = _liquidity_file(signature)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_liquidity(signature: str, data: Dict[str, Any]) -> None:
    f = _liquidity_file(signature)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data), encoding="utf-8")


def _maybe_decay(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entry or OVERLAY_DECAY_SEC <= 0:
        return entry
    try:
        ts = entry.get("ts")
        if not ts:
            return entry
        last = datetime.fromisoformat(ts)
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        diff = (now - last).total_seconds()
        if diff >= OVERLAY_DECAY_SEC:
            entry["consumed"] = float(entry.get("consumed", 0.0)) * OVERLAY_DECAY_FACTOR
            entry["ts"] = _now_iso()
    except Exception:
        return entry
    return entry


def _simulate_buy_with_cost(book: Dict[str, Any], max_cost: float) -> Dict[str, Any]:
    remaining_cost = max_cost
    filled_shares = 0.0
    total_cost = 0.0
    levels_filled: List[Dict[str, float]] = []

    asks = _sorted_levels(book.get("asks"), reverse=False)
    for level in asks:
        if remaining_cost <= 0:
            break
        try:
            price = float(level["price"])
            available = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 0 or available <= 0:
            continue

        max_shares_at_level = remaining_cost / price
        take = min(available, max_shares_at_level)
        if take <= 0:
            continue

        cost_here = take * price
        filled_shares += take
        total_cost += cost_here
        remaining_cost -= cost_here
        levels_filled.append({"price": price, "shares": take, "cost": cost_here})

    avg_price = total_cost / filled_shares if filled_shares > 0 else None
    return {
        "filled": filled_shares,
        "cost": total_cost,
        "avg_price": avg_price,
        "unfilled_cost": max(0.0, remaining_cost),
        "levels": levels_filled,
    }


def _simulate_sell_shares(book: Dict[str, Any], desired_shares: float) -> Dict[str, Any]:
    remaining_shares = desired_shares
    filled_shares = 0.0
    proceeds = 0.0
    levels_filled: List[Dict[str, float]] = []

    bids = _sorted_levels(book.get("bids"), reverse=True)
    for level in bids:
        if remaining_shares <= 0:
            break
        try:
            price = float(level["price"])
            available = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 0 or available <= 0:
            continue

        take = min(available, remaining_shares)
        if take <= 0:
            continue

        proceeds_here = take * price
        filled_shares += take
        proceeds += proceeds_here
        remaining_shares -= take
        levels_filled.append({"price": price, "shares": take, "proceeds": proceeds_here})

    avg_price = proceeds / filled_shares if filled_shares > 0 else None
    return {
        "filled": filled_shares,
        "proceeds": proceeds,
        "avg_price": avg_price,
        "unfilled_shares": max(0.0, remaining_shares),
        "levels": levels_filled,
    }


def _front_depth_usd(book: Dict[str, Any], side: str, top_n: int = OVERLAY_TOP_N) -> float:
    levels = book.get("asks" if side == "BUY" else "bids", [])
    total = 0.0
    for lvl in levels[:top_n]:
        try:
            price = float(lvl["price"])
            size = float(lvl["size"])
        except Exception:
            continue
        total += price * size
    return total


def _apply_overlay_to_book(real_book: Dict[str, Any], overlay: Dict[str, Any], token_id: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    token_overlay = overlay.get(token_id, {})
    ask_eaten = token_overlay.get("asks", {}) or {}
    bid_eaten = token_overlay.get("bids", {}) or {}

    new_overlay_asks: Dict[str, Dict[str, Any]] = {}
    new_overlay_bids: Dict[str, Dict[str, Any]] = {}

    new_asks = []
    for lvl in real_book.get("asks", []):
        price_str = str(lvl.get("price"))
        try:
            real_size = float(lvl["size"])
        except Exception:
            continue

        entry = ask_eaten.get(price_str)
        if isinstance(entry, dict):
            entry = _maybe_decay(entry)
            consumed = float(entry.get("consumed", 0.0))
        else:
            consumed = float(entry or 0.0)
            entry = {"consumed": consumed, "ts": _now_iso()}

        if consumed < 0:
            consumed = 0.0
        if consumed > real_size:
            consumed = real_size
            entry["consumed"] = consumed

        remain = real_size - consumed
        if remain > 1e-9:
            new_asks.append({"price": lvl["price"], "size": remain})

        if consumed > 1e-9:
            entry["consumed"] = consumed
            entry.setdefault("ts", _now_iso())
            new_overlay_asks[price_str] = entry

    new_bids = []
    for lvl in real_book.get("bids", []):
        price_str = str(lvl.get("price"))
        try:
            real_size = float(lvl["size"])
        except Exception:
            continue

        entry = bid_eaten.get(price_str)
        if isinstance(entry, dict):
            entry = _maybe_decay(entry)
            consumed = float(entry.get("consumed", 0.0))
        else:
            consumed = float(entry or 0.0)
            entry = {"consumed": consumed, "ts": _now_iso()}

        if consumed < 0:
            consumed = 0.0
        if consumed > real_size:
            consumed = real_size
            entry["consumed"] = consumed

        remain = real_size - consumed
        if remain > 1e-9:
            new_bids.append({"price": lvl["price"], "size": remain})

        if consumed > 1e-9:
            entry["consumed"] = consumed
            entry.setdefault("ts", _now_iso())
            new_overlay_bids[price_str] = entry

    if new_overlay_asks or new_overlay_bids:
        token_overlay["asks"] = new_overlay_asks
        token_overlay["bids"] = new_overlay_bids
        overlay[token_id] = token_overlay
    elif token_id in overlay:
        overlay.pop(token_id, None)

    adjusted_book = {"asks": new_asks, "bids": new_bids}
    return adjusted_book, overlay


def _record_consumed(
    signature: str,
    token_id: str,
    side: str,
    levels: List[Dict[str, float]],
    traded_usd: float,
    front_depth_usd: float,
) -> None:
    if front_depth_usd <= 0:
        return
    impact_ratio = traded_usd / front_depth_usd if front_depth_usd > 0 else 0.0
    if impact_ratio < OVERLAY_MIN_IMPACT:
        return

    with _liquidity_lock:
        data = _load_liquidity(signature)
        tok = data.setdefault(token_id, {})
        side_key = "asks" if side == "BUY" else "bids"
        side_map = tok.setdefault(side_key, {})

        for lvl in levels or []:
            price = lvl.get("price")
            consumed_shares = float(lvl.get("shares") or lvl.get("size") or 0.0)
            if consumed_shares <= 0:
                continue
            price_str = str(price)
            entry = side_map.get(price_str)
            if isinstance(entry, dict):
                prev = float(entry.get("consumed", 0.0))
            else:
                prev = float(entry or 0.0)
            side_map[price_str] = {
                "consumed": prev + consumed_shares,
                "ts": _now_iso(),
            }

        data[token_id] = tok
        _save_liquidity(signature, data)


def _position_file(signature: str) -> Path:
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "data" / "agent_data" / signature / "position" / "position.jsonl"


def _ensure_position(signature: str, init_cash: float = 10000.0, init_datetime: Optional[str] = None) -> None:
    pos_file = _position_file(signature)
    if pos_file.exists():
        return
    pos_file.parent.mkdir(parents=True, exist_ok=True)
    init_positions = {"CASH": float(init_cash)}
    if not init_datetime:
        init_datetime = str(get_config_value("INIT_DATETIME")) or "1970-01-01T00:00:00Z"
    pos_file.write_text(json.dumps({"timestamp": init_datetime, "id": 0, "positions": init_positions}) + "\n", encoding="utf-8")


def _get_latest(signature: str) -> tuple[Dict[str, Any], int]:
    pos_file = _position_file(signature)
    if not pos_file.exists():
        return {}, -1
    last = {}
    last_id = -1
    with pos_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            last = doc
            last_id = int(doc.get("id", last_id))
    return last.get("positions", {}), last_id

@function_tool
def buy(market_slug: str, outcome: str, cost_usd: float) -> Dict[str, Any]:
    """
    Buy shares for a given market outcome using notional USD (ignores slippage/fees).

    Args:
        market_slug: Polymarket market slug
        outcome: outcome name (e.g., 'Yes'/'No' or option label)
        cost_usd: how much USD to spend
        cost_usd: how much USD to spend (float)
    """
    signature = get_config_value("SIGNATURE")
    if not signature:
        return {"error": "SIGNATURE not set in runtime env"}

    if cost_usd <= 0:
        return {"error": "cost_usd must be positive"}

    _ensure_position(signature)
    positions, last_id = _get_latest(signature)

    cash = float(positions.get("CASH", 0.0))
    if cash < cost_usd:
        return {"error": "Insufficient cash", "cash": cash, "required": cost_usd}

    try:
        _, token_id = _resolve_market_and_token(market_slug, outcome)
    except Exception as exc:
        return {"error": str(exc)}

    try:
        book = _get_order_book(token_id, signature)
    except Exception as exc:
        return {"error": f"failed to fetch order book: {exc}"}

    front_depth_usd = _front_depth_usd(book, side="BUY")
    simulation = _simulate_buy_with_cost(book, float(cost_usd))
    filled_shares = simulation.get("filled", 0.0) or 0.0
    total_cost = simulation.get("cost", 0.0) or 0.0

    if filled_shares <= 0 or total_cost <= 0:
        return {"error": "No asks available or insufficient liquidity", "requested_cost": cost_usd}

    if total_cost > cash:
        return {"error": "Insufficient cash for execution", "cash": cash, "required": total_cost}

    _record_consumed(
        signature=signature,
        token_id=token_id,
        side="BUY",
        levels=simulation.get("levels") or [],
        traded_usd=total_cost,
        front_depth_usd=front_depth_usd,
    )

    new_positions = dict(positions)
    new_positions["CASH"] = cash - total_cost
    key = f"{market_slug}:{outcome}"
    new_positions[key] = float(new_positions.get(key, 0.0)) + filled_shares

    pos_file = _position_file(signature)
    current_datetime = str(get_config_value("CURRENT_DATETIME")) or (datetime.utcnow().replace(tzinfo=timezone.utc).isoformat())
    with pos_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": current_datetime,
            "id": last_id + 1,
            "this_action": {
                "action": "buy",
                "market": market_slug,
                "outcome": outcome,
                "requested_cost": float(cost_usd),
                "spent": total_cost,
                "shares": filled_shares,
                "avg_price": simulation.get("avg_price"),
                "partial_fill": total_cost + 1e-9 < cost_usd,
                "levels": simulation.get("levels"),
            },
            "positions": new_positions,
        }) + "\n")

    write_config_value("IF_TRADE", True)
    return {
        "ok": True,
        "positions": new_positions,
        "spent": total_cost,
        "filled_shares": filled_shares,
        "avg_price": simulation.get("avg_price"),
        "requested_cost": float(cost_usd),
        "unspent_budget": simulation.get("unfilled_cost"),
        "partial_fill": total_cost + 1e-9 < cost_usd,
        "fill_levels": simulation.get("levels"),
    }


@function_tool
def sell(market_slug: str, outcome: str, shares: float) -> Dict[str, Any]:
    """
    Sell shares for a given market outcome (mark-to-market using current token price).
    """
    signature = get_config_value("SIGNATURE")
    if not signature:
        return {"error": "SIGNATURE not set in runtime env"}

    if shares <= 0:
        return {"error": "shares must be positive"}

    _ensure_position(signature)
    positions, last_id = _get_latest(signature)

    key = f"{market_slug}:{outcome}"
    have = float(positions.get(key, 0.0))
    if have < shares:
        return {"error": "Insufficient shares", "have": have, "want": shares}

    try:
        _, token_id = _resolve_market_and_token(market_slug, outcome)
    except Exception as exc:
        return {"error": str(exc)}

    try:
        book = _get_order_book(token_id, signature)
    except Exception as exc:
        return {"error": f"failed to fetch order book: {exc}"}

    front_depth_usd = _front_depth_usd(book, side="SELL")
    simulation = _simulate_sell_shares(book, float(shares))
    filled_shares = simulation.get("filled", 0.0) or 0.0
    proceeds = simulation.get("proceeds", 0.0) or 0.0

    if filled_shares <= 0 or proceeds <= 0:
        return {"error": "No bids available or insufficient liquidity", "requested_shares": shares}

    _record_consumed(
        signature=signature,
        token_id=token_id,
        side="SELL",
        levels=simulation.get("levels") or [],
        traded_usd=proceeds,
        front_depth_usd=front_depth_usd,
    )

    new_positions = dict(positions)
    new_positions[key] = have - filled_shares
    new_positions["CASH"] = float(new_positions.get("CASH", 0.0)) + proceeds

    pos_file = _position_file(signature)
    current_datetime = str(get_config_value("CURRENT_DATETIME")) or (datetime.utcnow().replace(tzinfo=timezone.utc).isoformat())
    with pos_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": current_datetime,
            "id": last_id + 1,
            "this_action": {
                "action": "sell",
                "market": market_slug,
                "outcome": outcome,
                "requested_shares": float(shares),
                "shares": filled_shares,
                "avg_price": simulation.get("avg_price"),
                "proceeds": proceeds,
                "partial_fill": filled_shares + 1e-9 < shares,
                "levels": simulation.get("levels"),
            },
            "positions": new_positions,
        }) + "\n")

    write_config_value("IF_TRADE", True)
    return {
        "ok": True,
        "positions": new_positions,
        "proceeds": proceeds,
        "filled_shares": filled_shares,
        "avg_price": simulation.get("avg_price"),
        "requested_shares": float(shares),
        "unfilled_shares": simulation.get("unfilled_shares"),
        "partial_fill": filled_shares + 1e-9 < shares,
        "fill_levels": simulation.get("levels"),
    }


def settle(market_slug: str, signature: str) -> Dict[str, Any]:
    """
    Settle a closed market using Gamma market info (no CLOB /markets):
    winner outcomes pay out 1.0 per share, losers 0.
    """

    _ensure_position(signature)
    positions, last_id = _get_latest(signature)

    m = get_market_info_fn(market_slug)
    if not isinstance(m, dict) or ("error" in m):
        return {"error": "market not found"}

    status = str(m.get("status", "")).strip().lower()
    is_closed = bool(m.get("closed")) or status in ("closed", "resolved", "settled")
    if not is_closed:
        return {"error": "market not closed yet"}

    winners = set()
    tokens = m.get("tokens") or []
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict):
        winners = {
            str(t["outcome"]).strip()
            for t in tokens
            if t.get("winner")
            and t.get("outcome") is not None
            and str(t.get("outcome")).strip()
        }
    if not winners:
        for key in ("winningOutcome", "winning_outcome", "resolvedOutcome", "resolved_outcome", "result"):
            val = m.get(key)
            if val is None:
                continue
            values = val if isinstance(val, list) else [val]
            winners = {str(x).strip() for x in values if x is not None and str(x).strip()}
            if winners:
                break

    if not winners:
        return {"error": "market resolution unavailable"}

    new_positions = dict(positions)
    realized = 0.0
    keys = [k for k in list(new_positions.keys()) if k.startswith(f"{market_slug}:")]
    for k in keys:
        outcome = k.split(":", 1)[1]
        qty = float(new_positions.get(k, 0.0))
        if qty <= 0:
            continue
        payout = qty * (1.0 if (winners and outcome in winners) else 0.0)
        realized += payout
        new_positions[k] = 0.0

    new_positions["CASH"] = float(new_positions.get("CASH", 0.0)) + realized

    pos_file = _position_file(signature)
    current_datetime = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    with pos_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": current_datetime,
            "id": last_id + 1,
            "this_action": {"action": "settle", "market": market_slug, "realized": realized},
            "positions": new_positions,
        }) + "\n")
    return {"ok": True, "realized": realized, "positions": new_positions}

if __name__ == "__main__":
    from agents.tool_context import ToolContext
    import asyncio
    args = dict(market_slug = "will-the-supreme-court-overturn-gay-marriage-in2025", outcome = "No", cost_usd = 5000)
    ctx = ToolContext(
        context=None,
        tool_name="buy",
        tool_call_id="buy-test",
        tool_arguments=json.dumps(args)
    )
    result = asyncio.run(
        buy.on_invoke_tool(
            ctx,
            json.dumps(args)
        )
    )
    print(result)