"""Arithmetic computed in code from CONTEXT, so the models do not have to add up figures themselves.

calculator_block(row) returns a <calculator> block with key figures (monthly net spending, averages, account totals,
debt and interest) and, for every number in the assistant's replies, the closest value computed from CONTEXT or
said by the customer. It works on any numeric tables in CONTEXT; the named formulas cover the task's banking fields.
"""
import re
from collections import defaultdict
from itertools import product

NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:[   ,.]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)(\s?%)?")
THOUSANDS = re.compile(r"\s?(k|к|тыс|thousand|mil)(?![a-zа-я])", re.I)
HORIZONS = (1, 3, 6, 12)
MAX_REPLY_NUMBERS = 30


def parse_number(token):
    """'26,967' / '13 257' / '21 300,80' / '1.750,00' / '245.50' -> float."""
    token = re.sub(r"[   ]", "", token)
    seps = [i for i, ch in enumerate(token) if ch in ",."]
    if not seps:
        return float(token)
    last = seps[-1]
    decimals = len(token) - last - 1
    if decimals == 3 and (len(seps) > 1 or len(token[:last]) <= 3):  # thousands separator(s) only
        return float(re.sub(r"[,.]", "", token))
    return float(re.sub(r"[,.]", "", token[:last]) + "." + token[last + 1:])


def num(value):
    """A context value as a float: numbers and numeric strings such as "1,750.00"; anything else is 0."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and NUMBER.fullmatch(value.strip()):
        try:
            return parse_number(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def fmt(value):
    return f"{value:,.2f}"


def _month(row):
    for key in ("month", "date"):
        if isinstance(row.get(key), str) and re.match(r"\d{4}-\d{2}", row[key]):
            return row[key][:7]
    return None


def _tables(value, path="context"):
    """Every list of dicts in the context, plus dicts keyed by month, as (path, rows)."""
    if isinstance(value, list) and value and all(isinstance(r, dict) for r in value):
        yield path, value
        for i, r in enumerate(value):
            yield from _tables(r, f"{path}[{i}]")
    elif isinstance(value, dict):
        month_keys = [k for k in value if re.match(r"\d{4}-\d{2}", str(k))]
        if month_keys and len(month_keys) == len(value):
            rows = [{"month": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in value.items()]
            yield path, rows
        for k, v in value.items():
            yield from _tables(v, f"{path}.{k}")


def _numeric(rows):
    keys = {k for r in rows for k, v in r.items() if num(v)}
    return sorted(keys)


def _key(keys, *needles, exclude=()):
    return next((k for k in keys if all(n in k.lower() for n in needles) and not any(x in k.lower() for x in exclude)), None)


def references(row):
    """(value, description) pairs computed from CONTEXT and from amounts the customer mentioned."""
    ctx, refs = row["context"], []
    today = str(ctx.get("today", ""))[:7]

    def add(value, description):
        if isinstance(value, (int, float)) and abs(value) >= 0.005:
            refs.append((round(abs(value), 2), description))

    def walk(value, path="context"):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            add(value, f"{path} (as given)")
    walk(ctx)

    balances, rates = [], []
    for path, rows in _tables(ctx):
        keys = _numeric(rows)
        name = path.replace("context.", "")
        # Monthly tables: net spending per month and over windows of months
        if any(_month(r) for r in rows):
            purchases = _key(keys, "purchase", exclude=("count",)) or _key(keys, "spend") or _key(keys, "gross") or _key(keys, "total_spent")
            refunds = _key(keys, "refund")
            to_others = _key(keys, "transfer", "other")
            amount = _key(keys, "amount")
            by_month = defaultdict(lambda: defaultdict(float))
            counts = defaultdict(int)
            for r in rows:
                m = _month(r)
                if not m:
                    continue
                for k in keys:
                    by_month[m][k] += abs(num(r.get(k)))
                counts[m] += 1
                if amount:  # transaction-like rows: sums by month and by category, merchant, type
                    for field in ("category", "merchant", "type", "status"):
                        if r.get(field):
                            by_month[m][f"{field}={r[field]}"] += abs(num(r.get(amount)))
                            by_month[m][f"count {field}={r[field]}"] += 1
            metrics = {}
            for m, values in by_month.items():
                for k, v in values.items():
                    metrics[(m, k)] = v
                if purchases:
                    net = values[purchases] - (values[refunds] if refunds else 0)
                    metrics[(m, f"{purchases} − {refunds}" if refunds else purchases)] = net
                    if to_others:
                        metrics[(m, f"{purchases} − {refunds or 0} + {to_others}")] = net + values[to_others]
                if amount:
                    metrics[(m, f"count of {name} rows")] = counts[m]
            for (m, k), v in metrics.items():
                add(v, f"{name} {m}: {k}")
            months = sorted(by_month)
            complete = [m for m in months if not today or m < today]
            measures = sorted({k for (_, k) in metrics})
            for window_months in [complete[-n:] for n in range(2, len(complete) + 1)] + [months]:
                if len(window_months) < 2:
                    continue
                label = f"{window_months[0]}..{window_months[-1]}"
                for k in measures:
                    vals = [metrics.get((m, k), 0.0) for m in window_months]
                    if any(vals):
                        add(sum(vals), f"{name} {label}: sum of {k}")
                        add(sum(vals) / len(vals), f"{name} {label}: average per month of {k}")
            # Whole-table sums by merchant or category, and their monthly average
            if amount:
                for field in ("merchant", "category"):
                    totals = defaultdict(float)
                    for r in rows:
                        if r.get(field):
                            totals[r[field]] += abs(num(r.get(amount)))
                    for value, total in totals.items():
                        add(total, f"{name}: total {field}={value}")
                        if len(months) > 1:
                            add(total / len(months), f"{name}: average per month {field}={value} over {len(months)} months")
        # Accounts: totals, available after holds
        balance = _key(keys, "balance")
        hold = _key(keys, "hold") or _key(keys, "held")
        if balance:
            for r in rows:
                b = num(r.get(balance))
                label = str(r.get("product") or r.get("name") or "account")
                balances.append((label, b, b - num(r.get(hold)) if hold else b))
        for r in rows:
            for k in keys:
                rate = num(r.get(k))
                if "rate" in k.lower() and rate > 0:
                    rates.append((f"{r.get('product', name)} {k}", rate))
                    refs.append((round(rate * 100 if rate <= 1 else rate, 2), f"[%] {r.get('product', name)} {k}"))
    if balances:
        total = sum(b for _, b, _ in balances)
        available = sum(a for _, _, a in balances)
        add(total, "sum of all account balances")
        add(available, "sum of balances minus on-hold amounts")
        for label, b, a in balances:
            if a != b:
                add(a, f"{label}: balance minus on-hold")
            add(total - b, f"all balances except {label}")
            add(available - a, f"available balances except {label}")
    credit = ctx.get("credit_statement") if isinstance(ctx.get("credit_statement"), dict) else {}
    debt = num(credit.get("total_debt")) or None
    if isinstance(debt, (int, float)):
        if balances:
            add(sum(b for _, b, _ in balances) - debt, "sum of balances minus credit total_debt")
            add(sum(a for _, _, a in balances) - debt, "available balances minus credit total_debt")
        for k in ("paid_this_period", "minimum_payment", "no_interest_payment"):
            if num(credit.get(k)):
                add(debt - num(credit.get(k)), f"credit total_debt − {k}")
        rate = num(credit.get("annual_rate"))
        if isinstance(rate, (int, float)) and rate:
            r = rate / 100 if rate > 1 else rate
            add(debt * r / 12, "credit total_debt × annual_rate / 12 (one month of interest)")
        msi = credit.get("msi_plans") or []
        if msi:
            add(sum(num(p.get("monthly_payment")) * num(p.get("remaining_payments")) for p in msi if isinstance(p, dict)), "remaining MSI installments total")
        if rate:
            rates.append(("credit_statement.annual_rate", rate))
            refs.append((round(rate * 100 if rate <= 1 else rate, 2), "[%] credit_statement.annual_rate"))
    # Amounts said in the conversation: the customer's are references; both serve as principals for interest
    said = []
    for m in row["conversation"]:
        for match in NUMBER.finditer(str(m.get("content", ""))):
            value = parse_number(match.group(1))
            if THOUSANDS.match(m["content"][match.end():]):
                value *= 1000
            if value < 100 or match.group(2):
                continue
            said.append((m["role"], value))
            if m["role"] == "user":
                add(value, "amount the customer said")
                if isinstance(debt, (int, float)):
                    add(debt - value, f"credit total_debt − amount the customer said ({fmt(value)})")
    principals = [(label, b) for label, b, _ in balances] + \
        [(f"{'customer' if role == 'user' else 'assistant'} amount {fmt(v)}", v) for role, v in dict.fromkeys(said)]
    for (rate_label, rate), (p_label, p), months in product(rates, principals, HORIZONS):
        r = rate / 100 if rate > 1 else rate
        add(p * r * months / 12, f"{p_label} × {rate_label} for {months} months")
        if months == 12:
            add(p * (1 + r), f"{p_label} plus one year at {rate_label}")
    return refs


def calculator_block(row):
    refs = references(row)
    if not refs:
        return ""
    key = [f"- {fmt(v)} = {d}" for v, d in refs
           if not d.startswith("[%]") and ("−" in d and ".." not in d or "average per month of" in d and "−" in d
                                           or d.startswith(("sum of", "available balances", "credit total_debt", "remaining MSI")))]
    lines = ["<calculator>",
             "Arithmetic computed in code from CONTEXT and the customer's messages. It is exact; use it to check the assistant's numbers.",
             "Key figures:", *key[:24],
             "Numbers in the assistant's replies and the closest computed value (tolerance: rounding to 0.5% or 1 MXN):"]
    seen = 0
    for i, m in enumerate(row["conversation"], 1):
        if m["role"] != "assistant":
            continue
        for match in NUMBER.finditer(m["content"]):
            if seen >= MAX_REPLY_NUMBERS:
                break
            value, percent = parse_number(match.group(1)), bool(match.group(2))
            if THOUSANDS.match(m["content"][match.end():]):
                value *= 1000
            plain_int = "," not in match.group(1) and "." not in match.group(1)
            if not percent and (value <= 31 and plain_int or 2000 <= value <= 2100 and plain_int):
                continue  # small counts, days and years are left to the model
            seen += 1
            pool = [(v, d) for v, d in refs if d.startswith("[%]") == percent] or refs
            best = min(pool, key=lambda x: abs(x[0] - value))
            tolerance = 0.05 if percent else max(1.0, 0.005 * value)
            verdict = f"matches {fmt(best[0])} = {best[1]}" if abs(best[0] - value) <= tolerance else \
                f"NO MATCH; closest computed value {fmt(best[0])} = {best[1]}"
            if abs(best[0] - value) <= tolerance and best[1] == "amount the customer said":
                data = [(v, d) for v, d in pool if d != "amount the customer said" and not d.startswith("customer amount")]
                near = min(data, key=lambda x: abs(x[0] - value)) if data else None
                if near and abs(near[0] - value) > tolerance:
                    verdict = (f"equals an amount the customer said; the closest value computed from CONTEXT is "
                               f"{fmt(near[0])} = {near[1]}")
            lines.append(f"- [{i}] {match.group(0).strip()}: {verdict}")
    # Values computed for merchants, categories and products the assistant names in its replies
    replies = " ".join(m["content"].lower() for m in row["conversation"] if m["role"] == "assistant")
    entities = {e for _, d in refs for e in re.findall(r"(?:merchant|category|product)=([^,;:)]+)", d)}
    entities |= {d.split(":")[0].split(" × ")[0].split(" balance")[0].strip() for _, d in refs if " balance" in d}
    related = [f"- {fmt(v)} = {d}" for v, d in refs
               if any(e and e.lower() in replies and e.lower() in d.lower() for e in entities) and "(as given)" not in d]
    if related:
        lines += ["Computed values for merchants, categories and products named in the replies:", *related[:16]]
    lines.append("</calculator>")
    return "\n".join(lines)
