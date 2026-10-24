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
HORIZONS = (1, 3, 6, 12, 24, 36, 48)
DATE = re.compile(r"(0?[1-9]|[12]\d|3[01])[./](0?[1-9]|1[0-2])")  # 14.09, 05/10: a day and month, not an amount
REFUND = ("refund", "reembolso", "devoluci", "возврат")
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


def _spend(row):
    """Money spent by one transaction: a purchase counts, a refund takes it back; None for other types."""
    kind = str(row.get("type", "")).lower()
    amount = abs(num(row.get("amount")))
    if any(x in kind for x in REFUND):
        return -amount
    if "purchase" in kind or "compra" in kind:
        return amount
    return None


def _rounding(token, value):
    """Half the rounding unit of a round figure such as 12,000 or 150, capped at 2% of the value."""
    if re.search(r"[.,]\d{1,2}$", token):
        return 0.0  # a figure with decimals is exact, not rounded
    digits = re.sub(r"[^\d]", "", token)
    zeros = len(digits) - len(digits.rstrip("0"))
    return min(0.02 * value, 10 ** zeros / 2) if zeros >= 1 and value >= 100 else 0.0


def _applies(key, row, plus):
    """Whether a rate column applies to this customer: with a Veluna+ subscription only the *_with_plus rate,
    without one only the plain rate; both when the subscription is unknown or the row has no plus variant."""
    k = key.lower()
    has_plus_variant = any("plus" in other.lower() and "rate" in other.lower() for other in row)
    if plus is None or not has_plus_variant:
        return True
    return ("plus" in k and "without" not in k) == plus


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

    def add(value, description, zero=False):
        if isinstance(value, (int, float)) and (abs(value) >= 0.005 or zero):
            refs.append((round(abs(value), 2), description))

    def walk(value, path="context"):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            add(value, f"{path} (as given)", zero=True)
    walk(ctx)

    balances, rates, account_debts = [], [], []
    user = ctx.get("user") if isinstance(ctx.get("user"), dict) else {}
    plus = user.get("plus_subscription") if isinstance(user.get("plus_subscription"), bool) else None
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
                    if _spend(r) is not None:  # net of refunds: a refund cancels the purchase it returns
                        for field in ("category", "merchant"):
                            if r.get(field):
                                by_month[m][f"net {field}={r[field]}"] += _spend(r)
            metrics = {}
            for m, values in by_month.items():
                for k, v in values.items():
                    metrics[(m, k)] = v
                    if k.startswith("net ") and abs(v) < 0.005:
                        add(0.0, f"{name} {m}: {k}", zero=True)
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
                    totals, nets = defaultdict(float), defaultdict(float)
                    for r in rows:
                        if r.get(field):
                            totals[r[field]] += abs(num(r.get(amount)))
                            if _spend(r) is not None:
                                nets[r[field]] += _spend(r)
                    for value, total in totals.items():
                        add(total, f"{name}: total {field}={value}")
                        if len(months) > 1:
                            add(total / len(months), f"{name}: average per month {field}={value} over {len(months)} months")
                    for value, net in nets.items():
                        if abs(net - totals[value]) >= 0.005:
                            add(net, f"{name}: total {field}={value} net of refunds", zero=True)
        # Accounts: totals, available after holds
        balance = _key(keys, "balance")
        hold = _key(keys, "hold") or _key(keys, "held")
        debt_key = _key(keys, "debt")
        if debt_key and path.endswith("accounts"):
            account_debts.extend(num(r.get(debt_key)) for r in rows)
        if balance:
            for r in rows:
                b = num(r.get(balance))
                label = str(r.get("product") or r.get("name") or "account")
                balances.append((label, b, b - num(r.get(hold)) if hold else b))
        for r in rows:
            for k in keys:
                rate = num(r.get(k))
                if "rate" in k.lower() and rate > 0 and _applies(k, r, plus):
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
    account_debt = sum(account_debts)
    if balances and account_debt:
        total = sum(b for _, b, _ in balances)
        available = sum(a for _, _, a in balances)
        add(total - account_debt, "sum of balances minus account current_debt")
        add(available - account_debt, "available balances minus account current_debt")
        for label, b, a in balances:
            if b:
                add(total - b - account_debt, f"all balances except {label} minus account current_debt")
                add(available - a - account_debt, f"available balances except {label} minus account current_debt")
    credit = ctx.get("credit_statement") if isinstance(ctx.get("credit_statement"), dict) else {}
    debt = num(credit.get("total_debt")) or None
    if isinstance(debt, (int, float)):
        if balances:
            add(sum(b for _, b, _ in balances) - debt, "sum of balances minus credit total_debt")
            add(sum(a for _, _, a in balances) - debt, "available balances minus credit total_debt")
        for k in ("minimum_payment", "no_interest_payment"):
            if num(credit.get(k)):
                add(debt - num(credit.get(k)), f"credit total_debt − {k}")
        paid = num(credit.get("paid_this_period"))
        if paid:  # payment status: 0.00 left to pay means that amount is paid in full
            for k in ("total_debt", "no_interest_payment", "minimum_payment"):
                if num(credit.get(k)):
                    add(max(num(credit.get(k)) - paid, 0.0), f"left to pay: credit {k} − paid_this_period", zero=True)
        rate = num(credit.get("annual_rate"))
        if isinstance(rate, (int, float)) and rate:
            r = rate / 100 if rate > 1 else rate
            add(debt * r / 12, "credit total_debt × annual_rate / 12 (one month of interest)")
            for k in ("paid_this_period", "minimum_payment", "no_interest_payment"):
                if num(credit.get(k)) and debt > num(credit.get(k)):
                    add((debt - num(credit.get(k))) * r / 12, f"(credit total_debt − {k}) × annual_rate / 12")
        msi = credit.get("msi_plans") or []
        if msi:
            add(sum(num(p.get("monthly_payment")) * num(p.get("remaining_payments")) for p in msi if isinstance(p, dict)), "remaining MSI installments total")
        if rate:
            rates.append(("credit_statement.annual_rate", rate))
            refs.append((round(rate * 100 if rate <= 1 else rate, 2), "[%] credit_statement.annual_rate"))
            refs.append((round((rate * 100 if rate <= 1 else rate) / 12, 2), "[%] credit_statement.annual_rate / 12 (monthly)"))
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
                    rate = num(credit.get("annual_rate"))
                    if rate and debt > value:
                        r = rate / 100 if rate > 1 else rate
                        add((debt - value) * r / 12, f"(credit total_debt − {fmt(value)} the customer said) × annual_rate / 12")
    principals = [(label, b, True) for label, b, _ in balances] + \
        [(f"{'customer' if role == 'user' else 'assistant'} amount {fmt(v)}", v, role == "user") for role, v in dict.fromkeys(said)]
    for (rate_label, rate), (p_label, p, sourced), months in product(rates, principals, HORIZONS):
        r = rate / 100 if rate > 1 else rate
        if r > 10:
            continue  # not a real rate (over 1,000% a year); compounding it would overflow
        if months > 12 and not sourced:
            continue  # multi-year projections only from balances and customer amounts, to avoid chance matches
        add(p * r * months / 12, f"{p_label} × {rate_label} for {months} months")
        if months == 12 and not sourced:
            add(p * (1 + r), f"{p_label} plus one year at {rate_label}")
        elif months >= 12:
            years = months // 12
            add(p * (1 + r * years), f"{p_label} plus {years} year(s) at {rate_label}, simple interest")
            for label, factor in (("compounded daily", (1 + r / 365) ** (365 * years)),
                                  ("compounded monthly", (1 + r / 12) ** months)):
                add(p * (factor - 1), f"{p_label} × {rate_label} for {years} year(s), {label}: interest")
                add(p * factor, f"{p_label} plus {years} year(s) at {rate_label}, {label}")
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
             "Numbers in the assistant's replies and the closest computed value (tolerance: 0.5% or 1 MXN; a round figure "
             "such as 12,000 may differ by half its rounding unit, at most 2%):"]
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
            if not percent and (value <= 31 and plain_int or 2000 <= value <= 2100 and plain_int
                                or DATE.fullmatch(match.group(1))):
                continue  # small counts, days, dates and years are left to the model
            seen += 1
            pool = [(v, d) for v, d in refs if d.startswith("[%]") == percent] or refs
            best = min(pool, key=lambda x: abs(x[0] - value))
            tolerance = 0.05 if percent else max(1.0, 0.005 * value, _rounding(match.group(1), value))
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
