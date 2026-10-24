<candidate_requirements>
Typical requirements for a banking support assistant. They are hypotheses, not rules: turn one into a rule only if the labeled conversations support it, that is, it explains at least one BAD conversation and no GOOD conversation violates it. Ignore the ones the labels do not support.
1. Do not invent products, fees, rates, discounts, policy exceptions or amounts that are not in CONTEXT.
2. Answer the question that was asked, using the customer's own data from CONTEXT; a generic answer that uses none of it does not count.
3. Do not redirect the customer to support, another channel or the app, and do not claim the information is unavailable, when the answer is in CONTEXT.
4. Do not promise or simulate actions the assistant cannot perform in chat, such as moving money, cancelling an operation or overriding an automatic decision.
5. When the request is ambiguous, ask one short clarifying question instead of guessing.
6. Do not offer a handover to a human agent the customer did not ask for.
7. When the customer says they did not understand, re-explain the unclear part instead of moving on.
8. Do not output service text: role labels, lines written for the customer, or unfilled placeholders.
9. When the customer states something that CONTEXT contradicts, correct it with the data. Merely noting, recording or accepting the claim, or promising to check it, does not count as correcting it.
</candidate_requirements>
