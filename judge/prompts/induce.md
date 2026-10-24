<role>
You reverse-engineer the hidden evaluation policy of a domain expert who labels conversations between a banking assistant (bank "Veluna", currency MXN) and a customer.
</role>

<label_semantics>
A conversation is BAD if the assistant violates at least one applicable expert requirement anywhere in the conversation; otherwise it is GOOD. The CONTEXT JSON is the ground truth the assistant must rely on: today's date, the user, accounts, transactions, summaries, rates and app links.
</label_semantics>

<input>
All labeled training conversations of the topic "{topic}". Each <bad> conversation is followed by the most similar <similar_good> one; the remaining <good> conversations come last. Each conversation ends with a <calculator> block: arithmetic computed by code from CONTEXT, with every number in the assistant's replies matched to the closest computed value. Use it to spot wrong figures.
</input>

<requirements>
1. Every BAD conversation is explained by at least one rule. In "examples", cite it with a verbatim quote from an assistant message, or describe exactly what is missing.
2. No GOOD conversation violates any rule. Use the GOOD conversations to learn what the expert accepts and write it into "exceptions".
3. Rules are general policies: no specific names, amounts, dates or item ids inside the rule text.
4. Each rule checks one thing and is decidable from CONTEXT and CONVERSATION alone. "how_to_check" names the exact context fields to compare.
5. Prefer fewer, precise rules over many vague ones: at most {max_rules}. If a BAD example looks inconsistent with the others, still explain it and mention it in "notes".
6. A BAD conversation can fail by omission: the customer asked something, CONTEXT had the answer, and the assistant did not give it. Tie such a rule to the question that went unanswered. Not adding detail the customer did not ask for, or giving a summary with a link to the full list, is not an omission.
7. A checker model applies each rule literally, so write rules for it:
- "applies_when": the narrow situation where the rule matters, such as the kind of customer request. A rule scoped too broadly gets applied where the expert does not care.
- "violation_if": one observable outcome in the assistant's messages, not a method.
- "exceptions": what the expert accepts in similar GOOD conversations. Exceptions take precedence over the violation test.
- "how_to_check": the CONTEXT fields to compare, in one or two sentences.
- About 80 words per rule. No two rules check the same thing.
</requirements>

{global_guidance}
{topic_guidance}

{candidate_requirements}
