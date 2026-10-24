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
5. Prefer fewer, precise rules over many vague ones: at most 12. If a BAD example looks inconsistent with the others, still explain it and mention it in "notes".
6. State what the assistant should have done, not only what it did wrong. A BAD conversation can fail by omission: the assistant did not give required information, did not correct the customer, did not use data it had. Write "violation_if" so that it also catches such failures, not just a literally wrong statement.
</requirements>

{global_guidance}
{topic_guidance}

{candidate_requirements}
