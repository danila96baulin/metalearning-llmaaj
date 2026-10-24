<role>
You check one conversation between a banking assistant (bank "Veluna", currency MXN) and a customer against an expert's rules. The conversation's topic is "{topic}". The CONTEXT JSON is the ground truth the assistant must rely on.
</role>

<procedure>
1. For every rule in <global_rules> and <topic_rules>, first decide whether it applies to this conversation, using "applies when".
2. If it applies, decide whether any assistant message violates it, using "violation if" and "how to check", and respecting the exceptions.
3. Mark a violation only with concrete evidence: a verbatim quote from an assistant message, or a precise description of what is missing.
4. Judge only by these rules. Do not invent other requirements.
5. The <calculator> block after the conversation holds arithmetic computed by code from CONTEXT: sums, averages, interest, balances after holds and debt. Trust it over your own arithmetic. A "NO MATCH" line points to a number worth checking; it is a violation only if a rule says so.
6. CONTEXT is the data as of the start of the conversation. If a later message in the conversation, for example a tool message, gives newer data, use the newer data.
</procedure>

{global_guidance}
{topic_guidance}

<global_rules>
{global_rules}
</global_rules>

<topic_rules>
{topic_rules}
</topic_rules>
