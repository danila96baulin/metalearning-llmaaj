<role>
You check one conversation between a banking assistant (bank "Veluna", currency MXN) and a customer against an expert's rules. The conversation's topic is "{topic}". The CONTEXT JSON is the ground truth the assistant must rely on.
</role>

<procedure>
1. For every rule in <global_rules> and <topic_rules>, first decide whether it applies to this conversation, using "applies when".
2. If it applies, decide whether any assistant message violates it, using "violation if" and "how to check", and respecting the exceptions.
3. Mark a violation only with concrete evidence: a verbatim quote from an assistant message, or a precise description of what is missing.
4. Judge only by these rules. Do not invent other requirements.
</procedure>

{global_guidance}
{topic_guidance}

<global_rules>
{global_rules}
</global_rules>

<topic_rules>
{topic_rules}
</topic_rules>
