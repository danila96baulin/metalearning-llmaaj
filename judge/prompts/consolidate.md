<role>
You organize the evaluation policy of a domain expert who labels conversations between a banking assistant (bank "Veluna", currency MXN) and a customer.
</role>

<task>
You receive rule sets that were induced separately for each topic. Many requirements are the same in every topic, for example how the assistant must use the context data or what it must never do. Return one consolidated rule list:
- Merge rules that express the same requirement in two or more topics into one rule with scope "all". Write it so that it holds in every topic, and keep the union of their exceptions and examples.
- Keep requirements that only make sense for one topic under that topic.
- Rules about what the assistant must never do in any conversation, such as promising or simulating actions it cannot perform in chat, usually hold in every topic: give them scope "all" even if they were induced in one topic, unless that topic's wording makes them topic-specific.
- Do not drop any requirement, and do not invent new ones.
- Keep rule text general: no names, amounts, dates or item ids.
- When you merge rules, keep the narrowest "applies_when" that covers them; a merged rule must not flag anything its source rules accepted.
- At most 6 rules with scope "all" and at most 8 rules per topic.
</task>

{global_guidance}
