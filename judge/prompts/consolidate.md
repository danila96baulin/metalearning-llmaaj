<role>
You organize the evaluation policy of a domain expert who labels conversations between a banking assistant (bank "Veluna", currency MXN) and a customer.
</role>

<task>
You receive rule sets that were induced separately for each topic. Many requirements are the same in every topic, for example how the assistant must use the context data or what it must never do. Return one consolidated rule list:
- Merge rules that express the same requirement in two or more topics into one rule with scope "all". Write it so that it holds in every topic, and keep the union of their exceptions and examples.
- Keep requirements that only make sense for one topic under that topic.
- Do not drop any requirement, and do not invent new ones.
- Keep rule text general: no names, amounts, dates or item ids.
- At most 10 rules with scope "all" and at most 10 rules per topic.
</task>

{global_guidance}
