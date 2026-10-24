<role>
You maintain the expert's rules for the topic "{topic}" of a banking assistant (bank "Veluna", currency MXN).
</role>

<label_semantics>
A conversation is BAD if the assistant violates at least one applicable rule anywhere in it; otherwise it is GOOD. The CONTEXT JSON is the ground truth.
</label_semantics>

<task>
A checker applied the global rules and the topic rules to labeled training conversations and made the errors shown. Revise the topic rules so that the missed BAD conversations are caught and the wrongly flagged GOOD conversations are not.
- The global rules are read-only; change only the topic rules.
- Allowed changes: narrow or clarify a rule, add an exception, add a missing rule, remove a rule that only causes false flags.
- Keep rules general: no names, amounts, dates or item ids. At most 12 topic rules.
- Return the complete revised topic rules. Keep the id of every rule you keep or edit; give new rules an id that is not in use.
</task>

{global_guidance}
{topic_guidance}

{candidate_requirements}
