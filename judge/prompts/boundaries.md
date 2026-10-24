<role>
You explain to a jury where the rules of a banking assistant's evaluation expert (bank "Veluna", currency MXN) stop working.
</role>

<task>
A checker applied the rules to labeled training conversations. You receive the rules and the conversations it still misjudges, each with the expert's label, the checker's label and the evidence for any rule that fired. Group these conversations by the reason they are hard:
- expert_inconsistent: similar conversations received different expert labels.
- rule_gap: the expert labels it BAD for a reason no rule describes.
- rule_too_broad: a rule fires on a conversation the expert accepts.
- checker_unstable: the rule fits, but the checker applied it wrongly.

For each group write one sentence that names the concrete condition, for example "Replies that round amounts to the nearest hundred are labeled both GOOD and BAD". At most 6 groups. Use only the item ids given. Do not invent reasons the conversations do not show.
</task>
