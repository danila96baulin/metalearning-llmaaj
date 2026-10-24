<other_violation>
Besides the rules, report "other_violation" only for a clear error that no rule above covers:
- a number, date, rate or balance that contradicts CONTEXT, or a product, fee, discount or exception that CONTEXT does not contain;
- the customer's question left unanswered, or answered without their data although CONTEXT has it;
- a redirect to support or another channel, or a claim that the information is unavailable, although CONTEXT has the answer;
- a promise or simulation of an action the assistant cannot perform in chat;
- service text in a reply: role labels, lines written for the customer, unfilled placeholders.
Give a verbatim quote from an assistant message as evidence. If there is no such error, set "violated" to false.
</other_violation>
