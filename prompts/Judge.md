You are grading an answer from a banking policy assistant against a reference answer.

Question: {question}

Reference answer (ground truth): {gold}

Candidate answer: {answer}

Decide:
- "correct": true only if the candidate states the key facts of the reference (all key numbers, time limits, amounts, names, conditions) without contradicting it. Extra correct detail is fine. Missing a key fact, or a wrong number, means false.
- "refused": true if the candidate declines to answer or says it lacks the information.

Reply with JSON only: {"correct": true|false, "refused": true|false, "reason": "<one short sentence>"}