# Expert judge: stop rules learned from an expert

Our solution to the Plata challenge below. It is for teams who need an LLM judge that agrees with their domain expert and can show why.

**What it does.** The expert labels conversations like a credit policy with stop factors: one violated rule makes a conversation `bad`. `adapt` recovers that policy from the labeled training conversations as an explicit rulebook. A strong model induces rules per topic from each `bad` conversation next to its most similar `good` one and merges the requirements shared by all topics into global rules. A checker model audits the rules blind on the training set. Refinement then revises a topic's rules, or adds an exception to a global rule that holds in that topic only, and is kept only if training accuracy rises and does not fall on a held-out half. Rules that flag as many `good` as `bad` conversations, and topic rules backed by fewer than two `bad` ones, are disabled. Code computes the arithmetic the checker needs from the context, such as sums, averages, interest and what is left to pay. `eval` checks every rule on every test conversation three times; code labels it `bad` if an active rule is violated in a majority of runs. `EXPLANATION.md` is rendered from the same rulebook and prompt files that `eval` uses.

**Expertise inside.** All rules come from the expert's labels in the training file; none are written by hand. The decision logic follows the task's own semantics ("any violation of an applicable rule makes the conversation bad").

**Models.** Inducer `gemini-4-argon` through the [Gemini API](https://ai.google.dev/gemini-api/terms) and checker `gpt-6.1-sol` through the [OpenAI API](https://openai.com/policies/services-agreement); both keys are needed (`GEMINI_API_KEY`, `OPENAI_API_KEY`).

**Results on the public data** (`macro_topic_accuracy` on `public_test`, adapt on `public_train`):

| Judge | Accuracy |
|---|---|
| Majority label per topic | 0.626 |
| Same checker model (`gpt-6.1-sol`), one plain request, no learned rules | 0.625 |
| This judge, two fresh runs in Docker (`adapt` from scratch each time) | 0.972 and 0.989 |

### Run with Docker

```sh
cp .env.example .env            # set GEMINI_API_KEY and OPENAI_API_KEY
docker build --platform linux/amd64 -t expert-judge .
docker run --rm --env-file .env -v "$PWD/in:/input" -v "$PWD/art:/artifacts" expert-judge adapt
docker run --rm --env-file .env -v "$PWD/in:/input" -v "$PWD/art:/artifacts" -v "$PWD/out:/output" expert-judge eval
```

`adapt` reads `/input/train.jsonl` and writes `/artifacts` (rulebook, prompts, training audit, `EXPLANATION.md`). `eval` reads `/input/test.jsonl` and `/artifacts` and writes `/output/predictions.csv`. Configuration is in [.env.example](.env.example). If the model API fails, both stages log a clear error, finish with exit code 0, and keep each topic's majority training label for unjudged conversations. Malformed input rows never stop a run: every `item_id` gets a label.

### Run locally

```sh
pip install -r requirements.txt
scripts/run_public.sh           # prepare inputs, adapt, eval, score into out/public
python -m judge baseline --test out/public/input/test.jsonl --output out/baseline
```

Experiments are archived to `../plata-expert-judge/runs` (override with `RUNS_ARCHIVE`) with per-item status, vote split and fired rules:

```sh
python scripts/run_cv.py my_try       # 5-fold CV of adapt -> eval on train + test, fixed folds in data/public_folds.csv
python scripts/stats.py my_try        # per-topic Wilson CI, bootstrap CI for macro (add --split cv for a CV run)
python scripts/stats.py run_a run_b   # exact McNemar and paired bootstrap of the macro difference
```

---

# A Judge That Learns from an Expert (task)

Infer an expert's evaluation rules from labeled conversations, then assign one of two labels to new conversations: `good` or `bad`. Evaluate the entire conversation. Any violation of an applicable rule makes the conversation `bad`.

**Dataset sizes.** Public: 150 training examples and 80 test examples. Final evaluation: 190 private training examples and 296 private test examples.

The public and private datasets cover the same four topics. The private dataset introduces additional expert requirements whose violations are not represented in the public data, so your solution must adapt again using the training set provided for the final evaluation. The hidden rules are not shared with participants.

## Files

```text
data/
  public_train.csv           # 150 labeled examples for adaptation
  public_test.csv            # 80 labeled examples for local evaluation
src/
  common.py                  # data loading and format validation
  prepare_inputs.py          # CSV -> JSONL; removes labels from prediction inputs
  evaluate.py                # prediction validation and metric calculation
  test_evaluation.py         # evaluation contract tests
README.md
```

The public training set contains 90 `good` and 60 `bad` examples; the public test set contains 50 `good` and 30 `bad` examples. Both sets cover all four topics.

Public test labels are provided for local evaluation. During the final evaluation, your solution receives a labeled private training set and an unlabeled private test set. Adaptation starts with a clean artifacts directory for each public or private run.

## Data format

CSV files use UTF-8 encoding and include a header row. Use a CSV parser: cells may contain quotes, commas, and JSON.

| Column | Contents |
|---|---|
| `item_id` | An opaque, unique example identifier. |
| `topic` | `financial_position`, `savings`, `spending`, or `credit`. |
| `context` | A JSON object containing the source data needed to evaluate the assistant's response. |
| `conversation` | A JSON array of messages with `role` and `content` fields. |
| `label` | The ground-truth label: `good` or `bad`. Omitted from the final prediction input. |
| `input_es` | A Spanish version of the complete input: JSON containing `context` and `conversation`. |
| `input_en` | An English version of the complete input in the same format. |

Translations are alternative representations of the same example, not additional rows or separate tasks. They share the same `item_id`, `topic`, and label. By default, inputs use the original `context` and `conversation`; you can select a translation when preparing inputs. Record the selected language with your results to make experiments comparable.

## Quick start

The scripts in `src/` require Python 3.10 or later. No external libraries or API keys are needed to run these scripts. Run the commands below from the repository root.

Prepare inputs for the two stages of your solution:

```sh
python3 src/prepare_inputs.py --train data/public_train.csv --test data/public_test.csv --out-dir out/public_input
```

The new directory will contain labeled `train.jsonl` and unlabeled `test.jsonl`. Both files contain only the required fields; alternative translations are not duplicated in the inputs. Add `--language es` or `--language en` to select a translation.

During adaptation, use only the training set. During prediction, use the unlabeled test set and the artifacts produced during adaptation. The labeled public test set is intended for scoring and development; the final private test set will be hidden.

Your solution must output **exactly one label for every test ID**. Probabilities and confidence scores are not required. For local evaluation, write predictions as CSV:

```csv
item_id,label
FIRST_EXAMPLE_ID,good
SECOND_EXAMPLE_ID,bad
```

The local scorer also accepts JSONL with exactly the same two fields:

```json
{"item_id":"EXAMPLE_ID","label":"good"}
```

Column names and class labels are case-sensitive. Row order does not matter. Missing, duplicate, or unknown IDs, extra fields, and labels other than `good` or `bad` are rejected with a nonzero exit code.

Measure prediction quality for a complete prediction file:

```sh
python3 src/evaluate.py --test data/public_test.csv --predictions predictions.csv --out out/score.json
```

## Submitting your solution

**Fork this repository and push your complete solution to your team's fork.** Click **Fork** on GitHub, then clone your fork. Replace `YOUR_ACCOUNT` with the account or organization that owns your fork:

```sh
git clone https://github.com/YOUR_ACCOUNT/metalearning-llmaaj.git
cd metalearning-llmaaj
```

Commit the source code, a Dockerfile, dependency specifications, and instructions for building and running both `adapt` and `eval`. Include any files needed to reproduce the solution, and document configuration through environment variables. Keep credentials out of the repository.

After committing your solution, push it to your fork and record the exact commit:

```sh
git push origin HEAD
git rev-parse HEAD
```

Send the organizers **the URL of your fork and the commit SHA**. The organizers will evaluate that submitted version. Make sure they can access your fork and fetch the submitted commit.

Your Docker image must support two commands, `adapt` and `eval`, with this interface:

| Stage | Inputs | Outputs |
|---|---|---|
| `adapt` | Labeled training examples at `/input/train.jsonl`. | Adaptation artifacts in `/artifacts`, including the automatically generated explanation for the jury. |
| `eval` | Unlabeled test examples at `/input/test.jsonl` and the saved `/artifacts` directory. | `/output/predictions.csv` with exactly the columns `item_id,label`, as shown above. |

`eval` runs in a separate container, so persist everything it needs from `adapt` in `/artifacts`.

## Final evaluation

**You submit a runnable solution; the organizers run it themselves and calculate the official result.** Local scores help you develop your pipeline.

1. The organizers clone your fork, check out the submitted commit, and build your Docker image using the supplied instructions.
2. They prepare a new, labeled private training set and an unlabeled private test set. Private test labels remain with the organizers and are never passed to your solution.
3. They run `adapt` on the private training set with a fresh, empty artifacts directory. This stage learns the expert's criteria and automatically produces both prediction artifacts and an explanation for the jury.
4. They run `eval` in a separate container using the private test inputs and those saved artifacts. Your solution writes one `good` or `bad` prediction for every test ID.
5. The organizers validate the prediction file and calculate `acc = macro_topic_accuracy` against the hidden labels. The score is printed and saved to a file.
6. The jury evaluates the explanation generated during that same adaptation run using the five criteria below. The organizers combine `acc` and the jury points using **Final score = 10 × acc + 0.4 × I**.

## Automated metric

The automated metric is **`macro_topic_accuracy`** (`acc`): calculate accuracy separately for each of the four topics, then average those four values. Every topic has equal weight, regardless of its number of examples.

```text
acc = (accuracy_credit + accuracy_financial_position + accuracy_savings + accuracy_spending) / 4
score = macro_topic_accuracy = acc
```

Scores range from 0 to 1; higher is better. For example, topic accuracies of 0.8, 0.7, 0.9, and 0.6 give `acc = 0.75`. Model/API cost does not affect the score, and no cost reports are required for evaluation.

The bundled `src/evaluate.py` calculates this metric directly: its `macro_topic_accuracy` and `score` fields contain the same value. The organizer evaluator uses the same metric. Jury points are added separately as described below.

The scorer also reports overall accuracy, Cohen's kappa, and mean kappa across topics as diagnostics. In confusion matrices, rows represent ground-truth labels and columns represent predictions; the class order is `good`, `bad`. If kappa has a zero denominator, it is reported as 0.

Brier score and log loss are not calculated because solutions output labels only. The scorer evaluates one saved set of predictions without rerunning the model. Identical predictions against the same test set produce identical scores. This does not imply that the solution itself is deterministic.

A difference of a few thousandths in the score is not, by itself, evidence of a statistically significant advantage.

## Project defense and final score

**Final score = 10 × acc + 0.4 × I**, where **acc = `macro_topic_accuracy`** and **I ∈ [0, 5]** is the sum of points in the table below.

The automated component contributes up to 10 points and the jury component up to 2 points. Final scores range from 0 to 12. For example, `acc = 0.80` and `I = 4` give a final score of `10 × 0.80 + 0.4 × 4 = 9.6`.

The jury evaluates an artifact automatically generated by `adapt` on a new training set. Include this explanation alongside the artifacts used for prediction. The examples below are hypothetical illustrations for a text-to-`good`/`bad` task.

The interpretability score **`I` ranges from 0 to 5** and is the sum of five binary criteria:

| Criterion | +1 point | 0 points |
|---|---|---|
| **Clear criteria** | "Unsupported claims indicate a bad response; asking for clarification when essential data are missing indicates a good response." A decision tree with understandable conditions also qualifies. | "The model assesses text quality," or a list of unexplained technical features. |
| **Clear priorities** | "A factual error outweighs completeness." For a decision tree, the order of conditions and thresholds is visible. | Criteria are listed, but it is unclear how conflicting criteria produce the final `good`/`bad` label. |
| **Links to training examples** | Specific texts, their expert labels, and an explanation of which criterion each example illustrates are provided. | Only invented examples are given, or a generic claim such as "the training data support this." |
| **Clear boundaries** | "Labels for short responses are inconsistent: similar texts received different labels," supported by examples from the training set. | "The model sometimes makes mistakes," without specific conditions or examples. |
| **Consistency with the implementation** | The described tree is actually used for prediction; the displayed rules appear in the final prompt; a table of weights matches the model. | A polished explanation is attached separately, but its connection to the working solution is not demonstrated. |

**Rules for the jury:** each row receives exactly 0 or 1 point. The format is unrestricted: text, rules, a tree, a table, or a visualization. Length and literary quality earn no additional points; a classical model can receive all five points.

"Common sense" here means checking whether the explanation is understandable. An unusual expert preference is acceptable when it is supported by the labels.

## Check the evaluation scripts

```sh
python3 -m unittest discover -s src -p 'test_*.py'
```
