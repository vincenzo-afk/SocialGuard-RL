# Contributing to SocialGuard-RL

Thank you for helping improve SocialGuard-RL. Contributions should keep the environment deterministic, the API contract explicit, and the project’s synthetic-data boundary clear.

## Development setup

Create a virtual environment and install the repository dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a focused branch name such as `feat/task-name`, `fix/issue-name`, or `docs/topic-name`. Keep unrelated changes out of the same pull request.

## Before opening a pull request

Run the tests relevant to the changed code and then run the full suite when practical:

```bash
pytest
```

If the change touches the HTTP server, exercise `/healthz`, `/reset`, and `/step` locally. If the change touches the video or README, run the HyperFrames check from `video/`:

```bash
cd video
npm install
npm run check
```

Update the README, YAML configuration, or `openenv.yaml` whenever a public command, endpoint, task, action, observation, reward, or configuration key changes.

## Pull requests

A pull request should explain the motivation, summarize the implementation, and list the checks that were run. Call out changes to deterministic seeds, reward formulas, task termination, API response shapes, security behavior, or generated artifacts. Include a small reproduction or test case for bug fixes.

Do not commit credentials, `.env` files, model checkpoints, graph caches, local virtual environments, `node_modules`, or generated debugging output. Preserve the project’s synthetic-data scope and do not add integrations that send real user or moderation data without an explicit design and safety review.

## Commit expectations

Use concise, imperative commit subjects. Keep each commit reviewable and avoid committing generated dependencies. Changes should be attributable to the repository owner or the contributor who authored them; do not add automated-agent attribution to commits.
