## Summary

Describe what changed and why.

## Scope

- [ ] Environment or task behavior
- [ ] API or configuration
- [ ] Training or grading
- [ ] Dashboard
- [ ] Documentation
- [ ] Video asset

## Validation

List the commands you ran and their results.

```text
pytest
```

## Compatibility and risk

Describe any changes to observations, actions, rewards, termination behavior, endpoints, configuration keys, or security controls. State whether the change is backward compatible.

## Data and security

Confirm that the change uses synthetic or non-sensitive data and that no credentials, `.env` files, model checkpoints, caches, or generated local dependencies are included.

## Checklist

- [ ] Tests or validation were run.
- [ ] Documentation was updated where behavior or commands changed.
- [ ] Generated artifacts and secrets are excluded unless intentionally required.
- [ ] The change preserves deterministic seeded behavior where applicable.
