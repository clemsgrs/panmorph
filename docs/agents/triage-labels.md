# Agent triage labels

This repository uses the canonical agent workflow labels without aliases.

| Workflow state | GitHub label |
|---|---|
| Eligible for dispatch | `ready-for-agent` |
| Implementation active | `agent:in-progress` |
| Pull request in the merge train | `agent:in-review` |
| Gate failed and awaiting another attempt | `agent:needs-changes` |

Closed issues represent delivered work. Dependencies are recorded in each issue's
`## Blocked by` section.
