# Playwright Browser Tools

These programs drive the Cesium viewer to collect repeatable loading, resource, screenshot, and
visible-tile evidence. They are executable research tools, not `@playwright/test` test cases.

Dependencies are isolated in this directory. Install them from the repository root with `just sync`,
then invoke tools through their registered Typer tasks, for example:

```bash
just task analysis.profile-interactive-path --dry-run
just task quality.audit-visible-tiles --dry-run
```

Runtime inputs such as `VIEWER_URL`, `PROFILE_OUTPUT`, and `AUDIT_OUTPUT` are supplied through the
environment so generated evidence remains outside Git.
