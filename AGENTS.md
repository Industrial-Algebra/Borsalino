# Borsalino — Agent Operating Guide

## Gitflow (Non-Negotiable)

Borsalino follows IA gitflow as defined in the
[`ia-gitflow`](https://github.com/Industrial-Algebra/ia-toolkit/blob/main/skills/ia-gitflow/SKILL.md)
skill. Read it before touching branches.

### Branch Model

```
feature/* ──PR──▶ develop ──release PR──▶ main ──tag v*──▶ publish
                     ▲                                        │
                     └──────── backmerge (merge commit) ──────┘
```

### Hard Rules

1. **Never push directly to `main` or `develop`.** Both are protected.
   No direct pushes — not "just a CI fix", not "a one-liner", not "it's faster".
   Branch it, PR it, let CI run. This is enforced by GitHub branch protection.

2. **Every release to `main` is followed by a `main → develop` backmerge**
   using a merge commit (never squash). This is the last step of releasing,
   not an optional chore.

3. **Release-only commits (version bump, changelog dating) live on a
   `release/*` branch**, not on `develop` or `main`.

### What went wrong before (do not repeat)

- **Direct pushes to main**: commits `17fec79` and `e835219` were pushed
  directly to `main` during CI emergencies, bypassing review. Branch protection
  now prevents this mechanically.
- **Silent `develop` recreation**: when `develop` disappeared (due to
  `delete_branch_on_merge = true`), previous sessions silently recreated it
  from `main` without investigating or reporting the root cause. That setting
  is now disabled. If `develop` is ever missing, **investigate why before
  recreating it**.
- **Skipped backmerges**: release PRs were merged without backmerging `main`
  to `develop`, causing the branches to diverge in history.

### Branch Protection

Both `main` and `develop` have:
- Required PR review (1 approval)
- Required status checks (Format, Clippy, Test, Documentation)
- `enforce_admins: true` — rules apply to everyone
- `allow_force_pushes: false`
- `allow_deletions: false`

### Release Checklist

1. Feature PRs merged to `develop`
2. Version bump on a `release/*` branch off `develop`
3. Release PR: `release/*` → `main`
4. User reviews and merges
5. Tag `v*` on main → triggers publish workflow
6. **Backmerge** `main → develop` (merge commit)
7. Announce via `ia-website` skill

## Coding Standards

Follow the
[`ia-coding-standards`](https://github.com/Industrial-Algebra/ia-toolkit/blob/main/skills/ia-coding-standards/SKILL.md)
skill: TDD (test first), phantom types, `Result` not panic, exhaustive matching,
feature gates additive only, every public item documented.

## License

Apache-2.0. See `LICENSE` and `CONTRIBUTING.md`.
