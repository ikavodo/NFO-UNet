---
name: repo-cleanup
description: Decide what to commit before any commit in this repo. Use before committing (or when the user says "commit", "clean up the repo", "what should go into git"). Classifies every changed/untracked path into commit / ignore / ask, so only files that are useful later end up in git (the repo is public on GitHub).
---

# Repo cleanup before a commit

The repo (`github.com/ikavodo/NFO-UNet`) is **public**. Commit only what someone (the user, a collaborator, a future
session) would need later to understand or reproduce the work. Never stage with `git add -A` / `git add .`.

## 1. Inventory

```bash
git status --short
git diff --stat
du -sh <each untracked dir>            # sizes of untracked paths
find <untracked dir> -type f | head -100
```

Read `CLAUDE.md` (data manifest: "nothing large is in git"; hand labels are the force-added exception) and `.gitignore`
before deciding.

## 2. Classify every path

| class | examples here | action |
|---|---|---|
| **source we wrote** | `benchmark/**/*.py`, `*.sbatch`, our `.tex` sources, docs `*.md` / `docs/briefing/index.html`, tests | commit |
| **small generated figure that a committed doc needs and that is not regenerable from git alone** (needs cluster data) | poster / briefing figures under ~1 MB each, referenced by a committed `.tex` / `.html` | commit, and make sure the script that makes it is committed too |
| **generated, regenerable or large** | masks (`results/**/masks`), CSVs under `results/`, logs, build PDFs, contact sheets, option sheets not used by any doc | do not commit; add to `.gitignore` if it will keep reappearing |
| **build / editor junk** | `*.aux *.log *.fls *.fdb_latexmk *.synctex.gz`, `*:Zone.Identifier`, `__pycache__/`, `.idea/` | never; add to `.gitignore` |
| **third-party / copyrighted / licensed** | institutional LaTeX templates (TUM `Ressourcen/`, `Vorlagen/`, `Beispiele/`), logos, figures copied from papers, papers themselves, datasets | **ask the user** (public repo); default is do not commit |
| **secrets / personal data** | tokens, `.env`, e-mail lists, raw personal data | never |
| **anything > 5 MB** | example PDFs, checkpoints, videos | never in git; mention where it lives instead |

Unused leftovers (e.g. figures a later draft no longer references) are not "useful later": leave them out.

## 3. Propose, then act

- Show the user a short table: **commit** / **ignore** / **ask**, with one reason per line.
- Ask only about the **ask** rows (one question, options), never re-ask what the rules decide.
- Stage explicit paths only (`git add <path> ...`). Add `.gitignore` rules for recurring junk in the same commit.
- `git status --short` after staging: nothing unintended staged, nothing intended left out.
- Commit with a message that says what and why; end with the attribution lines the session gives. Do not push unless
  the user asks.

## Tidy mode: redundant things already in git

When asked to "make order" in the repo (not only before a commit):

1. List tracked files by area: `git ls-files | awk -F/ '{print $1"/"$2}' | sort | uniq -c | sort -rn`.
2. For each doc / asset, check whether anything still points to it:
   `grep -rlF "$(basename <file>)" --include=*.md --include=*.html --include=*.py --include=*.sh --include=*.sbatch .`
   and whether it describes code that no longer exists (CLAUDE.md: removed code lives at tag
   `archive/pre-cleanup-2026-10-21`).
3. Candidates: unreferenced docs about removed code, superseded plans (a newer doc or the findings doc covers them),
   duplicate copies of a file, generated files tracked by mistake. Keep: the legacy U-Net as published (CLAUDE.md),
   hand labels, `trials.json`, logs the findings cite.
4. Show the candidate list with one reason each; delete only what the user approves (`git rm`), in its own commit
   whose message names the archive tag for anything removed.
