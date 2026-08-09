# Releasing `universal_dtypes`

All publishing runs from a single workflow, [`.github/workflows/wheels.yml`](../.github/workflows/wheels.yml),
with three triggers. Publishing is **keyless** — [PyPI Trusted Publishing
(OIDC)](https://docs.pypi.org/trusted-publishers/), no API tokens.

| Goal | Trigger | Command | Publishing job |
|---|---|---|---|
| **PyPI, automated** | push to `main` | merge a `feat:`/`fix:`/`perf:`/`refactor:` commit | `publish-pypi` |
| **PyPI, manual** | `release: published` | `gh release create vX.Y.Z --target main --generate-notes --latest` | `publish-pypi` |
| **TestPyPI, dry-run** | `workflow_dispatch` | `gh workflow run wheels.yml --ref main` | `publish-testpypi` |

## Release to PyPI (production)

Two paths, both ending in the `publish-pypi` job (OIDC → PyPI).

### 1. Automated — the normal path

Merge a [conventional commit](https://www.conventionalcommits.org/) to `main`:

```bash
# a feat:/fix:/perf:/refactor: commit, merged to main
git commit -m "fix: correct posit16 rounding"
```

On the push, the `release` job runs `semantic-release`, which:

1. computes the next version from the commits since the last tag,
2. updates `pyproject.toml` + `CHANGELOG.md`, commits `chore(release): vX.Y.Z`,
   tags `vX.Y.Z`, pushes, and creates the GitHub Release,
3. then the `build-wheels`/`build-sdist` jobs build for the new tag and
   `publish-pypi` uploads to PyPI — **all in the same run**.

Non-release commits (`docs:`/`chore:`/`ci:`/`build:`) are a no-op: the `release`
job runs, finds nothing to release, and the build/publish jobs skip.

**Version policy:** conventional semver on `0.x`. `semantic-release` bumps the
**minor** on a `feat` and the **patch** on `fix`/`perf`/`refactor` (per
`pyproject.toml`). While on `0.x` (`allow_zero_version = true`,
`major_on_zero = false`) a breaking change bumps the **minor**, not the major — a
**major release is deliberate and manual** (see *Major release*, §3).

### 2. Manual — first release, or a deliberate minor/major

```bash
gh release create v0.2.0 --target main --generate-notes --latest
```

Publishing a GitHub Release fires the `release: published` trigger → build +
`publish-pypi`. Use this to:

- **Bootstrap the first release.** `semantic-release` cannot create the first
  release from a repo with no tags (and would otherwise mis-default) — cut it by
  hand. A baseline tag `v0.1.0` already exists, so this only matters for a fresh
  repo.
- **Cut an intentional minor.** First set `version` in `pyproject.toml`, commit
  it, then create the release at that version. (For a **major**, see §3.)

> `gh release create` authenticates with your **user** token, whose event
> *cascades* to the `release` trigger. A release created by CI's `GITHUB_TOKEN`
> would **not** cascade — which is why the automated path builds+publishes inside
> the same run rather than relying on a second workflow.

### 3. Major release — and why the first stable one is `2.0.0`

A major (`1.0`/`2.0`/…) is an **API-stability commitment**, so it is always
deliberate and manual. Two project-specific rules:

- **The first major is `2.0.0`, not `1.0.0`.** An accidental `1.0.0` was published
  very early (a `semantic-release` config bug — `allow_zero_version` defaulted to
  false and force-escaped `0.x` to `1.0.0`) and then **yanked**. PyPI never frees
  a version number: yanked *or* deleted, `1.0.0` can never be re-uploaded, so any
  attempt to publish a real `1.0.0` is rejected as a duplicate. The first stable
  major therefore **skips it**: `0.x → 2.0.0`.
- **Automation will not do it for you.** With `major_on_zero = false`, a breaking
  change bumps the *minor* while on `0.x`, and there are no `major_tags`. Left to
  `semantic-release`, a forced major from `0.x` would compute `1.0.0` — which PyPI
  rejects. So force the jump to `2.0.0` once, by hand.

To cut it:

1. Set `version = "2.0.0"` in `pyproject.toml`, and add a hand-written `## v2.0.0`
   section to `CHANGELOG.md` (`semantic-release` won't compute this jump).
2. Commit, then create an **annotated** tag and release at that version:
   ```bash
   git tag -a v2.0.0 -m "v2.0.0"   # annotated: semantic-release adopts it as the baseline
   git push origin v2.0.0
   gh release create v2.0.0 --target main --generate-notes --latest  # -> build + publish-pypi
   ```
3. Once `v2.0.0` is the baseline, automation resumes normally: `fix` → `2.0.1`,
   `feat` → `2.1.0`, and a `feat!:` / `BREAKING CHANGE:` footer → `3.0.0`
   (breaking-change detection is independent of the tag lists). The `0.x` notes
   and `major_on_zero` no longer apply; drop them at that point.

The yanked `1.0.0` is never revisited: `0.x → 2.0.0 → 2.x → 3.0.0`.

## Release to TestPyPI (dry-run)

One path — the `workflow_dispatch` trigger → `publish-testpypi`:

```bash
gh workflow run wheels.yml --ref main
# or: Actions → Build wheels → Run workflow → main
```

Builds the matrix and uploads to <https://test.pypi.org/project/universal-dtypes/>
with `skip-existing`. It **never** touches production PyPI and **never** bumps
the version — it publishes whatever `version` is currently in `pyproject.toml`.

Install-test what you published:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ universal_dtypes
```

(The `--extra-index-url` lets NumPy resolve from real PyPI while
`universal_dtypes` comes from TestPyPI.)

## One-time setup: Trusted Publishers

PyPI and TestPyPI are **separate registries** — register a publisher on **each**
(GitHub Actions provider):

- Owner: `stillwater-sc`
- Repository: `universal_dtypes` — **underscore**, matching the GitHub repo,
  not the PyPI-canonical `universal-dtypes`
- Workflow name: `wheels.yml`
- Environment name: `pypi`

All four must match the OIDC token exactly, or the upload fails with
`invalid-publisher`. For a project that does not exist on the registry yet, use
the **pending publisher** flow; once it exists, use the project's *Publishing*
page.

## Gotchas

- **Immutability.** Both registries reject re-uploading an existing version.
  `publish-pypi` deliberately has **no** `skip-existing` (a duplicate fails
  loudly); `publish-testpypi` sets `skip-existing` (a re-dispatch at the same
  version is a no-op).
- **Iterating on TestPyPI.** Because it won't overwrite a version, bump to a
  throwaway dev version (e.g. `0.1.0.dev1`) on a scratch branch to test new
  artifacts.
- **First-release guard.** The `release` job refuses to auto-release when no
  version tag exists yet — bootstrap the first tag deliberately (manual release
  above). This prevents `semantic-release` from inventing a version on a fresh
  repo.
- **No reusable workflows.** PyPI Trusted Publishing does not support them, so
  all publishing lives directly in `wheels.yml`.

See [`docs/design.md`](design.md) for the package's design and the
`universal_dtypes` ↔ `mtl5` boundary.
