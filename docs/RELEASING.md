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

**Version policy:** `semantic-release` manages the **patch** component
(`feat`/`fix`/`perf`/`refactor` → patch, per `pyproject.toml`); the package
stays on `0.x` (`allow_zero_version = true`). Bump the **minor/major** manually
(see below) to track the Universal release the dtypes target.

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
- **Cut an intentional minor/major.** First set `version` in `pyproject.toml`,
  commit it, then create the release at that version.

> `gh release create` authenticates with your **user** token, whose event
> *cascades* to the `release` trigger. A release created by CI's `GITHUB_TOKEN`
> would **not** cascade — which is why the automated path builds+publishes inside
> the same run rather than relying on a second workflow.

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
