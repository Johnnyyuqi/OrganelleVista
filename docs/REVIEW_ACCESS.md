# Private repository and reviewer access

The source repository is intended to remain **private**. Local Git initialization does not create a GitHub repository; verify the remote URL and visibility after creation.

## Create the private GitHub repository

After installing GitHub CLI and authenticating locally (do not paste a token into chat):

```bash
gh auth login --hostname github.com
cd /path/to/OrganelleVista
# Set the GitHub user or organization that should own the repository.
export REPO_OWNER=YOUR_GITHUB_USER_OR_ORG
gh repo create "$REPO_OWNER/OrganelleVista" --private --source=. --remote=origin --push
gh repo view "$REPO_OWNER/OrganelleVista" --json nameWithOwner,url,isPrivate
```

The final check must report `isPrivate: true`. If a repository with this name already exists, inspect it before adding a remote or pushing. Do not make the source repository public for peer review.

## Option A: anonymous review ZIP

```bash
python scripts/export_review.py
# Optionally replace additional identifying text:
python scripts/export_review.py --redact YOUR_NAME --redact YOUR_INSTITUTION
```

The export is written to ignored `review_exports/anonymous-code.zip`, with a SHA256 sidecar. It includes selected Git-indexed source files, reads their current working-tree contents, removes Git history and local outputs, uses fixed ZIP metadata, and replaces the project name and local home path. Training code, dependency licenses, and CLIP vocabulary are retained. Original-machine settings remain in ignored `.local/` and are not exported.

Inspect the ZIP for author names, affiliations, identifying URLs, and comments before submission. Automated replacement is not a guarantee of anonymity. Submit this ZIP through the venue's supplementary-material system to avoid granting repository access or exposing the repository owner. No online link is created by the export script.

## Option B: anonymous browsing link

[Anonymous GitHub](https://anonymous.4open.science/) provides anonymized repository browsing. Its [source documentation](https://github.com/tdurieux/anonymous_github) describes configurable identity-term replacement. For private sources, its documented [GitHub App flow](https://github.com/tdurieux/anonymous_github/blob/main/docs/github-app-setup.md) requires an installation with access to the selected private repository; actual availability depends on the hosted instance.

After the private repository exists, authorize only the intended repository through the service's browser flow, select a fixed review revision where available, configure identity terms, and inspect the result in a signed-out browser. Share only the generated anonymous URL. A private source does not make the anonymous view access-controlled: assume anyone with the generated link can read the shared snapshot unless the service explicitly provides stronger controls. The service receives access to source contents. No service connection or anonymous URL has been created by the local preparation step.

## Option C: named GitHub collaborators

Use GitHub repository Settings → Collaborators to invite named reviewers only when their identities are known and this is compatible with the review process. This exposes the repository owner and is **not anonymous**. No invitations have been sent.

Personal-account private repositories grant collaborators write access rather than a read-only reviewer role. If read-only GitHub access is required, use an organization repository with an appropriate role. See GitHub's [personal repository permissions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository).

## Local-only files

`outputs/`, model weights, data, credentials, `.local/`, and `review_exports/` are excluded from Git. The existing experiment output directory was retained during the folder rename. The source repository contains known training issues identified in review; renaming and packaging did not fix those algorithms or establish GPU correctness.
