# Security Notes

This project requires API keys (Anthropic, optionally Hugging Face and Weights
& Biases). Treat them with the same care as a password — leaking one to a
public GitHub repo can result in a multi-thousand-dollar bill within hours.

## Before you push the repo public

1. **Set a hard spending cap** on the Anthropic console.
   `Settings → Billing → Usage limits`. A $50 monthly cap is sensible for
   personal-project work. Email alerts at a lower threshold (e.g. $10) give
   early warning if something's wrong.

2. **Use a project-scoped key.** Generate a separate Anthropic key named
   `logistics-qa-lora` rather than reusing a key you already have. If it
   leaks, you rotate one thing, not your whole life.

3. **Confirm `.env` is gitignored.** Run `git check-ignore .env` — it should
   print `.env`. If it doesn't, `.env` will be committed.

4. **Install the pre-commit hook.**
   ```bash
   pip install pre-commit
   pre-commit install
   ```
   This runs `gitleaks` on every commit and refuses any commit containing
   anything resembling a credential.

5. **Run gitleaks once manually before the first push.**
   ```bash
   pip install gitleaks    # or download from github
   gitleaks detect --source .
   ```

## What's in the gitignore

`.env`, `*.key`, `*.pem`, `secrets.json`, all checkpoint files, all generated
datasets, and W&B local caches. See `.gitignore` for the full list.

## What to do if a key leaks

1. **Revoke the key immediately** at the provider's console
   (Anthropic, OpenAI, etc.). This is a single click and takes effect in seconds.
2. **Generate a new key** with a new name.
3. **Update your local `.env`** with the new key.
4. **If the leak was in the most recent commit**, `git filter-repo --invert-paths --path .env` or just delete and re-initialize the repo if it's small.
5. **Check the provider's usage dashboard** for any unfamiliar activity, and
   open a support ticket if there is any — providers often forgive charges
   resulting from a leaked key if you report it promptly.

## Server deployment

The FastAPI server in `server/app.py` is **not** designed to be exposed to the
public internet. It has no authentication and no per-request rate limiting.
For portfolio purposes, run it locally. If you must deploy it publicly:

- Put it behind authentication (an API gateway, Cloudflare Access, or even
  basic auth as a stopgap).
- Add rate limiting (e.g. `slowapi`) — without it, a single client can burn
  through your GPU budget.
- Set `USE_MOCK=true` for any public demo where you don't want real model
  costs.
