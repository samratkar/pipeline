import argparse

import config
from observability import get_langfuse
from prompts import DEFAULT_PROMPTS

# -----------------------------------
# Prompt Management - upload agent prompts to Langfuse
# -----------------------------------
#
#   python seed_prompts.py                 -> create v1 of any missing prompt
#   python seed_prompts.py --force         -> push a new version of every prompt
#   python seed_prompts.py --label staging -> label the new versions "staging"
#
# After seeding, edit prompts / promote versions in the Langfuse UI.
# The agents always load the version carrying PROMPT_LABEL (.env).


def prompt_exists(lf, name):

    try:
        lf.get_prompt(name, label=config.PROMPT_LABEL, cache_ttl_seconds=0)
        return True
    except Exception:
        return False


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--label", default=config.PROMPT_LABEL)
    parser.add_argument("--model", default=config.AGENT_MODEL,
                        help="Model stored in the prompt config")
    args = parser.parse_args()

    lf = get_langfuse()

    if lf is None:
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env first")

    for name, text in DEFAULT_PROMPTS.items():

        if not args.force and prompt_exists(lf, name):
            print(f"Skipping {name} (already exists)")
            continue

        prompt = lf.create_prompt(
            name=name,
            prompt=text,
            type="text",
            labels=[args.label],
            tags=["paper-review-saga"],
            config={"model": args.model},
            commit_message="Seeded from prompts.py"
        )

        print(f"Created {name} v{prompt.version} [{args.label}]")

    lf.flush()


if __name__ == "__main__":
    main()
