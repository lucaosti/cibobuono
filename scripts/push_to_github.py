"""
push_to_github.py — Commit and push updated JSON data files to GitHub.

Commits only the data/ directory JSON files and pushes to the current branch.
"""

import subprocess

from scripts.utils import DATA_DIR, PROJECT_ROOT, setup_logging, today_str

logger = setup_logging("git_push")


def git_commit_and_push(message: str | None = None) -> bool:
    """
    Stage data/*.json, commit with a descriptive message, and push.
    
    Returns True if successful, False otherwise.
    """
    if message is None:
        message = f"pipeline: update data {today_str()}"

    try:
        # Check if we're in a git repo
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"Not a git repository or git error: {result.stderr}")
            return False

        # Stage data files
        json_files = list(DATA_DIR.glob("*.json"))
        if not json_files:
            logger.info("No JSON files to commit")
            return True

        for f in json_files:
            rel_path = f.relative_to(PROJECT_ROOT)
            subprocess.run(
                ["git", "add", str(rel_path)],
                cwd=str(PROJECT_ROOT),
                timeout=10,
            )

        # Check if there are staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True,
            cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("No changes to commit")
            return True

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"Commit failed: {result.stderr}")
            return False
        logger.info(f"Committed: {message}")

        # Push
        result = subprocess.run(
            ["git", "push"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )
        if result.returncode != 0:
            logger.error(f"Push failed: {result.stderr}")
            return False

        logger.info("Pushed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Git operation timed out")
        return False
    except Exception as e:
        logger.error(f"Git error: {e}")
        return False


if __name__ == "__main__":
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else None
    success = git_commit_and_push(msg)
    if success:
        print("Git push completed successfully")
    else:
        print("Git push failed")
        sys.exit(1)
