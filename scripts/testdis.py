"""Robust test runner for the Discord notifier.

Features:
- Loads `.env` from the repository root if python-dotenv is available.
- Attempts standard import `scripts.discord_notify`.
- Falls back to loading `discord_notify.py` directly from the `scripts/`
  directory if package imports are not available.
- Runs `send_discord_message` and prints any exception instead of crashing.
"""

from pathlib import Path
import sys
import importlib.util


def load_env_from_repo_root():
	try:
		from dotenv import load_dotenv
	except Exception:
		return
	repo_root = Path(__file__).resolve().parents[1]
	dotenv_path = repo_root / ".env"
	if dotenv_path.exists():
		load_dotenv(dotenv_path)


def import_notifier():
	# Try package import first
	try:
		from scripts.discord_notify import send_discord_message
		return send_discord_message
	except Exception:
		pass

	# Fallback: load module from file path
	script_dir = Path(__file__).resolve().parent
	mod_path = script_dir / "discord_notify.py"
	if not mod_path.exists():
		raise ModuleNotFoundError("discord_notify module not found")

	spec = importlib.util.spec_from_file_location("discord_notify_local", str(mod_path))
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return getattr(module, "send_discord_message")


def main():
	load_env_from_repo_root()
	send = import_notifier()
	try:
		send("test message from run_batch")
		print("send_discord_message executed (no exception).")
	except Exception as e:
		print(f"send_discord_message failed: {e}")


if __name__ == "__main__":
	main()