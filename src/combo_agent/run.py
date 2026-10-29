from pathlib import Path
from sheldon.core.io import load_dotenv

def main():
    load_dotenv(Path.cwd() / ".env")
    import combo_agent.benchmark  # noqa: F401  registers combo-mechanism
    from sheldon.cli import main as sheldon_main
    return sheldon_main()

if __name__ == "__main__":
    raise SystemExit(main())