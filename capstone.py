

from __future__ import annotations

from manager_agent import run


def main() -> None:
    print("Multi-Agent RAG System")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not query:
            continue

        try:
            result = run(query)

            print(f"\nRoute: {result.get('route', 'unknown')}")
            print(f"\nAnswer:\n{result.get('answer', '')}\n")

        except Exception as exc:
            print(f"\nExecution failed: {exc}\n")


if __name__ == "__main__":
    main()

