"""``support-agent`` command-line interface.

Subcommands:
  demo       triage one message end-to-end (works with no API key)
  triage     triage a message given --text/--brand or --file (jsonl)
  classify   intent only, for a --text or --file
  eval       run the evaluation suites and refresh docs/report.md
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.panel import Panel

from .config import CONFIG

console = Console()


def _print_triage(t) -> None:
    body = [
        f"[bold]brand[/bold]        {t.brand}",
        f"[bold]intent[/bold]       {t.intent.intent}  "
        f"(conf {t.intent.confidence:.2f}, margin {t.intent.margin:.2f}, src {t.intent.source})",
        f"[bold]sentiment[/bold]    {t.signals.sentiment:+.2f}"
        + ("  [red](anger)[/red]" if t.signals.anger_flag else ""),
        f"[bold]retrieval[/bold]    best sim {t.signals.retrieval_max_sim:.2f}, "
        f"{len(t.reply.exemplars) if t.reply else 0} exemplars",
    ]
    if t.signals.compliance_hits:
        body.append(f"[bold]compliance[/bold]   {', '.join(t.signals.compliance_hits)}")
    if t.signals.pii_flags:
        body.append(f"[bold]pii[/bold]          {', '.join(t.signals.pii_flags)}")
    if t.reply:
        body.append("")
        body.append(f"[bold]draft reply[/bold]\n{t.reply.text}")
        if not t.reply.grounded:
            body.append(f"[red]grounding warning:[/red] {'; '.join(t.reply.groundedness_notes)}")
    body.append("")
    color = "green" if t.escalation.decision == "auto_send" else "yellow"
    body.append(f"[bold {color}]DECISION: {t.escalation.decision.upper()}[/bold {color}]")
    body.append(f"[italic]{t.escalation.reason}[/italic]")
    body.append("")
    body.append("[dim]rule trace: " + ", ".join(
        f"{rt.name}{'✓' if rt.fired else '·'}" for rt in t.escalation.trace
    ) + "[/dim]")
    console.print(Panel("\n".join(body), title="triage", border_style=color))


def _load_agent(args):
    from .pipeline import SupportAgent

    return SupportAgent.load()


def cmd_demo(args) -> int:
    from .pipeline import SupportAgent

    agent = SupportAgent.load()
    samples = [
        ("NimbusAir", "Where is my order 48213391? It's been 6 days and tracking hasn't moved."),
        ("PixelBank", "I was charged twice for the same thing and want the duplicate refunded now."),
        ("Streamly", "the app keeps crashing on the payments tab since the update, android 14"),
        ("FreshCart", "Just wanted to say your agent was lovely today, thank you!"),
        ("BoltMobile", "If this isn't fixed today I'm filing a chargeback and calling my lawyer."),
    ]
    for brand, text in samples:
        console.rule(f"[dim]{brand}[/dim]")
        console.print(f"[cyan]customer:[/cyan] {text}")
        _print_triage(agent.triage(text, brand))
    return 0


def cmd_triage(args) -> int:
    agent = _load_agent(args)
    if args.file:
        for line in open(args.file):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = agent.triage(row["text"], row.get("brand", "unknown"),
                             history=row.get("history"))
            print(json.dumps({
                "id": row.get("id"),
                "intent": t.intent.intent,
                "confidence": round(t.intent.confidence, 3),
                "decision": t.escalation.decision,
                "reason": t.escalation.reason,
                "reply": t.reply.text if t.reply else None,
            }, ensure_ascii=False))
        return 0
    if not args.text:
        console.print("[red]provide --text or --file[/red]")
        return 2
    _print_triage(agent.triage(args.text, args.brand or "unknown"))
    return 0


def cmd_classify(args) -> int:
    from .classify.model import IntentClassifier

    clf = IntentClassifier.load()
    texts = []
    if args.file:
        texts = [json.loads(line)["text"] for line in open(args.file) if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        console.print("[red]provide --text or --file[/red]")
        return 2
    for text, pred in zip(texts, clf.predict_batch(texts)):
        console.print(
            f"{pred.intent:<20} conf={pred.confidence:.3f} margin={pred.margin:.3f}  "
            f"[dim]{text[:70]}[/dim]"
        )
    return 0


def cmd_eval(args) -> int:
    from eval.report import run_and_render  # noqa: PLC0415 - optional import path

    run_and_render(limit=args.limit)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="support-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version="support-agent 0.1.0")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="triage a handful of built-in examples").set_defaults(func=cmd_demo)

    t = sub.add_parser("triage", help="full triage for a message or a jsonl file")
    t.add_argument("--text")
    t.add_argument("--brand")
    t.add_argument("--file")
    t.set_defaults(func=cmd_triage)

    c = sub.add_parser("classify", help="intent only")
    c.add_argument("--text")
    c.add_argument("--file")
    c.set_defaults(func=cmd_classify)

    e = sub.add_parser("eval", help="run eval suites and refresh docs/report.md")
    e.add_argument("--limit", type=int, default=None)
    e.set_defaults(func=cmd_eval)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _ = CONFIG  # touch config so env is read once, early
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
