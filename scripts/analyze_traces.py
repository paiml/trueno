#!/usr/bin/env python3
"""Analyze exported OTLP trace data for syscall profiling."""
import sys
import json
from collections import defaultdict

def main():
    if len(sys.argv) != 2:
        print("Usage: analyze_traces.py <trace_file.json>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    syscalls = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0})

    for trace in data.get("data", []):
        for span in trace.get("spans", []):
            op = span.get("operationName", "")
            duration = next(
                (t["value"] for t in span.get("tags", []) if t["key"] == "syscall.duration_us"),
                0
            )
            if op.startswith("syscall:"):
                name = op.split(": ")[1] if ": " in op else op
                syscalls[name]["count"] += 1
                syscalls[name]["total_us"] += duration
                syscalls[name]["max_us"] = max(syscalls[name]["max_us"], duration)

    print(f"Traces: {len(data.get('data', []))}")
    print(f"Total syscalls: {sum(s['count'] for s in syscalls.values())}")
    print(f"Total time: {sum(s['total_us'] for s in syscalls.values())}μs\n")
    print("Top syscalls by time:")

    for name, stats in sorted(syscalls.items(), key=lambda x: x[1]["total_us"], reverse=True)[:10]:
        avg = stats["total_us"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {name:20s} {stats['count']:5d} calls  {stats['total_us']:8d}μs  avg: {avg:6.1f}μs")

if __name__ == "__main__":
    main()
