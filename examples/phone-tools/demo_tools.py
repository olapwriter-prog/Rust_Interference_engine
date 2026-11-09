#!/usr/bin/env python3
"""Deterministic stand-ins for phone tools (used by tests and the /v1/ask demo).

Usage: demo_tools.py TOOL   (arguments arrive as JSON on stdin)
"""
import json
import sys
import time

tool = sys.argv[1]
try:
    args = json.loads(sys.stdin.read() or "{}")
except ValueError:
    args = {}

def out(v):
    print(json.dumps(v, ensure_ascii=False))

if tool == "weather":
    city = str(args.get("city") or "Lyon")[:40]
    out({"city": city, "temperature_c": 18.4, "condition": "Partly cloudy", "humidity": 0.62, "wind_kmh": 14,
         "next_hours": [{"time": "15:00", "temp_c": 19, "rain": "10%"}, {"time": "18:00", "temp_c": 16, "rain": "35%"}, {"time": "21:00", "temp_c": 13, "rain": "60%"}]})
elif tool == "calendar":
    out([{"time": "09:30", "title": "Stand-up", "place": "Video call"},
         {"time": "12:00", "title": "Lunch with Sam", "place": "Café Lumière"},
         {"time": "16:00", "title": "Dentist", "place": "12 Rue Victor Hugo"}])
elif tool == "tasks":
    out({"open": 3, "due_today": ["Pay electricity bill", "Send slides to Priya", "Buy groceries"]})
elif tool == "balance":
    out({"account": "Everyday", "balance": 2431.75, "currency": "EUR", "pending": -84.2})
elif tool == "spending":
    out([{"category": "Groceries", "amount": 182.4}, {"category": "Transport", "amount": 64.0}, {"category": "Eating out", "amount": 97.35}])
elif tool == "news":
    out(["City council approves new bike lanes", "Local team wins regional final", "Rail strike called off for Friday"])
elif tool == "files":
    q = str(args.get("query") or "")
    out([{"name": "Trip plan.pdf", "modified": "2026-11-02", "size_kb": 412}, {"name": f"Notes {q}".strip(), "modified": "2026-11-08", "size_kb": 3}])
elif tool == "battery":
    out({"percentage": 76, "status": "DISCHARGING", "temperature": 31.2, "health": "GOOD"})
elif tool == "steps":
    out(8432)
elif tool == "location":
    out({"latitude": 45.764, "longitude": 4.8357, "accuracy_m": 12, "place": "Lyon, France"})
elif tool == "messages":
    out([{"from": "Sam", "text": "Running 5 min late", "time": "11:52"}, {"from": "Bank", "text": "Card payment €12.40", "time": "10:03"}])
elif tool == "fail":
    print("Error: service unavailable (HTTP 503)", file=sys.stderr)
    sys.exit(2)
elif tool == "slow":
    time.sleep(30)
    out({"never": True})
elif tool == "text":
    print("Plain text result: no JSON here.\nSecond line.")
elif tool == "html":
    out({"title": "<script>alert('x')</script>", "body": "5 < 6 & \"quoted\""})
elif tool == "big":
    sys.stdout.write("x" * (1024 * 1024))
elif tool == "echo":
    out({"received": args})
else:
    print(f"unknown demo tool {tool}", file=sys.stderr)
    sys.exit(1)
