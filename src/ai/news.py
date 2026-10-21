"""Free headline source: Google News RSS. No API key, no cost.

Fetches recent headlines for a stock so the AI sentiment veto has something
real to read. Uses only the standard library (urllib + xml) so it adds no
dependency. Results are cached briefly by the caller (GeminiHelper).
"""
from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_UA = "Mozilla/5.0 (compatible; TradingBot/1.0)"

# Map NSE tradingsymbols to a friendlier news query where useful.
_NAME_HINT = {
    "RELIANCE": "Reliance Industries", "GAIL": "GAIL India", "TCS": "Tata Consultancy",
    "INFY": "Infosys", "SBIN": "State Bank of India", "HDFCBANK": "HDFC Bank",
}


def fetch_headlines(symbol: str, limit: int = 6, timeout: float = 8.0) -> list[str]:
    name = _NAME_HINT.get(symbol.upper(), symbol)
    query = urllib.parse.quote(f"{name} stock NSE share price")
    url = (f"https://news.google.com/rss/search?q={query}"
           f"&hl=en-IN&gl=IN&ceid=IN:en")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception:
        return []
    try:
        root = ET.fromstring(data)
    except Exception:
        return []
    titles = []
    for item in root.iter("item"):
        t = item.findtext("title")
        if t:
            titles.append(t.strip())
        if len(titles) >= limit:
            break
    return titles
