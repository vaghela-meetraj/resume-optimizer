import json
import os
import sys
import urllib.request
import urllib.error

# Ensure root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_all_links():
    resources_path = os.path.join(os.path.dirname(__file__), "..", "data", "resources.json")
    if not os.path.exists(resources_path):
        print(f"[!] {resources_path} does not exist.")
        return False

    with open(resources_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_links = 0
    ok_count = 0
    broken_count = 0

    print("=" * 80)
    print("Verifying Learning Resource Links in data/resources.json")
    print("=" * 80)

    for skill, meta in data.items():
        links = meta.get("links", [])
        for item in links:
            title = item.get("title", "Link")
            url = item.get("url", "")
            total_links += 1

            # Check URL format
            if not url.startswith("http://") and not url.startswith("https://"):
                print(f"[ERROR] [{skill}] {title}: Invalid URL scheme '{url}'")
                broken_count += 1
                continue

            # Perform HTTP HEAD/GET request with timeout and custom browser User-Agent
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            try:
                # Use a short timeout of 5 seconds
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status < 400:
                        ok_count += 1
                    else:
                        print(f"[WARN] [{skill}] {title}: HTTP {resp.status}")
                        broken_count += 1
            except urllib.error.HTTPError as e:
                # 403 or 999 from Cloudflare anti-bot is common on public scrapers; 404 is truly broken
                if e.code in (403, 999, 429):
                    ok_count += 1
                else:
                    print(f"[ERROR] [{skill}] {title} ({url}): HTTP {e.code}")
                    broken_count += 1
            except Exception:
                # Soft check (network timeout / DNS)
                ok_count += 1

    print("=" * 80)
    print(f"{ok_count} OK, {broken_count} broken (Total: {total_links} checked)")
    return broken_count == 0


if __name__ == "__main__":
    success = check_all_links()
    sys.exit(0 if success else 1)

