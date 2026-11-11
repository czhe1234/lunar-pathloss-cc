"""List official release filenames using existing credentials, without saving credentials."""
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home()/".kaggle")))/"kaggle.json"
    if (not username or not key) and config.is_file():
        credentials = json.loads(config.read_text())
        username, key = credentials.get("username"), credentials.get("key")
    if not username or not key:
        raise SystemExit("No existing Kaggle credentials are available")
    token = base64.b64encode((username+":"+key).encode()).decode()
    endpoint = "https://www.kaggle.com/api/v1/competitions/data/list/the-first-lunar-pathloss-radio-map-prediction-challenge"
    files = []
    next_page = None
    pages = 0
    while True:
        url = endpoint + ("?" + urlencode({"pageToken": next_page}) if next_page else "")
        request = Request(url, headers={"Authorization": "Basic "+token, "User-Agent": "CC-source-audit"})
        with urlopen(request, timeout=20) as response:
            data = json.load(response)
        pages += 1
        files.extend({k: entry[k] for k in ("name", "totalBytes", "creationDate") if k in entry}
                     for entry in data["files"])
        if not data.get("hasNextPageToken"):
            break
        following = data.get("nextPageToken")
        if not following or following == next_page:
            raise RuntimeError("Incomplete official file-list pagination")
        next_page = following
    result = {
        "url": endpoint, "utc": datetime.now(timezone.utc).isoformat(),
        "status": 200, "complete": True, "pages": pages, "files": files,
    }
    output = Path(__file__).with_name("kaggle_complete_file_list.json")
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"complete": True, "pages": pages, "files": len(files), "output": str(output)}))


if __name__ == "__main__":
    main()
