import os, csv
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
OUT  = os.path.join(ROOT, "inference", "drive_urls.csv")

CLASSES = ["Dresses","mixed","tshirt","women fashion"]

# ---------- AUTH ----------
from pydrive2.auth import GoogleAuth
from pydrive2.settings import LoadSettingsFile

settings_yaml = """
client_config_backend: file
client_config_file: client_secrets.json

save_credentials: True
save_credentials_backend: file
save_credentials_file: credentials.json

get_refresh_token: True
oauth_scope:
  - https://www.googleapis.com/auth/drive

auth_type: oauth2
oauth_flow:
  redirect_uri: http://localhost:8080/
"""

with open("settings.yaml", "w") as f:
    f.write(settings_yaml)

gauth = GoogleAuth("settings.yaml")
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)


# ---------- SCAN ----------
rows = []

for cls in CLASSES:
    folder_id = input(f"Enter Drive folder ID for {cls}: ")
    query = f"'{folder_id}' in parents and trashed=false"
    files = drive.ListFile({'q': query}).GetList()
    print(f"{len(files)} files found in {cls}")

    for f in files:
        url = f"https://drive.google.com/uc?export=view&id={f['id']}"
        rows.append((url, cls))

# ---------- SAVE ----------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url","label"])
    writer.writerows(rows)

print("drive_urls.csv created at:", OUT)
