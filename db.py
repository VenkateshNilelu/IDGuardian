"""
MongoDB Atlas connection + Analysis History / Saved Reports storage.

Moves what used to live only in the browser's localStorage (History, Saved
Reports) into a real database, so it survives a cleared cache and isn't
tied to one browser. Each browser still gets its own generated `client_id`
(see static/app.js) rather than requiring a login -- this project has no
auth system, so client_id is the closest thing to "whose history is this."

Follows the same graceful-degradation pattern as the semantic layer in
app.py: if MONGODB_URI isn't set or the cluster is unreachable, DB_AVAILABLE
stays False and the API routes return a clear 503 instead of the app
crashing at startup.
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError

load_dotenv()

MONGODB_URI = os.environ.get("MONGODB_URI", "").strip()

client: MongoClient | None = None
DB_AVAILABLE = False
DB_UNAVAILABLE_REASON = ""
history_col = None
saved_col = None

if not MONGODB_URI:
    DB_UNAVAILABLE_REASON = "MONGODB_URI is not set (copy .env.example to .env and fill it in)."
    print(f"WARNING: MongoDB unavailable ({DB_UNAVAILABLE_REASON})")
else:
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # fail fast at startup, not on first request
        db = client["idguardian"]
        history_col = db["history"]
        saved_col = db["saved_reports"]
        history_col.create_index([("client_id", 1), ("ts", DESCENDING)])
        saved_col.create_index([("client_id", 1), ("platform", 1), ("username", 1)], unique=True)
        DB_AVAILABLE = True
        print("MongoDB Atlas connected. History/Saved Reports are persisted server-side.")
    except PyMongoError as exc:
        DB_UNAVAILABLE_REASON = str(exc)
        print(f"WARNING: MongoDB unavailable ({exc}). History/Saved Reports will not persist.")


def _serialize(doc: dict) -> dict:
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    return doc


def list_history(client_id: str, limit: int = 50) -> list[dict]:
    cursor = history_col.find({"client_id": client_id}).sort("ts", DESCENDING).limit(limit)
    return [_serialize(d) for d in cursor]


def add_history(client_id: str, entry: dict) -> dict:
    doc = {**entry, "client_id": client_id, "ts": entry.get("ts") or int(time.time() * 1000)}
    result = history_col.insert_one(doc)
    doc["_id"] = result.inserted_id
    # Cap history at 50 most recent per client -- mirrors the old localStorage HISTORY_MAX.
    stale_ids = [d["_id"] for d in history_col.find({"client_id": client_id})
                 .sort("ts", DESCENDING).skip(50)]
    if stale_ids:
        history_col.delete_many({"_id": {"$in": stale_ids}})
    return _serialize(doc)


def delete_history_entry(client_id: str, entry_id: str) -> bool:
    from bson import ObjectId
    result = history_col.delete_one({"_id": ObjectId(entry_id), "client_id": client_id})
    return result.deleted_count > 0


def clear_history(client_id: str) -> int:
    return history_col.delete_many({"client_id": client_id}).deleted_count


def list_saved(client_id: str) -> list[dict]:
    cursor = saved_col.find({"client_id": client_id}).sort("ts", DESCENDING)
    return [_serialize(d) for d in cursor]


def add_saved(client_id: str, entry: dict) -> dict:
    doc = {**entry, "client_id": client_id, "ts": int(time.time() * 1000)}
    saved_col.update_one(
        {"client_id": client_id, "platform": doc["platform"], "username": doc["username"]},
        {"$set": doc}, upsert=True,
    )
    saved = saved_col.find_one({"client_id": client_id, "platform": doc["platform"], "username": doc["username"]})
    return _serialize(saved)


def delete_saved_entry(client_id: str, entry_id: str) -> bool:
    from bson import ObjectId
    result = saved_col.delete_one({"_id": ObjectId(entry_id), "client_id": client_id})
    return result.deleted_count > 0


def clear_saved(client_id: str) -> int:
    return saved_col.delete_many({"client_id": client_id}).deleted_count
