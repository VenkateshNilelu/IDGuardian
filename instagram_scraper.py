"""
Real Instagram profile scraping API — ported from the standalone Instagram
scrapper project (app/scraper/{web_api,instaloader_backend,agent}.py).

Only the scraping capability was carried over. Everything else that project did
(downloads/ folder, media files, dataset CSVs, job history, the multi-platform
registry, the FastAPI app/UI) was intentionally left behind: this module takes a
username or profile URL and returns a public profile as a plain dict, with no
disk writes and no other side effects.

Strategy, unchanged from the source project:
  1. Try Instagram's web_profile_info endpoint (curl_cffi / Chrome TLS) — fast, no login.
  2. Fall back to Instaloader if the web endpoint is blocked.
  3. If more than WEB_POST_PAGE_LIMIT posts were requested and the web backend
     succeeded, top up the post list via Instaloader (which paginates) while
     keeping the web response's more reliable profile fields.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from itertools import islice
from typing import Any
from urllib.parse import urlparse

INSTAGRAM_APP_ID = "936619743392459"
PROFILE_URL = "https://i.instagram.com/api/v1/users/web_profile_info/?username={username}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# web_profile_info returns one fixed page of ~12 posts and gives no anonymous way
# to page further; asking for more silently still returns 12, so requests beyond
# this are topped up via Instaloader, which does paginate.
WEB_POST_PAGE_LIMIT = 12

HANDLE_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")


def parse_instagram_target(raw: str) -> str:
    """Accept a bare username or an instagram.com profile URL; return the username."""
    raw = (raw or "").strip().lstrip("@")
    if "instagram.com" in raw:
        url = raw if "://" in raw else f"https://{raw}"
        parts = [p for p in urlparse(url).path.split("/") if p]
        raw = parts[0] if parts else raw
    return raw.strip()


def _parse_posts(user: dict[str, Any], max_posts: int) -> list[dict]:
    posts = []
    edges = (
        user.get("edge_owner_to_timeline_media", {}).get("edges")
        or user.get("edge_felix_video_timeline", {}).get("edges")
        or []
    )
    for edge in edges[:max_posts]:
        node = edge.get("node") or {}
        shortcode = node.get("shortcode") or ""
        caption_edges = (node.get("edge_media_to_caption") or {}).get("edges") or []
        caption = (caption_edges[0].get("node") or {}).get("text") or "" if caption_edges else ""
        is_video = bool(node.get("is_video"))
        ts = node.get("taken_at_timestamp")
        posts.append(
            {
                "shortcode": shortcode,
                "url": f"https://www.instagram.com/p/{shortcode}/" if shortcode else "",
                "caption": caption,
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
                "likes": (node.get("edge_liked_by") or node.get("edge_media_preview_like") or {}).get("count", 0) or 0,
                "comments_count": (node.get("edge_media_to_comment") or {}).get("count", 0) or 0,
                "is_video": is_video,
                "hashtags": re.findall(r"#(\w+)", caption),
                "mentions": re.findall(r"@(\w+)", caption),
            }
        )
    return posts


def _scrape_via_web(username: str, max_posts: int) -> dict:
    if not HANDLE_RE.match(username):
        raise ValueError("Invalid Instagram username")

    try:
        from curl_cffi import requests as cffi_requests

        session = cffi_requests.Session(impersonate="chrome")
    except Exception:
        import httpx

        session = httpx.Client(follow_redirects=True, timeout=60.0)

    headers = {
        "x-ig-app-id": INSTAGRAM_APP_ID,
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.instagram.com/{username}/",
        "Origin": "https://www.instagram.com",
    }

    try:
        resp = session.get(PROFILE_URL.format(username=username), headers=headers)
        if resp.status_code == 404:
            raise LookupError(f"User @{username} not found")
        if resp.status_code in (401, 403):
            raise PermissionError(
                "Instagram blocked the anonymous web request (403/401). "
                "Try again later, use a residential network, or rely on the Instaloader fallback."
            )
        if resp.status_code != 200:
            raise RuntimeError(f"Instagram returned HTTP {resp.status_code}")
        payload = resp.json() if hasattr(resp, "json") else json.loads(resp.content)
    finally:
        if hasattr(session, "close"):
            try:
                session.close()
            except Exception:
                pass

    user = (payload.get("data") or {}).get("user")
    if not user:
        raise LookupError(f"No public profile data for @{username}")

    posts_wanted = 0 if user.get("is_private") else max_posts
    posts = _parse_posts(user, posts_wanted) if posts_wanted > 0 else []

    return {
        "platform": "instagram",
        "username": user.get("username") or username,
        "user_id": str(user.get("id") or ""),
        "profile_url": f"https://www.instagram.com/{user.get('username') or username}/",
        "full_name": user.get("full_name") or "",
        "biography": user.get("biography") or "",
        "external_url": user.get("external_url"),
        "profile_pic_url": user.get("profile_pic_url_hd") or user.get("profile_pic_url"),
        "is_verified": bool(user.get("is_verified")),
        "is_private": bool(user.get("is_private")),
        "is_business_account": bool(user.get("is_business_account")),
        "business_category": user.get("business_category_name") or user.get("category_name"),
        "business_email": user.get("business_email"),
        "business_phone": user.get("business_phone_number"),
        "followers": (user.get("edge_followed_by") or {}).get("count", 0) or 0,
        "following": (user.get("edge_follow") or {}).get("count", 0) or 0,
        "posts_count": (user.get("edge_owner_to_timeline_media") or {}).get("count", 0) or 0,
        "posts": posts,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source": "web_api",
        "warnings": [],
    }


def _scrape_via_instaloader(username: str, max_posts: int) -> dict:
    import instaloader

    if not HANDLE_RE.match(username):
        raise ValueError("Invalid Instagram username")

    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        max_connection_attempts=3,
        quiet=True,
    )
    try:
        profile = instaloader.Profile.from_username(L.context, username)
    except instaloader.exceptions.ProfileNotExistsException as exc:
        raise LookupError(f"User @{username} not found") from exc
    except instaloader.exceptions.ConnectionException as exc:
        raise PermissionError(f"Instaloader could not reach Instagram: {exc}") from exc

    posts: list[dict] = []
    warnings: list[str] = []
    if not profile.is_private and max_posts > 0:
        try:
            for post in islice(profile.get_posts(), max_posts):
                caption = post.caption or ""
                posts.append(
                    {
                        "shortcode": post.shortcode,
                        "url": f"https://www.instagram.com/p/{post.shortcode}/",
                        "caption": caption,
                        "timestamp": post.date_utc.replace(tzinfo=timezone.utc).isoformat()
                        if post.date_utc
                        else None,
                        "likes": int(post.likes or 0),
                        "comments_count": int(post.comments or 0),
                        "is_video": bool(post.is_video),
                        "hashtags": list(post.caption_hashtags) if caption else [],
                        "mentions": list(post.caption_mentions) if caption else [],
                    }
                )
        except Exception as exc:
            warnings.append(
                f"Instaloader stopped after {len(posts)} of {max_posts} requested post(s): "
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "platform": "instagram",
        "username": profile.username,
        "user_id": str(profile.userid),
        "profile_url": f"https://www.instagram.com/{profile.username}/",
        "full_name": profile.full_name or "",
        "biography": profile.biography or "",
        "external_url": profile.external_url,
        "profile_pic_url": profile.profile_pic_url,
        "is_verified": bool(profile.is_verified),
        "is_private": bool(profile.is_private),
        "is_business_account": bool(profile.is_business_account),
        "business_category": getattr(profile, "business_category_name", None)
        or getattr(profile, "category_name", None),
        "business_email": None,
        "business_phone": None,
        "followers": int(profile.followers or 0),
        "following": int(profile.followees or 0),
        "posts_count": int(profile.mediacount or 0),
        "posts": posts,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source": "instaloader",
        "warnings": warnings,
    }


def scrape_instagram_profile(target: str, max_posts: int = 12) -> dict:
    """
    Scrape a public Instagram profile by username or profile URL.

    Tries the web endpoint first (fast, no login); falls back to Instaloader if
    it's blocked. If more than WEB_POST_PAGE_LIMIT posts are requested and the
    web backend succeeded, tops up the post list via Instaloader (which
    paginates) while keeping the web response's more reliable profile fields.

    Raises ValueError (bad input), LookupError (no such user), PermissionError
    (blocked/private), or RuntimeError (both backends failed).
    """
    username = parse_instagram_target(target)
    if not username:
        raise ValueError("Provide an Instagram username or profile URL.")
    max_posts = max(0, min(int(max_posts), 100))

    errors: list[str] = []
    profile: dict | None = None
    for backend in ("web", "instaloader"):
        try:
            profile = _scrape_via_web(username, max_posts) if backend == "web" else _scrape_via_instaloader(username, max_posts)
            break
        except LookupError:
            raise
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            continue

    if profile is None:
        raise RuntimeError(
            "All scrapers failed for this public profile.\n"
            + "\n".join(errors)
            + "\n\nTips: use a residential/home network, lower max_posts, or retry later. "
            "Private accounts only expose limited public fields."
        )

    if profile["source"] == "web_api" and not profile["is_private"] and max_posts > WEB_POST_PAGE_LIMIT:
        target_count = min(max_posts, profile["posts_count"] or max_posts)
        if len(profile["posts"]) < target_count:
            try:
                topped_up = _scrape_via_instaloader(username, max_posts)
                if len(topped_up["posts"]) > len(profile["posts"]):
                    profile["posts"] = topped_up["posts"]
                    profile["source"] = "web_api+instaloader"
                    profile["warnings"].extend(topped_up["warnings"])
            except Exception as exc:
                profile["warnings"].append(
                    f"Wanted {max_posts} posts but Instagram's profile endpoint returns only "
                    f"{WEB_POST_PAGE_LIMIT}; the Instaloader top-up failed ({exc})."
                )
            if len(profile["posts"]) < target_count:
                profile["warnings"].append(
                    f"Wanted {max_posts} posts, got {len(profile['posts'])}. Instagram's anonymous "
                    "profile endpoint returns 12 per page and rate-limits deeper paging; retry later."
                )

    return profile
