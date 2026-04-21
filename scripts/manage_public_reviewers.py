#!/usr/bin/env python3
"""Manage BasicAuth reviewers for the public concam bundle.

Commands:

    add <user> [--password P]   Add or re-set one user. Random password if
                                --password is omitted. Existing users are
                                preserved.
    remove <user>               Remove one user.
    list                        Print the current users.
    reset <user> [<user> ...]   Replace the file entirely with the given
                                users (random passwords).  Everyone not
                                listed stops being able to log in.

State lives in two files under ``~/public_html/concam/``:

    .htpasswd       Apache-readable hashed passwords. This is the only file
                    that governs who can log in.
    credentials.txt Human-readable record of the most-recent passwords the
                    script issued. Mode 0600. Editing this file by hand
                    DOES NOT change auth — the hash is in .htpasswd.

If you want a user with a specific password, run:
    python3 scripts/manage_public_reviewers.py add alice --password spring2026
"""

from __future__ import annotations

import argparse
import os
import secrets
import string
import subprocess
from pathlib import Path

PUBLIC_ROOT = Path.home() / "public_html" / "concam"
HTPASSWD_PATH = PUBLIC_ROOT / ".htpasswd"
CREDENTIALS_PATH = PUBLIC_ROOT / "credentials.txt"
PASSWORD_ALPHABET = string.ascii_letters + string.digits


def _random_password(length: int = 12) -> str:
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(length))


def _read_credentials() -> dict[str, str]:
    """Parse credentials.txt into {user: password}. Missing file → empty."""
    if not CREDENTIALS_PATH.exists():
        return {}
    out: dict[str, str] = {}
    for line in CREDENTIALS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def _write_credentials(creds: dict[str, str]) -> None:
    body = (
        "# ConCam reviewer credentials — keep private.\n"
        "# Site: https://hex.mit.edu/~prash/concam/\n"
        "# Auth is governed by .htpasswd; editing this file does nothing by itself.\n\n"
        + "\n".join(f"{u}\t{p}" for u, p in sorted(creds.items()))
        + "\n"
    )
    CREDENTIALS_PATH.write_text(body)
    os.chmod(CREDENTIALS_PATH, 0o600)


def _htpasswd_users() -> set[str]:
    if not HTPASSWD_PATH.exists():
        return set()
    users: set[str] = set()
    for line in HTPASSWD_PATH.read_text().splitlines():
        if ":" in line:
            users.add(line.split(":", 1)[0])
    return users


def _htpasswd_set(user: str, password: str) -> None:
    # Create the file if missing (-c), otherwise append/update in place.
    flag = "-bc" if not HTPASSWD_PATH.exists() else "-b"
    subprocess.run(
        ["htpasswd", flag, str(HTPASSWD_PATH), user, password],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.chmod(HTPASSWD_PATH, 0o644)


def _htpasswd_remove(user: str) -> None:
    subprocess.run(
        ["htpasswd", "-D", str(HTPASSWD_PATH), user],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def cmd_add(args: argparse.Namespace) -> None:
    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    password = args.password or _random_password()
    _htpasswd_set(args.user, password)
    creds = _read_credentials()
    creds[args.user] = password
    _write_credentials(creds)
    print(f"{args.user}\t{password}")


def cmd_remove(args: argparse.Namespace) -> None:
    if HTPASSWD_PATH.exists() and args.user in _htpasswd_users():
        _htpasswd_remove(args.user)
    creds = _read_credentials()
    creds.pop(args.user, None)
    _write_credentials(creds)
    print(f"removed {args.user}")


def cmd_list(_: argparse.Namespace) -> None:
    htpasswd_live = _htpasswd_users()
    creds = _read_credentials()
    all_users = htpasswd_live | creds.keys()
    for user in sorted(all_users):
        password = creds.get(user, "(password not recorded)")
        note = "" if user in htpasswd_live else " [ORPHAN: not in .htpasswd]"
        print(f"{user}\t{password}{note}")


def cmd_reset(args: argparse.Namespace) -> None:
    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    if HTPASSWD_PATH.exists():
        HTPASSWD_PATH.unlink()
    creds: dict[str, str] = {}
    for user in args.users:
        password = _random_password()
        _htpasswd_set(user, password)
        creds[user] = password
    _write_credentials(creds)
    for user, password in sorted(creds.items()):
        print(f"{user}\t{password}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add or re-set a user (others untouched).")
    p_add.add_argument("user")
    p_add.add_argument("--password", default=None,
                       help="Use this password (default: random).")
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("remove", help="Remove a user.")
    p_rm.add_argument("user")
    p_rm.set_defaults(func=cmd_remove)

    p_list = sub.add_parser("list", help="Show current users and passwords.")
    p_list.set_defaults(func=cmd_list)

    p_reset = sub.add_parser("reset", help="Replace the user set entirely.")
    p_reset.add_argument("users", nargs="+")
    p_reset.set_defaults(func=cmd_reset)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
