#!/usr/bin/env python3
"""Fetch perceptual-baseline assets from a remote server via SSH."""

from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import sys
import tarfile
from pathlib import Path

import paramiko


DEFAULT_REMOTE_PATHS = [
    "/root/autodl-tmp/v2_2/dog_reid_web/data/perceptual_baselines/atrw/zerodcepp",
    "/root/autodl-tmp/v2_2/dog_reid_web/data/perceptual_baselines/atrw/retinexnet",
    "/root/autodl-tmp/v2_2/dog_reid_web/data/perceptual_baselines/gzgc_zebra/zerodcepp",
    "/root/autodl-tmp/v2_2/dog_reid_web/data/perceptual_baselines/gzgc_zebra/retinexnet",
    "/root/dog_reid_storage/checkpoints/perceptual_baselines/atrw/zerodcepp",
    "/root/dog_reid_storage/checkpoints/perceptual_baselines/atrw/retinexnet",
    "/root/dog_reid_storage/checkpoints/perceptual_baselines/gzgc_zebra/zerodcepp",
    "/root/dog_reid_storage/checkpoints/perceptual_baselines/gzgc_zebra/retinexnet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=os.environ.get("SEETA_SSH_PASSWORD", ""))
    parser.add_argument(
        "--mode",
        choices=("scan", "fetch"),
        default="scan",
    )
    parser.add_argument(
        "--remote-path",
        action="append",
        dest="remote_paths",
        help="Remote path to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--local-root",
        default="downloads/perceptual_server_assets",
        help="Local extraction root for fetch mode.",
    )
    return parser.parse_args()


def connect_ssh(args: argparse.Namespace) -> paramiko.SSHClient:
    if not args.password:
        raise ValueError("SSH password is required via --password or SEETA_SSH_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        timeout=20,
    )
    return client


def run_remote(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command)
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    return code, out, err


def scan_paths(client: paramiko.SSHClient, paths: list[str]) -> int:
    overall_ok = True
    for path in paths:
        cmd = (
            f"if [ -e {shlex.quote(path)} ]; then "
            f"du -sh {shlex.quote(path)}; "
            f"find {shlex.quote(path)} | wc -l; "
            f"else echo MISSING {shlex.quote(path)}; exit 3; fi"
        )
        code, out, err = run_remote(client, cmd)
        print(f"=== {path}")
        if out.strip():
            print(out.strip())
        if err.strip():
            print(err.strip(), file=sys.stderr)
        if code != 0:
            overall_ok = False
    return 0 if overall_ok else 1


def is_safe_member(member_name: str) -> bool:
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith(".."):
        return False
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return False
    return True


def fetch_paths(client: paramiko.SSHClient, paths: list[str], local_root: Path) -> int:
    local_root.mkdir(parents=True, exist_ok=True)
    rel_paths = [path.lstrip("/") for path in paths]
    remote_cmd = "tar -C / -cf - " + " ".join(shlex.quote(path) for path in rel_paths)
    stdin, stdout, stderr = client.exec_command(remote_cmd)

    extracted = 0
    tar = tarfile.open(fileobj=stdout, mode="r|")
    try:
        for member in tar:
            if not is_safe_member(member.name):
                raise RuntimeError(f"Unsafe tar member: {member.name}")
            tar.extract(member, path=local_root)
            extracted += 1
            if extracted % 200 == 0:
                print(f"Extracted {extracted} members...")
    finally:
        tar.close()

    exit_code = stdout.channel.recv_exit_status()
    err = stderr.read().decode("utf-8", "ignore").strip()
    if err:
        print(err, file=sys.stderr)
    print(f"Extracted total members: {extracted}")
    return exit_code


def main() -> int:
    args = parse_args()
    paths = args.remote_paths or DEFAULT_REMOTE_PATHS
    client = connect_ssh(args)
    try:
        if args.mode == "scan":
            return scan_paths(client, paths)
        return fetch_paths(client, paths, Path(args.local_root))
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
