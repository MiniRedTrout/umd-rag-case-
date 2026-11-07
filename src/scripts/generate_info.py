import argparse
import json
import os
import subprocess
from datetime import datetime

from src.schemas.info import InfoResponse

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_pyproject_info(pyproject_path: str):
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
        return {"name": data["tool"]["poetry"]["name"], "version": data["tool"]["poetry"]["version"]}


def get_git_commit():
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return commit_hash
    except Exception:
        return os.getenv("GIT_HEAD_COMMIT", "unknown")


def get_git_branch():
    try:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        return branch_name
    except Exception:
        return os.getenv("GIT_CURRENT_BRANCH", "unknown")


def get_lang_version():
    return subprocess.check_output(["python3", "--version"]).decode().strip().split()[1]


def get_poetry_version():
    try:
        output = subprocess.check_output(["poetry", "--version"]).decode().strip()
        version = output.split()[-1].strip("()")
        return version
    except Exception:
        return "unknown"


def get_current_time():
    return datetime.utcnow().isoformat() + "Z"


def generate_info(pyproject_path: str) -> InfoResponse:
    py_info = load_pyproject_info(pyproject_path)
    return InfoResponse(
        name=py_info["name"],
        version=py_info["version"],
        git_current_branch=get_git_branch(),
        git_head_commit=get_git_commit(),
        lang="python",
        builder="poetry",
        lang_version=get_lang_version(),
        builder_version=get_poetry_version(),
        build_time=get_current_time(),
    )


def save_info(info: InfoResponse, filename: str):
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(info.model_dump(), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", "-pp", default=os.getenv("PYPROJECT_PATH", "pyproject.toml"))
    parser.add_argument("--output", "-o", default=os.getenv("SERVICE_INFO_PATH", "info.json"))
    args = parser.parse_args()

    info = generate_info(args.pyproject)
    save_info(info, args.output)
    print(f"✔ {args.output} успешно создан")
