from pydantic import BaseModel


class InfoResponse(BaseModel):
    """Schema for info endpoint response.

    Attributes:
        name: Project name from pyproject.toml
        version: Project version from pyproject.toml
        git_current_branch: Current git branch from GIT_CURRENT_BRANCH environment variable
        git_head_commit: Current git commit hash from GIT_HEAD_COMMIT environment variable
        lang: Programming language (hardcoded as "python")
        builder: Build tool (hardcoded as "poetry")
        lang_version: Python version from command line
        builder_version: Poetry version from command line
        build_time: Build timestamp in ISO format
    """

    name: str
    version: str
    git_current_branch: str
    git_head_commit: str
    lang: str
    builder: str
    lang_version: str
    builder_version: str
    build_time: str
