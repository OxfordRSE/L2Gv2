import glob
import nox

PYTHON_VERSIONS = "3.10"
REPO = "https://github.com/OxfordRSE/L2Gv2/blob/main/"


@nox.session
def lint(session):
    "Lint code using pylint and ruff"
    session.install("pylint", "ruff")
    session.run("pylint", "l2gv2/**/*.py")
    session.run("pylint", "tests/**/*.py")
    session.run("ruff", "check")
    session.run("ruff", "format", "--check")


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    "Run tests"
    session.run("pip", "install", ".[dev]")
    session.run("pytest", "-n", "auto", "--cov")


@nox.session(python=PYTHON_VERSIONS)
def notebooks(session):
    "Run Jupyter notebooks"
    session.run("pip", "install", ".[dev]")
    session.install("jupyter")
    exit_codes = []
    for n in glob.glob("examples/*.ipynb"):
        try:
            session.run("jupyter", "execute", n)
            exit_codes.append(0)
        except nox.command.CommandFailed as e:
            exit_codes.append(e.reason)
    if any(e != 0 for e in exit_codes):
        session.error("One or more Jupyter notebooks failed to execute")


@nox.session(name="unused-code", default=False)
def unused_code(session):
    """Shows unused code warnings using vulture.

    This test always passes, and is used to create GitHub issues
    to report on unused code.
    """
    session.install("vulture")
    out = session.run("vulture", "l2gv2", "tests", silent=True, success_codes=[0, 3])
    for line in out.splitlines():
        file, lineno, msg = line.split(":")
        if "variable" in msg:
            continue
        print(f"- [ ] [{file}:{lineno}]({REPO}{file}#L{lineno}) {msg}")
