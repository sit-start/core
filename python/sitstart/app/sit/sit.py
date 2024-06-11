#!/usr/bin/env python3

import rich  # noqa F401 # type: ignore
import typer

from sitstart.app.sit.sub import ec2, etc, git, ray

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)
app.add_typer(ec2.app, name="ec2")
app.add_typer(git.app, name="git")
app.add_typer(ray.app, name="ray")
app.add_typer(etc.app, name="etc")


if __name__ == "__main__":
    app()
