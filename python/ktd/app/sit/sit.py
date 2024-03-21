#!/usr/bin/env python3

import typer
from ktd.app.sit.sub import ec2, git, ray

app = typer.Typer()
app.add_typer(ec2.app, name="ec2")
app.add_typer(git.app, name="git")
app.add_typer(ray.app, name="ray")


if __name__ == "__main__":
    app()