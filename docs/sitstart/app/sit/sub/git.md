# git

[Core Index](../../../../README.md#core-index) / [sitstart](../../../index.md#sitstart) / [app](../../index.md#app) / [sit](../index.md#sit) / [sub](./index.md#sub) / git

> Auto-generated documentation for [sitstart.app.sit.sub.git](../../../../../python/sitstart/app/sit/sub/git.py) module.

- [git](#git)
  - [private_fork](#private_fork)

## private_fork

[Show source in git.py:13](../../../../../python/sitstart/app/sit/sub/git.py#L13)

Create a private fork of a repository.

#### Signature

```python
@app.command()
def private_fork(
    repository: Annotated[
        str, Argument(help="The URL of the repository to fork.", show_default=False)
    ],
    fork_name: Annotated[
        Optional[str], Option(help="Rename the forked repository.", show_default=False)
    ] = None,
    clone: Annotated[bool, Option(help="Clone the forked repository.")] = False,
    org: Annotated[
        Optional[str],
        Option(help="Create the fork in an organization.", show_default=False),
    ] = None,
): ...
```
