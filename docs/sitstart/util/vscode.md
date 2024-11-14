# vscode

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / vscode

> Auto-generated documentation for [sitstart.util.vscode](../../../python/sitstart/util/vscode.py) module.

- [vscode](#vscode)
  - [VSCodeTarget](#vscodetarget)
  - [open_vscode_over_ssh](#open_vscode_over_ssh)

## VSCodeTarget

[Show source in vscode.py:12](../../../python/sitstart/util/vscode.py#L12)

#### Signature

```python
class VSCodeTarget(Enum): ...
```



## open_vscode_over_ssh

[Show source in vscode.py:17](../../../python/sitstart/util/vscode.py#L17)

#### Signature

```python
def open_vscode_over_ssh(
    hostname: str,
    target: str | VSCodeTarget = DEFAULT_TARGET,
    path: str = DEFAULT_FOLDER,
) -> None: ...
```

#### See also

- [DEFAULT_FOLDER](#default_folder)
- [DEFAULT_TARGET](#default_target)
