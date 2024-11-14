# run

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / run

> Auto-generated documentation for [sitstart.util.run](../../../python/sitstart/util/run.py) module.

- [run](#run)
  - [Output](#output)
  - [run](#run-1)

## Output

[Show source in run.py:14](../../../python/sitstart/util/run.py#L14)

#### Signature

```python
class Output(Enum): ...
```



## run

[Show source in run.py:20](../../../python/sitstart/util/run.py#L20)

#### Signature

```python
def run(
    cmd: list[str], output: str | Output = Output.STD, check: bool = True, **kwargs: Any
) -> subprocess.CompletedProcess[bytes]: ...
```
