# util

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [cloudpathlib](./index.md#cloudpathlib) / util

> Auto-generated documentation for [sitstart.cloudpathlib.util](../../../python/sitstart/cloudpathlib/util.py) module.

- [util](#util)
  - [get_local_path](#get_local_path)

## get_local_path

[Show source in util.py:6](../../../python/sitstart/cloudpathlib/util.py#L6)

Returns the local path for the given CloudPath.

CloudPath.fspath caches files; this caches directories recursively
as well.

#### Signature

```python
def get_local_path(path: CloudPath | Path) -> Path: ...
```
