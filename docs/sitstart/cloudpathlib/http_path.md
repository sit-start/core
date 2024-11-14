# HttpPath

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [cloudpathlib](./index.md#cloudpathlib) / HttpPath

> Auto-generated documentation for [sitstart.cloudpathlib.http_path](../../../python/sitstart/cloudpathlib/http_path.py) module.

- [HttpPath](#httppath)
  - [HttpPath](#httppath-1)
    - [HttpPath().drive](#httppath()drive)
    - [HttpPath().etag](#httppath()etag)
    - [HttpPath().is_dir](#httppath()is_dir)
    - [HttpPath().is_file](#httppath()is_file)
    - [HttpPath().mkdir](#httppath()mkdir)
    - [HttpPath().stat](#httppath()stat)
    - [HttpPath().touch](#httppath()touch)
  - [HttpsPath](#httpspath)

## HttpPath

[Show source in http_path.py:11](../../../python/sitstart/cloudpathlib/http_path.py#L11)

#### Signature

```python
class HttpPath(CloudPath):
    def __init__(self, *args, **kwargs): ...
```

### HttpPath().drive

[Show source in http_path.py:18](../../../python/sitstart/cloudpathlib/http_path.py#L18)

#### Signature

```python
@property
def drive(self) -> str: ...
```

### HttpPath().etag

[Show source in http_path.py:57](../../../python/sitstart/cloudpathlib/http_path.py#L57)

#### Signature

```python
@property
def etag(self): ...
```

### HttpPath().is_dir

[Show source in http_path.py:22](../../../python/sitstart/cloudpathlib/http_path.py#L22)

#### Signature

```python
def is_dir(self) -> bool: ...
```

### HttpPath().is_file

[Show source in http_path.py:25](../../../python/sitstart/cloudpathlib/http_path.py#L25)

#### Signature

```python
def is_file(self) -> bool: ...
```

### HttpPath().mkdir

[Show source in http_path.py:28](../../../python/sitstart/cloudpathlib/http_path.py#L28)

#### Signature

```python
def mkdir(self, parents=False, exist_ok=False): ...
```

### HttpPath().stat

[Show source in http_path.py:34](../../../python/sitstart/cloudpathlib/http_path.py#L34)

#### Signature

```python
def stat(self) -> os.stat_result: ...
```

### HttpPath().touch

[Show source in http_path.py:31](../../../python/sitstart/cloudpathlib/http_path.py#L31)

#### Signature

```python
def touch(self, exist_ok: bool = True): ...
```



## HttpsPath

[Show source in http_path.py:63](../../../python/sitstart/cloudpathlib/http_path.py#L63)

#### Signature

```python
class HttpsPath(HttpPath): ...
```

#### See also

- [HttpPath](#httppath)
