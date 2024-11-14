# HttpClient

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [cloudpathlib](./index.md#cloudpathlib) / HttpClient

> Auto-generated documentation for [sitstart.cloudpathlib.http_client](../../../python/sitstart/cloudpathlib/http_client.py) module.

- [HttpClient](#httpclient)
  - [HttpClient](#httpclient-1)

## HttpClient

[Show source in http_client.py:23](../../../python/sitstart/cloudpathlib/http_client.py#L23)

#### Signature

```python
class HttpClient(Client):
    def __init__(
        self,
        file_cache_mode: str | FileCacheMode | None = None,
        local_cache_dir: str | os.PathLike | None = None,
        content_type_method: Callable | None = mimetypes.guess_type,
    ): ...
```
