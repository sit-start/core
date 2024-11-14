# google_drive

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / google_drive

> Auto-generated documentation for [sitstart.util.google_drive](../../../python/sitstart/util/google_drive.py) module.

- [google_drive](#google_drive)
  - [get_file](#get_file)
  - [get_file_from_path](#get_file_from_path)
  - [get_folder_contents](#get_folder_contents)
  - [get_folder_paths](#get_folder_paths)
  - [get_path](#get_path)
  - [walk](#walk)

## get_file

[Show source in google_drive.py:35](../../../python/sitstart/util/google_drive.py#L35)

#### Signature

```python
def get_file(service: Any, file_id: str, fields: list[str] | None = None) -> File: ...
```

#### See also

- [File](#file)



## get_file_from_path

[Show source in google_drive.py:40](../../../python/sitstart/util/google_drive.py#L40)

Get a File dictionary with the given fields for the given path.

`path` must be an absolute path and start with '/' or 'My Drive'.

#### Signature

```python
def get_file_from_path(
    service: Any, path: str, fields: list[str] | None = None
) -> File: ...
```

#### See also

- [File](#file)



## get_folder_contents

[Show source in google_drive.py:76](../../../python/sitstart/util/google_drive.py#L76)

Returns a 3-tuple of folder, folders and files in the folder.

`folder` can be a File dictionary or the file id.

#### Signature

```python
def get_folder_contents(
    service: Any,
    folder: File | str,
    file_fields: list[str] | None = None,
    folders_only: bool = False,
) -> tuple[File, list[File], list[File]]: ...
```

#### See also

- [File](#file)



## get_folder_paths

[Show source in google_drive.py:140](../../../python/sitstart/util/google_drive.py#L140)

#### Signature

```python
def get_folder_paths(
    service: Any,
    root_folder_id: str | None = None,
    folder_ids_to_ignore: list[str] | None = None,
) -> dict[str, str]: ...
```



## get_path

[Show source in google_drive.py:22](../../../python/sitstart/util/google_drive.py#L22)

#### Signature

```python
def get_path(
    service: Any, file_id: str, file_fields: list[str] | None = None
) -> list[dict[str, str]]: ...
```



## walk

[Show source in google_drive.py:110](../../../python/sitstart/util/google_drive.py#L110)

#### Signature

```python
def walk(
    service: Any,
    folder_id: str,
    fields: list[str] | None = None,
    topdown: bool = True,
    folders_only: bool = False,
    folder_ids_to_ignore: list[str] | None = None,
): ...
```
