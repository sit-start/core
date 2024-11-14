# migrate_to_obsidian

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [notes](./index.md#notes) / migrate_to_obsidian

> Auto-generated documentation for [sitstart.notes.migrate_to_obsidian](../../../python/sitstart/notes/migrate_to_obsidian.py) module.

- [migrate_to_obsidian](#migrate_to_obsidian)
  - [main](#main)
  - [process](#process)

## main

[Show source in migrate_to_obsidian.py:110](../../../python/sitstart/notes/migrate_to_obsidian.py#L110)

#### Signature

```python
def main(): ...
```



## process

[Show source in migrate_to_obsidian.py:17](../../../python/sitstart/notes/migrate_to_obsidian.py#L17)

Updates a directory of Markdown documents for use with Obsidian.

Specifically, this updates the asset locations and links to be
consistent with the flat directory structure in an Obsidian vault.

The new assset directory should correspond to the 'Files and links'
> 'Default location for attachments' in Obsidian. It's also
recommended to set 'New link format' to 'Relative path to file'
under the same 'Files and Links' section.

#### Signature

```python
def process(
    src_dir: str,
    tgt_dir: str,
    new_asset_dir: str = "assets",
    min_modified_date: datetime = datetime.min,
    include_front_matter: bool = False,
): ...
```
