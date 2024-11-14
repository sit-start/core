# HAM10k

[Core Index](../../../../README.md#core-index) / [sitstart](../../../index.md#sitstart) / [ml](../../index.md#ml) / [data](../index.md#data) / [datasets](./index.md#datasets) / HAM10k

> Auto-generated documentation for [sitstart.ml.data.datasets.ham10k](../../../../../python/sitstart/ml/data/datasets/ham10k.py) module.

- [HAM10k](#ham10k)
  - [HAM10k](#ham10k-1)
    - [HAM10k().dataset_root](#ham10k()dataset_root)
    - [HAM10k().download](#ham10k()download)

## HAM10k

[Show source in ham10k.py:52](../../../../../python/sitstart/ml/data/datasets/ham10k.py#L52)

HAM10000 Dataset

Dermatoscopic images of common pigmented skin lesions from:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

#### Signature

```python
class HAM10k(VisionDataset):
    def __init__(
        self,
        root: str = DEFAULT_DATASET_ROOT,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
        aws_profile: str | None = None,
    ) -> None: ...
```

#### See also

- [DEFAULT_DATASET_ROOT](../index.md#default_dataset_root)

### HAM10k().dataset_root

[Show source in ham10k.py:149](../../../../../python/sitstart/ml/data/datasets/ham10k.py#L149)

#### Signature

```python
@property
def dataset_root(self) -> str: ...
```

### HAM10k().download

[Show source in ham10k.py:153](../../../../../python/sitstart/ml/data/datasets/ham10k.py#L153)

#### Signature

```python
def download(self) -> None: ...
```
