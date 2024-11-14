# util

[Core Index](../../../README.md#core-index) / [sitstart](../../index.md#sitstart) / [aws](../index.md#aws) / [ec2](./index.md#ec2) / util

> Auto-generated documentation for [sitstart.aws.ec2.util](../../../../python/sitstart/aws/ec2/util.py) module.

- [util](#util)
  - [get_instance_name](#get_instance_name)
  - [get_instances](#get_instances)
  - [get_unique_instance_name](#get_unique_instance_name)
  - [kill_instances](#kill_instances)
  - [kill_instances_with_name](#kill_instances_with_name)
  - [update_ssh_config_for_instance](#update_ssh_config_for_instance)
  - [update_ssh_config_for_instances_with_name](#update_ssh_config_for_instances_with_name)
  - [wait_for_cloud_init](#wait_for_cloud_init)
  - [wait_for_instance_with_id](#wait_for_instance_with_id)
  - [wait_for_stack_with_name](#wait_for_stack_with_name)

## get_instance_name

[Show source in util.py:24](../../../../python/sitstart/aws/ec2/util.py#L24)

#### Signature

```python
def get_instance_name(instance: ServiceResource) -> str | None: ...
```



## get_instances

[Show source in util.py:46](../../../../python/sitstart/aws/ec2/util.py#L46)

Returns a list of EC2 instances with the given name and state(s)

Supports the same wildcards as the AWS CLI, e.g. "web*" or "web-1?".

#### Signature

```python
def get_instances(
    name: str | None = None,
    ids: list[str] = [],
    states: list[str] | None = None,
    session: boto3.Session | None = None,
) -> list[ServiceResource]: ...
```



## get_unique_instance_name

[Show source in util.py:31](../../../../python/sitstart/aws/ec2/util.py#L31)

Returns a name for the given EC2 instance unique among running instances.

#### Signature

```python
def get_unique_instance_name(instance: ServiceResource) -> str: ...
```



## kill_instances

[Show source in util.py:119](../../../../python/sitstart/aws/ec2/util.py#L119)

#### Signature

```python
def kill_instances(
    session: boto3.Session,
    instances: list[ServiceResource],
    update_ssh_config: bool = True,
    kill_stacks: bool = False,
) -> None: ...
```



## kill_instances_with_name

[Show source in util.py:105](../../../../python/sitstart/aws/ec2/util.py#L105)

#### Signature

```python
def kill_instances_with_name(
    session: boto3.Session,
    instance_name: str,
    states: list[str] | None = None,
    update_ssh_config: bool = True,
    kill_stacks: bool = False,
) -> None: ...
```



## update_ssh_config_for_instance

[Show source in util.py:146](../../../../python/sitstart/aws/ec2/util.py#L146)

#### Signature

```python
def update_ssh_config_for_instance(
    instance: ServiceResource, ssh_config_path: str | os.PathLike = "$HOME/.ssh/config"
) -> None: ...
```



## update_ssh_config_for_instances_with_name

[Show source in util.py:159](../../../../python/sitstart/aws/ec2/util.py#L159)

Update the SSH config for all running instances with the given name

#### Signature

```python
def update_ssh_config_for_instances_with_name(
    session: boto3.Session, instance_name: str = "?*"
) -> None: ...
```



## wait_for_cloud_init

[Show source in util.py:81](../../../../python/sitstart/aws/ec2/util.py#L81)

#### Signature

```python
def wait_for_cloud_init(instance: ServiceResource): ...
```



## wait_for_instance_with_id

[Show source in util.py:66](../../../../python/sitstart/aws/ec2/util.py#L66)

#### Signature

```python
def wait_for_instance_with_id(
    instance_id: str,
    session: boto3.Session | None = None,
    delay_sec: int = 15,
    max_attempts: int = 20,
    wait_on: str = "instance_status_ok",
) -> None: ...
```



## wait_for_stack_with_name

[Show source in util.py:91](../../../../python/sitstart/aws/ec2/util.py#L91)

#### Signature

```python
def wait_for_stack_with_name(
    stack_name: str,
    session: boto3.Session | None = None,
    delay_sec: int = 15,
    max_attempts: int = 20,
) -> None: ...
```
