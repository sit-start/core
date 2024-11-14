# util

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [aws](./index.md#aws) / util

> Auto-generated documentation for [sitstart.aws.util](../../../python/sitstart/aws/util.py) module.

- [util](#util)
  - [get_aws_session](#get_aws_session)
  - [is_logged_in](#is_logged_in)
  - [sso_login](#sso_login)
  - [update_aws_env](#update_aws_env)

## get_aws_session

[Show source in util.py:53](../../../python/sitstart/aws/util.py#L53)

#### Signature

```python
def get_aws_session(profile: str | None = None) -> boto3.Session: ...
```



## is_logged_in

[Show source in util.py:17](../../../python/sitstart/aws/util.py#L17)

Check if the given session is logged in

#### Signature

```python
def is_logged_in(session: Optional[boto3.Session] = None) -> bool: ...
```



## sso_login

[Show source in util.py:37](../../../python/sitstart/aws/util.py#L37)

Login to AWS via SSO if not already logged in

#### Signature

```python
def sso_login(profile_name=None) -> None: ...
```



## update_aws_env

[Show source in util.py:58](../../../python/sitstart/aws/util.py#L58)

Update AWS environment variables.

Useful for components like pyarrow that rely on environment
variables for AWS credentials.

#### Signature

```python
def update_aws_env(
    session: boto3.Session | None = None, profile: str | None = None
) -> boto3.Session: ...
```
