# ec2

[Core Index](../../../../README.md#core-index) / [sitstart](../../../index.md#sitstart) / [app](../../index.md#app) / [sit](../index.md#sit) / [sub](./index.md#sub) / ec2

> Auto-generated documentation for [sitstart.app.sit.sub.ec2](../../../../../python/sitstart/app/sit/sub/ec2.py) module.

#### Attributes

- `InstanceNameArg` - Arguments and options: Annotated[str, Argument(help='The instance name or name pattern.', show_default=False)]


- [ec2](#ec2)
  - [create](#create)
  - [kill](#kill)
  - [list](#list)
  - [open](#open)
  - [refresh](#refresh)
  - [start](#start)
  - [stop](#stop)

## create

[Show source in ec2.py:92](../../../../../python/sitstart/app/sit/sub/ec2.py#L92)

Create a devserver with the given name and arguments.

#### Signature

```python
@app.command()
def create(
    instance_name: InstanceNameArg,
    profile: ProfileOpt = None,
    instance_type: InstanceTypeOpt = DEFAULT_INSTANCE_TYPE,
    open_vscode: OpenVscodeOpt = False,
    no_dotfiles: NoDotfilesOpt = False,
    dotfiles_repo: DotfilesRepoOpt = DEFAULT_DOTFILES_REPO,
) -> None: ...
```

#### See also

- [DEFAULT_DOTFILES_REPO](#default_dotfiles_repo)
- [DEFAULT_INSTANCE_TYPE](#default_instance_type)
- [DotfilesRepoOpt](#dotfilesrepoopt)
- [InstanceNameArg](#instancenamearg)
- [InstanceTypeOpt](#instancetypeopt)
- [NoDotfilesOpt](#nodotfilesopt)
- [OpenVscodeOpt](#openvscodeopt)
- [ProfileOpt](#profileopt)



## kill

[Show source in ec2.py:198](../../../../../python/sitstart/app/sit/sub/ec2.py#L198)

Terminate instances with the given name.

#### Signature

```python
@app.command()
def kill(instance_name: InstanceNameArg, profile: ProfileOpt = None) -> None: ...
```

#### See also

- [InstanceNameArg](#instancenamearg)
- [ProfileOpt](#profileopt)



## list

[Show source in ec2.py:206](../../../../../python/sitstart/app/sit/sub/ec2.py#L206)

List instances.

#### Signature

```python
@app.command()
def list(
    profile: ProfileOpt = None,
    show_killed: ShowKilledOpt = False,
    compact: CompactOpt = False,
) -> None: ...
```

#### See also

- [CompactOpt](#compactopt)
- [ProfileOpt](#profileopt)
- [ShowKilledOpt](#showkilledopt)



## open

[Show source in ec2.py:269](../../../../../python/sitstart/app/sit/sub/ec2.py#L269)

Open VS Code on the instance with the given name.

#### Signature

```python
@app.command()
def open(
    instance_name: InstanceNameArg,
    target: TargetOpt = DEFAULT_TARGET,
    path: PathOpt = DEFAULT_FOLDER,
) -> None: ...
```

#### See also

- [DEFAULT_FOLDER](../../../util/vscode.md#default_folder)
- [DEFAULT_TARGET](../../../util/vscode.md#default_target)
- [InstanceNameArg](#instancenamearg)
- [PathOpt](#pathopt)
- [TargetOpt](#targetopt)



## refresh

[Show source in ec2.py:262](../../../../../python/sitstart/app/sit/sub/ec2.py#L262)

Refresh hostnames in the SSH config for all running named instances.

#### Signature

```python
@app.command()
def refresh(profile: ProfileOpt = None) -> None: ...
```

#### See also

- [ProfileOpt](#profileopt)



## start

[Show source in ec2.py:147](../../../../../python/sitstart/app/sit/sub/ec2.py#L147)

Start instances with the given name.

#### Signature

```python
@app.command()
def start(
    instance_name: InstanceNameArg,
    profile: ProfileOpt = None,
    open_vscode: OpenVscodeOpt = False,
) -> None: ...
```

#### See also

- [InstanceNameArg](#instancenamearg)
- [OpenVscodeOpt](#openvscodeopt)
- [ProfileOpt](#profileopt)



## stop

[Show source in ec2.py:178](../../../../../python/sitstart/app/sit/sub/ec2.py#L178)

Stop instances with the given name.

#### Signature

```python
@app.command()
def stop(instance_name: InstanceNameArg, profile: ProfileOpt = None) -> None: ...
```

#### See also

- [InstanceNameArg](#instancenamearg)
- [ProfileOpt](#profileopt)
