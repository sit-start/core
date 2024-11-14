# ray

[Core Index](../../../../README.md#core-index) / [sitstart](../../../index.md#sitstart) / [app](../../index.md#app) / [sit](../index.md#sit) / [sub](./index.md#sub) / ray

> Auto-generated documentation for [sitstart.app.sit.sub.ray](../../../../../python/sitstart/app/sit/sub/ray.py) module.

#### Attributes

- `ConfigOpt` - Arguments and options: Annotated[str, Option(help=f'The Ray cluster config path, or filename or stem in {CLUSTER_CONFIG_ROOT}.')]


- [ray](#ray)
  - [down](#down)
  - [list_jobs](#list_jobs)
  - [monitor](#monitor)
  - [stop_job](#stop_job)
  - [submit](#submit)
  - [up](#up)

## down

[Show source in ray.py:368](../../../../../python/sitstart/app/sit/sub/ray.py#L368)

Tear down a Ray cluster.

#### Signature

```python
@app.command()
def down(
    config: ConfigOpt = DEFAULT_CONFIG,
    profile: ProfileOpt = None,
    cluster_name: ClusterNameOpt = None,
    workers_only: WorkersOnlyOpt = False,
    keep_min_workers: KeepMinWorkersOpt = False,
    prompt: PromptOpt = False,
    verbose: VerboseOpt = False,
    kill: KillOpt = False,
    show_output: ShowOutputOpt = False,
) -> None: ...
```

#### See also

- [ClusterNameOpt](#clusternameopt)
- [ConfigOpt](#configopt)
- [DEFAULT_CONFIG](#default_config)
- [KeepMinWorkersOpt](#keepminworkersopt)
- [KillOpt](#killopt)
- [ProfileOpt](#profileopt)
- [PromptOpt](#promptopt)
- [ShowOutputOpt](#showoutputopt)
- [VerboseOpt](#verboseopt)
- [WorkersOnlyOpt](#workersonlyopt)



## list_jobs

[Show source in ray.py:252](../../../../../python/sitstart/app/sit/sub/ray.py#L252)

List all jobs on the active Ray cluster.

#### Signature

```python
@app.command()
def list_jobs(dashboard_port: DashboardPortOpt = DASHBOARD_PORT) -> None: ...
```

#### See also

- [DASHBOARD_PORT](../../../ray/cluster.md#dashboard_port)
- [DashboardPortOpt](#dashboardportopt)



## monitor

[Show source in ray.py:408](../../../../../python/sitstart/app/sit/sub/ray.py#L408)

Monitor autoscaling on a Ray cluster.

#### Signature

```python
@app.command()
def monitor(config: ConfigOpt = DEFAULT_CONFIG) -> None: ...
```

#### See also

- [ConfigOpt](#configopt)
- [DEFAULT_CONFIG](#default_config)



## stop_job

[Show source in ray.py:241](../../../../../python/sitstart/app/sit/sub/ray.py#L241)

Stops a job on the active Ray cluster.

#### Signature

```python
@app.command()
def stop_job(
    submission_id: SubmissionIdArg,
    delete: DeleteOpt = False,
    dashboard_port: DashboardPortOpt = DASHBOARD_PORT,
) -> None: ...
```

#### See also

- [DASHBOARD_PORT](../../../ray/cluster.md#dashboard_port)
- [DashboardPortOpt](#dashboardportopt)
- [DeleteOpt](#deleteopt)
- [SubmissionIdArg](#submissionidarg)



## submit

[Show source in ray.py:260](../../../../../python/sitstart/app/sit/sub/ray.py#L260)

Run a job on a Ray cluster.

#### Signature

```python
@app.command()
def submit(
    script: ScriptArg,
    profile: ProfileOpt = None,
    config: ScriptConfigOpt = None,
    dashboard_port: DashboardPortOpt = DASHBOARD_PORT,
    clone_venv: CloneVenvOpt = False,
    description: DescriptionOpt = None,
) -> str: ...
```

#### See also

- [CloneVenvOpt](#clonevenvopt)
- [DASHBOARD_PORT](../../../ray/cluster.md#dashboard_port)
- [DashboardPortOpt](#dashboardportopt)
- [DescriptionOpt](#descriptionopt)
- [ProfileOpt](#profileopt)
- [ScriptArg](#scriptarg)
- [ScriptConfigOpt](#scriptconfigopt)



## up

[Show source in ray.py:289](../../../../../python/sitstart/app/sit/sub/ray.py#L289)

Create or update a Ray cluster.

#### Signature

```python
@app.command()
def up(
    config: ConfigOpt = DEFAULT_CONFIG,
    profile: ProfileOpt = None,
    min_workers: MinWorkersOpt = None,
    max_workers: MaxWorkersOpt = None,
    no_restart: NoRestartOpt = False,
    restart_only: RestartOnlyOpt = False,
    cluster_name: ClusterNameOpt = None,
    prompt: PromptOpt = False,
    verbose: VerboseOpt = False,
    open_vscode: OpenVscodeOpt = False,
    show_output: ShowOutputOpt = False,
    no_config_cache: NoConfigCacheOpt = False,
    sync_dotfiles: SyncDotfilesOpt = False,
    no_port_forwarding: NoPortForwardingOpt = False,
    forward_port: ForwardPortOpt = None,
) -> None: ...
```

#### See also

- [ClusterNameOpt](#clusternameopt)
- [ConfigOpt](#configopt)
- [DEFAULT_CONFIG](#default_config)
- [ForwardPortOpt](#forwardportopt)
- [MaxWorkersOpt](#maxworkersopt)
- [MinWorkersOpt](#minworkersopt)
- [NoConfigCacheOpt](#noconfigcacheopt)
- [NoPortForwardingOpt](#noportforwardingopt)
- [NoRestartOpt](#norestartopt)
- [OpenVscodeOpt](#openvscodeopt)
- [ProfileOpt](#profileopt)
- [PromptOpt](#promptopt)
- [RestartOnlyOpt](#restartonlyopt)
- [ShowOutputOpt](#showoutputopt)
- [SyncDotfilesOpt](#syncdotfilesopt)
- [VerboseOpt](#verboseopt)
