# etc

[Core Index](../../../../README.md#core-index) / [sitstart](../../../index.md#sitstart) / [app](../../index.md#app) / [sit](../index.md#sit) / [sub](./index.md#sub) / etc

> Auto-generated documentation for [sitstart.app.sit.sub.etc](../../../../../python/sitstart/app/sit/sub/etc.py) module.

- [etc](#etc)
  - [archive_trial](#archive_trial)
  - [test_config](#test_config)
  - [update_requirements](#update_requirements)

## archive_trial

[Show source in etc.py:82](../../../../../python/sitstart/app/sit/sub/etc.py#L82)

Archive a trial.

#### Signature

```python
@app.command()
def archive_trial(
    trial_id: Annotated[
        str, Argument(help="The trial ID to archive.", show_default=False)
    ],
    run_group: Annotated[
        Optional[str],
        Option(
            help="The run group. If not specified, runs are searched based on trial ID and project name.",
            show_default=False,
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        Option(
            help=f"Name of the experiment config in {CONFIG_ROOT}.", show_default=False
        ),
    ] = None,
    storage_path: Annotated[
        Optional[str],
        Option(
            help="The storage path. Defaults to the storage path in the config.",
            show_default=False,
        ),
    ] = None,
    project_name: Annotated[
        Optional[str],
        Option(
            help="The project name. Defaults to the project name in the config.",
            show_default=False,
        ),
    ] = None,
) -> None: ...
```



## test_config

[Show source in etc.py:166](../../../../../python/sitstart/app/sit/sub/etc.py#L166)

Test loading, resolving, and instantiating a Hydra experiment config.

#### Signature

```python
@app.command()
def test_config(
    name: Annotated[
        str,
        Argument(
            help=f"Name of the experiment config in {CONFIG_ROOT}.", show_default=False
        ),
    ],
    no_resolve: Annotated[
        bool,
        Option("--no-resolve", help="Don't resolve the config.", show_default=False),
    ] = False,
    tune: Annotated[
        bool,
        Option(
            "--tune",
            help="Test a tuning config. Adds --resolve-trial=sample if instantiating and --resolve-trial=exclude otherwise.",
            show_default=False,
        ),
    ] = False,
    resolve_trial: Annotated[
        TrialResolution,
        Option(help="How to process the config's trial node if resolving the config."),
    ] = TrialResolution.RESOLVE,
    instantiate: Annotated[
        bool, Option("--instantiate", help="Instantiate the config.", show_default=False)
    ] = False,
) -> None: ...
```

#### See also

- [TrialResolution](../../../ml/experiments/util.md#trialresolution)



## update_requirements

[Show source in etc.py:44](../../../../../python/sitstart/app/sit/sub/etc.py#L44)

Update a Python requirements file with Pigar.

#### Signature

```python
@app.command()
def update_requirements(
    project_path: Annotated[
        str, Argument(help="The project path.", show_default=True)
    ] = DEFAULT_PROJECT_PATH,
    requirements_path: Annotated[
        str, Option(help="The requirements file path.", show_default=True)
    ] = DEFAULT_REQUIREMENTS_PATH,
    package_variants: Annotated[
        list[str], Option(help="Package variants to include.", show_default=True)
    ] = DEFAULT_PACKAGE_VARIANTS,
) -> None: ...
```

#### See also

- [DEFAULT_PACKAGE_VARIANTS](#default_package_variants)
- [DEFAULT_PROJECT_PATH](#default_project_path)
- [DEFAULT_REQUIREMENTS_PATH](#default_requirements_path)
