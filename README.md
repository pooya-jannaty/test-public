<style>
details { 
    summary {
        font-size: 18px; 
        color: blue
    }
    margin-left: 10px;
}
details {
    summary.h3 {
        font-size: 16px; 
        color: blue;
    }
    margin-left: 20px;
}
</style>

# Transfer1 Code Walkthrough

## Config System

<details><summary>Concept: Lazy calls</summary>

Using OmegaConf’s DictConfig for lazy calling in PyTorch provides a clean, modular, and dynamic way to define and instantiate models, optimizers and datasets.

* You define **what** to instantiate in your config.
* You **don’t instantiate** it immediately.
* Later, at runtime, you call a function like hydra.utils.instantiate(cfg) to create the actual object.

This gives you separation of **declaration** (in config) vs. *execution* (in code)

`LazyDict` is actually an alias for `DictConfig` from the omegaconf library. Lazy calls, especially as used in conjunction with `DictConfig` and Hydra/OmegaConf, provide a powerful way to delay instantiation of objects (like models, optimizers, datasets) until the actual runtime, rather than when the config is parsed. Here's an example:

YAML config:
```yaml
model:
  _target_: mymodule.models.ResNet
  depth: 50
```

Python code:
```python
from hydra.utils import instantiate
model = instantiate(cfg.model)
```
No need to import or instantiate the model inside the config—just describe it.
</details>

<details><summary>Primer: Hydra</summary>

We use [Hydra](https://hydra.cc/docs/intro/) for advanced configuration composition and overriding. Here's a primer:

- Hydra lets you categorize your configs by `group`s. 
  - Each option in a group is identified by its `name`. 
  - Much like radio buttons, you can only select/activate one `name` in each `group`. 
- `node` specifies the config class type for each `name`. 
- Once you select one `name` in each `group`, Hydra will copy over the contents of the selected configs and put them under the path indicated by `package`. 

Here's an example to concretize the concepts of `group`, `name`, `node` and `package`. Imagine you have these dataclasses:

```python
@attr.define
class UnetConfig:
    channels: int = 64
    num_blocks: int = 4

@attr.define
class ViTConfig:
    hidden_dim: int = 512
    depth: int = 6
```

This is how you register them into Hydra’s config system:

```python
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

# Register Unet under the 'net' group.
cs.store(
    group="net",
    name="unet",
    node=UnetConfig,
    package="model"
)

# Register ViT under the same group.
cs.store(
    group="net",
    name="vit",
    node=ViTConfig,
    package="model.net" # Paths can be nested.
)
```

Now if you select `unet` as the default config, like so
```yaml
defaults:
    net: unet
```

Hydra will put the contents of `UnetConfig` under the path `model` and compose the final configuration as follows.

```yaml
model:
    channels: 64
    num_blocks: 4
```

Conversely, if you select `vit`, the final config that Hydra will compose will look like:
```yaml
model:
    net:
        hidden_dim: 512
        depth: 6
```

<details><summary class=h3>What is the `_self_` field anyway?</summary>

`_self_` will insert the current file’s fields (outside defaults) at its position, and since later fields override earlier ones, we can use it to control the override precedence. Consider the example

```yaml
defaults:
    net: unet
    _self_
    model.num_blocks: 8
```

This will result in the config

```yaml
model:
    channels: 64
    num_blocks: 8
```

whereas 

```yaml
defaults:
    _self_
    net: unet
    model.num_blocks: 8
```

would have not changed the default, as it would be equivalent to

```yaml
model:
    num_blocks: 8
    channels: 64
    num_blocks: 4
```

but because later variables override earlier ones, `num_blocks: 8` would not have taken effect.

</details>

</details>

<details><summary>Concept: Hydra + attrs</summary>

Hydra commonly pais with YAML config files. In this codebase however, instead of pairing Hydra with YAML, we opted for pairing it with a Python-based structured config class using `attrs`. 

So for example, instead of

```yaml
defaults:
    _self_
    data_train: null
    data_val: null
    ...
```

we do

```python
@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": None},
            {"data_val": None},
            ...
        ]
)
```

- The pattern `attrs.field(factory=lambda: ...)` is a common way for indicating the default value for mutable fields, e.g. lists. 
- Had we done somthing like `attrs.field(["_self_", ...])`, we would run into trouble when instantiating more than one `Config` object as now the list would be shared between all objects.
</details>

<details><summary>Cosmos Training Config</summary>

Cosmos configs are registered at `register_configs()`.

`cosmos_transfer1/utils/config.py` defines the base config class:

```python
class Config:
    model: LazyDict
    optimizer: LazyDict
    scheduler: LazyDict
    dataloader_train: LazyDict
    dataloader_val: LazyDict
    job: JobConfig
    trainer: TrainerConfig
    checkpoint: CheckpointConfig
```

`cosmos_transfer1/diffusion/config/config_train.py` actually sets the default values for the base class:

```python
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": None},
            {"data_val": None},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"callbacks": None},
            #
            {"net": None},
            {"net_ctrl": None},
            {"hint_key": "control_input_edge"},
            {"conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
            {"pixel_corruptor": None},
            {"fsdp": None},
            {"ema": "power"},
            {"checkpoint": "local"},
            {"ckpt_klass": "multi_rank"},
            {"tokenizer": "vae1"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )
    model_obj: LazyDict = L(VideoDiffusionModelWithCtrl)(
        config=PLACEHOLDER,
    )
    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)
```
</details>

## Train.py

## EMA

## DiT Blocks

## Control Blocks

