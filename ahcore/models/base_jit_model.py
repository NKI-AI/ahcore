from pathlib import Path
from typing import Any

from torch.jit import ScriptModule, load
from torch import nn

from transformers.modeling_utils import PreTrainedModel


class BaseHuggingfaceModel(nn.Module):

    def __init__(self, model: PreTrainedModel, pretrained_model_name_or_path: str, **kwargs) -> None:
        super().__init__()

        self.model: model = model.from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, x):
        model_input = (
            x if type(x) is dict else {"pixel_values": x}
        )  # todo check if huggingface models sometimes other things???
        model_output = self.model(**model_input)
        return model_output.last_hidden_states

    def get_attentions(self, x):
        model_input = {"pixel_values": x}
        model_output = self.model(**model_input)
        return model_output.attentions

    def get_raw_output(self, x):
        model_input = {"pixel_values": x}
        model_output = self.model(**model_input)
        return model_output

    def get_output_at_keys(self, x, keys):
        if isinstance(keys, str):
            keys = [keys]

        model_input = {"pixel_values": x}
        model_output = self.model(**model_input)
        return {model_output[key] for key in keys} if len(keys) > 1 else model_output[keys[0]]


class BaseAhcoreJitModel(ScriptModule):
    """
    Base class for the jit compiled models in Ahcore.
    """

    def __init__(self, model: ScriptModule) -> None:
        """
        Constructor for the AhcoreJitModel class.

        Parameters
        ----------
        model: ScriptModule
            The jit compiled model.

        Returns
        -------
        None
        """
        super().__init__()  # type: ignore
        self._model = model

    @classmethod
    def from_jit_path(cls, jit_path: Path, output_mode: str) -> Any:
        """
        Load a jit compiled model from a file path.

        Parameters
        ----------
        jit_path : Path
            The path to the jit compiled model.

        output_mode : str
            The output mode of the model. This is used to determine the forward function of the model.

        Returns
        -------
        An instance of the AhcoreJitModel class.
        """
        model = load(jit_path)  # type: ignore
        return cls(model)

    def extend_model(self, modules: dict[str, nn.Module]) -> None:
        """
        Add modules to a jit compiled model.

        Parameters
        ----------
        modules : dict[str, Module]
            A dictionary of modules to add to the model.

        Returns
        -------
        None
        """
        for key, value in modules.items():
            self._model.add_module(name=key, module=value)
