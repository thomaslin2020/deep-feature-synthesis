from abc import ABC

from dfs.primitives.base.primitive_base import PrimitiveBase


class TransformPrimitive(PrimitiveBase, ABC):
    """Feature for dataframe that is a based off one or more other features
    in that dataframe.

    Args:
        use_full_dataframe (bool): If True, feature function depends on all values of dataframe
            and will receive these values as input, regardless of specified instance ids
    """

    use_full_dataframe: bool = False

    def _get_name(self, base_feature_names: list[str]):
        name = f'{self.name.upper()}({", ".join(base_feature_names)}{self.get_args()})'
        if self.num_output_features == 1:
            return name
        return [f"{name}[{i}]" for i in range(self.num_output_features)]
