import typing
from abc import ABCMeta, abstractmethod
from functools import partial
from inspect import signature
from typing import Union, Optional

import polars as pl
import polars.selectors as cs
from num2words import num2words

TDataType = Union[pl.PolarsDataType, frozenset[pl.PolarsDataType]]
TFrame = Union[pl.DataFrame, pl.LazyFrame]
# noinspection PyProtectedMember
TInputType = Union[cs._selector_proxy_, tuple[cs._selector_proxy_]]


class PrimitiveBase(metaclass=ABCMeta):
    """Base class for all primitives.

    name: str - Name of the primitive
    input_types: list[DataType] - Types of inputs
    return_type: DataType - Type of return
    default_value - Default value this feature returns if no standard found. Defaults to np.nan
    max_stack_depth: int - Maximum number of features in the largest chain proceeding downward from this feature's base features.
    num_output_features: int - Number of columns in feature matrix associated with this feature

    base_of - whitelist of primitives that can have this primitive in input_types
    base_of_exclude - blacklist of primitives that can have this primitive in input_types

    stack_on - whitelist of primitives that can be in input_types
    stack_on_exclude - blacklist of primitives that can be in signature
    stack_on_self: bool - determines if primitive can be in input_types for self

    commutative: bool - If True will only make one feature per unique set of base features
    description_template: Union[str, list[str]] - Description template of the primitive. Input column descriptions are passed as positional arguments to the template.
    Slice number (if present) in "nth" form is passed to the template via the `nth_slice` keyword argument. Multi-output primitives can use a list to differentiate
    between the base description and a slice description.
    """

    name: str
    input_types: list[TInputType]
    return_type: TDataType

    default_value = float("nan")
    max_stack_depth: int = None
    num_output_features: int = 1

    base_of: Optional[list] = None
    base_of_exclude: Optional[list] = None

    stack_on: Optional[list] = None
    stack_on_exclude: Optional[list] = None
    stack_on_self: bool = True

    commutative: bool = False
    description_template: Union[str, list[str]] = None

    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, "input_types"):
            return

        if len(cls.input_types) == 0:
            raise ValueError(f"Primitive {cls.name} must have at least one input type")

        if len(cls.input_types) > 1:
            num_inputs = cls.get_num_inputs()
            for input_type in cls.input_types[1:]:
                if len(input_type) != num_inputs:
                    raise ValueError(f"Primitive {cls.name} must have the same number of inputs for each input type")

    def __lt__(self, other: "PrimitiveBase"):
        return (self.name + self.get_args()) < (other.name + other.get_args())

    @abstractmethod
    def _get_name(self, *args, **kwargs):
        ...

    @typing.final
    def get_name(self, *args, **kwargs):
        self.check_col_count(self.input_types, self.get_num_inputs())
        return self._get_name(*args, **kwargs)

    @classmethod
    def get_num_inputs(cls) -> int:
        assert len(cls.input_types) > 0, "Primitive must have at least one input type"
        if isinstance(cls.input_types[0], tuple):
            return len(cls.input_types[0])
        return 1

    @abstractmethod
    def _apply(self, df: TFrame) -> pl.Expr:
        ...

    @typing.final
    def apply(self, df: TFrame, columns: list[str], group_cols: list[str]):
        for col in (group_cols or []):
            if col not in df.columns:
                raise ValueError(f"Group column {col} not found in DataFrame")
            if col in columns:
                raise ValueError(f"Group column {col} cannot be used as input")

        self.check_col_count(columns, self.get_num_inputs())

        expr = self._apply(df.select(columns))
        output_name = self.get_name(columns)

        if output_name in df.columns:  # Since we are doing DFS, old features shouldn't be overwritten
            raise ValueError(f"Column {output_name} already exists in DataFrame")

        if group_cols:
            expr = expr.over(group_cols)

        return df.with_columns(expr.alias(output_name))

    @staticmethod
    def check_col_count(inputs: list, count: int):
        if count != len(inputs):
            raise ValueError(f"Expected {count} inputs, got {len(inputs)}")

    def get_args(self, format_to_string: bool = True):
        values = []

        args = signature(self.__class__).parameters.items()
        for name, arg in args:
            # assert that arg is attribute of primitive
            error = '"{}" must be attribute of {}'
            assert hasattr(self, name), error.format(name, self.__class__.__name__)

            value = getattr(self, name)  # check if args are the same type
            if isinstance(value, type(arg.default)):
                if arg.default == value:  # skip if default value
                    continue
            values.append((name, value))

        if not format_to_string:
            return values

        strings = []
        for name, value in values:
            string = f"{name}={value}"  # format arg to string
            strings.append(string)

        if len(strings) == 0:
            return ""

        return ", " + ", ".join(strings)

    def get_description(self, input_column_descriptions,
                        slice_num=None, template_override=None):
        template = template_override or self.description_template
        if template:
            if isinstance(template, list):
                if slice_num is not None:
                    slice_index = slice_num + 1
                    if slice_index < len(template):
                        return template[slice_index].format(
                            *input_column_descriptions,
                            nth_slice=num2words(slice_index, to="ordinal_num"),
                        )
                    else:
                        if len(template) > 2:
                            raise IndexError("Slice out of range of template")
                        return template[1].format(
                            *input_column_descriptions,
                            nth_slice=num2words(slice_index, to="ordinal_num"),
                        )
                else:
                    template = template[0]
            return template.format(*input_column_descriptions)

        # generic case:
        name = self.name.upper() if self.name is not None else type(self).__name__
        if slice_num is not None:
            nth_slice = num2words(slice_num + 1, to="ordinal_num")
            description = "the {} output from applying {} to {}".format(
                nth_slice,
                name,
                ", ".join(input_column_descriptions),
            )
        else:
            description = "the result of applying {} to {}".format(
                name,
                ", ".join(input_column_descriptions),
            )
        return description

    # @staticmethod
    # def flatten_nested_input_types(input_types: list[DataType]):
    #     """Flattens nested column schema inputs into a single list."""
    #
    #     types = []
    #     for input_type in input_types:
    #         if isinstance(input_type, frozenset):
    #             for item in input_types:
    #                 types.append(item)
    #         else:
    #             types.append(input_type)
    #     return input_types
