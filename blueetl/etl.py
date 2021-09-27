import collections
import logging

import pandas as pd

L = logging.getLogger(__name__)

# Naming conventions
#  level: number, or name of the level in the MultiIndex
#  condition: name of the level (e.g. seed), similar to level but it cannot be numeric
#  labels: possible values for a specific level of the index


class ETLBaseAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        assert isinstance(obj.index, pd.MultiIndex), "Only MultiIndexes are supported"
        assert all(isinstance(n, str) for n in obj.index.names), "All the levels must have a name"

    def conditions(self):
        """Names for each of the index levels."""
        return self._obj.index.names

    def complementary_conditions(self, conditions):
        """Return the difference between the object conditions and the specified conditions.

        Args:
            conditions: single condition or list of conditions used to calculate the difference.
        """
        if not isinstance(conditions, (tuple, list)):
            conditions = [conditions]
        # TODO: raise a KeyError if any condition is not found in self.conditions?
        return self._obj.index.names.difference(conditions)

    def labels(self):
        """Unique labels for each level."""
        return [self.labels_of(condition) for condition in self.conditions()]

    def labels_of(self, condition):
        """Unique labels for a specific level in the index.

        Args:
            condition (str): condition name.
        """
        return self._obj.index.unique(condition)

    def remove_condition(self, condition):
        """Remove one or more conditions.

        Args:
            condition: single condition or list of conditions to remove.
        """
        return self._obj.droplevel(condition, axis=0)

    def keep_condition(self, condition):
        """Remove the conditions not specified.

        Args:
            condition: single condition or list of conditions to keep.
        """
        return self._obj.droplevel(self.complementary_conditions(condition), axis=0)

    def add_condition(self, condition, value, inner=False):
        """Add a new condition in the outermost or innermost level with the given value.

        Args:
            condition: condition to be added.
            value: value of the condition.
            inner (bool): if True, add the condition in the innermost position.
        """
        result = pd.concat([self._obj], axis="index", keys=[value], names=[condition])
        if inner:
            result = result.reorder_levels(list(range(1, result.index.nlevels)) + [0])
        return result

    def filter(self, drop_level=True, **kwargs):
        """Filter the dataframe based on some conditions on the index.

        Args:
            drop_level (bool): True to drop the conditions from the returned object.
            kwargs: conditions used to filter, specified as name=value.
        """
        if not kwargs:
            return self._obj
        labels, values = zip(*kwargs.items())
        return self._obj.xs(level=labels, key=values, drop_level=drop_level, axis=0)

    def pool(self, conditions, func):
        """Remove one or more conditions grouping by the remaining conditions.

        Args:
            conditions: single condition or list of conditions to be removed from the index.
            func: function that should accept a single element.
                If the returned value is a Series, it will be used as an additional level
                in the MultiIndex of the returned object.
        """
        complementary_conditions = self.complementary_conditions(conditions)
        return self._obj.groupby(complementary_conditions).apply(func)

    def iter_named_index(self):
        """Iterate over the index, yielding a namedtuple for each element.

        It can be used as an alternative to the pandas iteration over the index
        to yield named tuples instead of standard tuples.
        """
        IndexNames = collections.namedtuple("IndexNames", self._obj.index.names, rename=True)
        for i in self._obj.index:
            yield IndexNames(*i)


class ETLSeriesAccessor(ETLBaseAccessor):
    def unpool(self, func):
        """Apply the given function to the object elements and add a condition to the index.

        Args:
            func: function that should accept a single element and return a Series object.
                The name of that Series will be used as the name of the new level
                in the MultiIndex of the returned object.
        """
        return self._obj.apply(func).stack()

    def merge(self, other):
        # FIXME: to be removed if redundant
        return pd.concat([self._obj, other.reindex_like(self._obj)])

    def map(self, func):
        # FIXME: to be removed if redundant
        return self._obj.map(func)

    def remodel(self, func):
        """Apply func and return a new Series.

        Args:
            func: generator function accepting the Series as argument, and yielding tuples
                (value, conditions) that will be concatenated to build a new Series.

        Returns:
            (pd.Series) result of the concatenation.
        """
        return concat_tuples(func(self._obj))

    def iter_named_items(self):
        """Iterate over the items, yielding a tuple (named_index, value) for each element.

        The returned named_index is a namedtuple representing the value of the index.
        """
        return zip(self._obj.etl.iter_named_index(), iter(self._obj))


class ETLDataFrameAccessor(ETLBaseAccessor):
    pass


def register_accessors():
    """Register the accessors.

    It must be called once, before accessing the etl namespace.
    """
    L.info("Registering etl pandas accessors")
    # alternatively, the accessors could be registered with a decorator,
    # but this call is more explicit and it doesn't need any unused import
    pd.api.extensions.register_series_accessor("etl")(ETLSeriesAccessor)
    pd.api.extensions.register_dataframe_accessor("etl")(ETLDataFrameAccessor)


def safe_concat(iterable, *args, **kwargs):
    """Build and return a Series or a Dataframe from an iterable of objects with the same index.

    Args:
        iterable: iterable of Series or DataFrames.
            All the objects must be of the same type and they must have the same index,
            or an exception is raised.

    Returns:
        (pd.Series, pd.DataFrame) result of the concatenation, same type of the input elements.
    """

    def _ordered(obj):
        nonlocal order
        if order is None:
            order = obj.index.names
        return obj if order == obj.index.names else obj.reorder_levels(order)

    order = None
    return pd.concat((_ordered(obj) for obj in iterable), *args, **kwargs)


def concat_tuples(iterable, *args, **kwargs):
    """Build and return a Series from an iterable of tuples (value, conditions).

    Args:
        iterable: iterable of tuples (value, conditions), where
            - value is a single value that will be added to the Series
            - conditions is a dict containing the conditions to be used for the MultiIndex.
              The keys of the conditions must be the same for each tuple of the iterable,
              or an exception is raised.

    Returns:
        (pd.Series) result of the concatenation.
    """

    def _index(conditions):
        arrays = [[v] for v in conditions.values()]
        names = list(conditions)
        return pd.MultiIndex.from_arrays(arrays, names=names)

    iterable = (pd.Series([data], index=_index(conditions)) for data, conditions in iterable)
    return safe_concat(iterable, *args, **kwargs)
