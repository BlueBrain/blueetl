import pandas as pd

# level: number, or name of the level in the MultiIndex
# condition: name of the level (e.g. seed)
# labels: possible values for a specific level of the index


class ETLBaseAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        pass

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
        # TODO: raise an exception if the conditions are not included in self.conditions?
        return self._obj.index.names.difference(conditions)

    def complementary_conditions2(self, conditions):
        return self._obj.index.droplevel(conditions).names

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

    def add_condition(self, condition, value):
        """Add a new condition in the outermost level with the given value.

        Args:
            condition: condition to be added.
            value: value of the condition.
        """
        return pd.concat([self._obj], axis="index", keys=[value], names=[condition])

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

    def unpool(self, func):
        """Apply the given function to the object elements and add a condition to the index.

        Args:
            func: function that should accept a single element and return a Series object.
                The name of that Series will be used as the name of the new level
                in the MultiIndex of the returned object.
        """
        return self._obj.apply(func).stack()

    def pool(self, conditions, func):
        """Remove one or more conditions grouping by the remaining conditions.

        Args:
            conditions: single condition or list of conditions to be removed from the index.
            func: function that should accept a single element.
                If the returned value is a Series, it will be used as an additional level
                in the MultiIndex of the returned object.
        """
        # if func is None:
        #     func = lambda x: x
        complementary_conditions = self.complementary_conditions(conditions)
        return self._obj.groupby(complementary_conditions).apply(func)

    def merge(self, other):
        # FIXME: to be removed if redundant
        return pd.concat([self._obj, other.reindex_like(self._obj)])

    def map(self, func):
        # FIXME: to be removed if redundant
        # return self._obj.map(func)
        return self._obj.apply(func)


@pd.api.extensions.register_series_accessor("etl")
class ETLSeriesAccessor(ETLBaseAccessor):
    pass


@pd.api.extensions.register_dataframe_accessor("etl")
class ETLDataFrameAccessor(ETLBaseAccessor):
    pass


def concat_with_same_index(objs, *args, **kwargs):
    order = objs[0].index.names
    return pd.concat([obj.reorder_levels(order) for obj in objs], *args, **kwargs)
