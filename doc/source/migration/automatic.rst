Automatic migration
~~~~~~~~~~~~~~~~~~~

You can automatically migrate a configuration used by an old version of BlueETL, executing in a virtualenv with the latest BlueETL installed::

    blueetl migrate-config INPUT_CONFIG_FILE OUTPUT_CONFIG_FILE

However, you may need to manually:

- Copy any commented lines from the old configuration, or they will be lost.
- Verify the names of the node_sets, if the old configuration contained target names.

If you prefer to migrate the configuration manually, see the pages describing the required changes for each version.
