"""
Plugin System – Dynamic loading of algorithm and physics model plugins.
Allows extending the simulator without modifying core code.
"""

import importlib
import os
from abc import ABC, abstractmethod
from typing import Optional


class PluginBlock(ABC):
    """
    Base class for all pluggable processing blocks.
    Any new algorithm or physics model must implement this interface.
    """

    @abstractmethod
    def initialize(self, config: dict = None):
        """Initialize the plugin with configuration parameters."""
        pass

    @abstractmethod
    def process(self, data):
        """Process input data and return output."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """Return current metrics from this block."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class PluginLoader:
    """
    Discovers and loads PluginBlock subclasses from a directory.
    
    Usage:
        loader = PluginLoader()
        loader.scan_directory('plugins/')
        block = loader.get_plugin('MyAlgorithm')
    """

    def __init__(self):
        self._plugins: dict[str, type] = {}

    def register(self, plugin_class: type):
        """Manually register a plugin class."""
        if not issubclass(plugin_class, PluginBlock):
            raise TypeError(f"{plugin_class.__name__} must subclass PluginBlock")
        self._plugins[plugin_class.__name__] = plugin_class

    def scan_directory(self, directory: str):
        """
        Scan a directory for Python modules containing PluginBlock subclasses.
        Each .py file is imported and inspected.
        """
        if not os.path.isdir(directory):
            return

        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = filename[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name, os.path.join(directory, filename)
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        # Find all PluginBlock subclasses in the module
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and
                                issubclass(attr, PluginBlock) and
                                attr is not PluginBlock):
                                self._plugins[attr.__name__] = attr
                except Exception as e:
                    print(f"[PluginLoader] Failed to load {filename}: {e}")

    def get_plugin(self, name: str) -> Optional[PluginBlock]:
        """Instantiate and return a plugin by name."""
        plugin_class = self._plugins.get(name)
        if plugin_class:
            return plugin_class()
        return None

    def list_plugins(self) -> list[str]:
        """Return names of all registered plugins."""
        return list(self._plugins.keys())

    def __repr__(self):
        return f"PluginLoader(plugins={self.list_plugins()})"
