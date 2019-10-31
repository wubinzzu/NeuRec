import logging
from neurec.util.singleton import Singleton
from neurec.util import reader
from neurec.data.properties import types

class Properties(metaclass=Singleton):
    """A class to handle property settings."""
    def __init__(self, properties="", section="DEFAULT"):
        """Setups the class with an empty properties dictionary."""
        self.logger = logging.getLogger(__name__)
        self.__section = section
        self.__properties = self.setProperties(properties) if properties else {}

    def setSection(self, name):
        """Sets the section of the property file to read from.

        name -- name of the section
        """
        self.__section = name

    def setProperty(self, name, value):
        """Sets the value for a property.

        name -- name of the property
        value -- value for the property
        """
        self.__properties[name] = value

    def setProperties(self, path):
        """Reads a properties file to load the property values.

        path -- path to properties file
        """
        properties = reader.file(path)

        for key, value in properties[self.__section].items():
            self.setProperty(key, self.__convertProperty(key, value))

    def getProperty(self, name):
        """Returns the value for a property."""
        try:
            return self.__properties[name]
        except KeyError:
            raise KeyError('Key ' + str(name) + ' not found in properties. Add to your properties')

    def getProperties(self, names):
        """Returns a dictionary of values for the properties names.

        names -- list of property names
        """
        values = {}

        for name in names:
            values[name] = self.getProperty(name)

        return values

    def __convertProperty(self, name, value):
        """Converts a property's value to its required form.

        name -- name of property
        value -- value to convert, can be a list or single value
        """
        conversions = types[name] if type(types[name]) ==  list else [types[name]]
        converted_value = value

        try:
            for conversion in conversions:
                converted_value = conversion(converted_value)
        except KeyError:
            raise KeyError("Could not convert property " + str(name) + ". Key not found in types. Add to neurec.data.properties.types")
        except ValueError:
            raise ValueError("Could not covert the value of " + str(name) + '. ' + str(value) + " does not match type set in neurec.data.properties.types")

        return converted_value