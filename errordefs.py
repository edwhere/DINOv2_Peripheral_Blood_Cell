"""Define error classes applicable to this project."""

class UnknownStorageFormat(Exception):
    """Raise an error if the stored model does not include the 'type' field, which indicates
    adherence to the formats defined in this project."""
    pass

class DirectoryNotFoundError(Exception):
    """Raise an exception if a directory does not exist."""
    pass
