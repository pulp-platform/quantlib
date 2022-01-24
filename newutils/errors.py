def quantlib_err_msg(class_name: str = "") -> str:
    """Create a header for QuantLib error messages.

    Arguments:
        class_name: the (optional) class name of the object that triggered the
            error.

    """
    return "[QuantLib Error] " + (f"{class_name}: " if class_name != "" else class_name)
