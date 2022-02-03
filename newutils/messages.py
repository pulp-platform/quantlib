from functools import partial


_QUANTLIB_LOG_HEADER = "[QuantLib] "
_QUANTLIB_WNG_HEADER = "[QuantLib warning] "
_QUANTLIB_ERR_HEADER = "[QuantLib error] "


def quantlib_msg_header(header: str, obj_name: str = "") -> str:
    """Create a header for QuantLib log or error messages.

    Arguments:
        obj_name: the (optional) name of the function or object class that
            is triggering the logging or error.

    """
    return header + (f"{obj_name}: " if obj_name != "" else obj_name)


quantlib_log_header = partial(quantlib_msg_header, header=_QUANTLIB_LOG_HEADER)
quantlib_wng_header = partial(quantlib_msg_header, header=_QUANTLIB_WNG_HEADER)
quantlib_err_header = partial(quantlib_msg_header, header=_QUANTLIB_ERR_HEADER)
