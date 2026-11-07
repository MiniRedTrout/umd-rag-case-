import ssl
from functools import cached_property
from os import PathLike
from ssl import Purpose, SSLContext, create_default_context
from typing import Union

from src.utils.logger import logger


class SSLProtection:
    def __init__(self, ca_file):
        self._ca_file = ca_file

    @staticmethod
    def __create_ssl_context(*, cafile: Union[str, bytes, PathLike[str], PathLike[bytes], None] = None) -> SSLContext:
        """
        Simple helper, that creates an :class:`~ssl.SSLContext` based on params similar to
        those in `kafka-python`_, but with some restrictions like:

        Arguments:
            cafile (str): Certificate Authority file path containing certificates
                used to sign broker certificates. If CA not specified (by either
                cafile, capath, cadata) default system CA will be used if found by
                OpenSSL. For more information see
                :meth:`~ssl.SSLContext.load_verify_locations`.
                Default: :data:`None`
        """
        if cafile:
            logger.info(f"Loading SSL CA from {cafile}")
        context = create_default_context(Purpose.SERVER_AUTH, cafile=cafile)
        return context

    @staticmethod
    def __build_disabled_ssl_context():
        ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ssl_context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3
        return ssl_context

    @cached_property
    def ssl_context(self):
        if self._ca_file is None:
            logger.warning("Disabling ssl")
            return self.__build_disabled_ssl_context()
        ssl_context = self.__create_ssl_context(cafile=self._ca_file)
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        return ssl_context
