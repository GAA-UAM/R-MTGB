import os
import sys
import logging


class FileHandler:
    log_dir = "log"

    try:
        os.mkdir(log_dir)
    except:
        for dirname, _, filenames in os.walk("logs"):
            for log in filenames:
                os.remove(os.path.join(dirname, log))
    log_file = os.path.join("log", "log.log")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file, "a", "utf-8"),
            # logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("Monitor")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("%s.log" % "log", "a")
    formatter = logging.Formatter("%(asctime)s  - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    @classmethod
    def debug(cls, msg):
        cls.logger.debug(msg)

    @classmethod
    def info(cls, msg):
        cls.logger.info(msg)

    @classmethod
    def warning(cls, msg):
        cls.logger.warning(msg)

    @classmethod
    def error(cls, msg):
        cls.logger.error(msg)


class StreamHandler:

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.info("This is a log message!")

    @classmethod
    def info(cls, msg):
        cls.logger.info(msg)

    @classmethod
    def warning(cls, msg):
        cls.logger.warning(msg)

    @classmethod
    def error(cls, msg):
        cls.logger.error(msg)
