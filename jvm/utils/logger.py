import logging
import platform
import time
from typing import Optional

import GPUtil
import psutil


def get_logger(
        logger_dir: str,
        filename: Optional[str] = None,
        format: str = '%(asctime)s %(levelname)s %(message)s'
) -> logging.Logger:
    """Get logger.

    Args:
        logger_dir (str)
        filename (Optional[str], optional): Defaults to None. 
            If not specified, the log file name is the current time.
        format (str)
    """
    if filename is None:
        filename = f"{time.strftime('%Y-%m-%d %H%M%S')}.log"

    logging.basicConfig(level=logging.INFO,
                        format=format,
                        handlers=[
                            logging.FileHandler(logger_dir + '/' + filename),
                            logging.StreamHandler()
                        ])

    return logging.getLogger()


def loggin_system_info(logger: logging.Logger) -> None:
    """
    This function retrieves system information including IP address, CPU, RAM, device name, and operating system details.
    
    Args: 
        logger (logging.Logger): Logger object
    """
    try:
        cpu_info = platform.processor()
        ram_info = psutil.virtual_memory()
        os_info = platform.platform()

        logger.info(f"CPU: {cpu_info}")
        logger.info(f"RAM: {ram_info.total / (1024 ** 3):.2f} GB")
        logger.info(f"OS: {os_info}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


def loggin_gpu_info(logger: logging.Logger) -> None:
    """
    This function retrieves GPU information including
         GPU id, name, load, free memory, used memory, total memory, temperature, and uuid.

    Args:
        logger (logging.Logger): Logger object

    References: 
        [1] https://thepythoncode.com/article/get-hardware-system-information-python#GPU_info
    """

    try:
        gpus = GPUtil.getGPUs()
    except Exception as e:
        gpus = []
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("No GPU found. Please check your GPU driver.",
                     exc_info=True)

    for gpu in gpus:

        gpu_id = gpu.id
        gpu_name = gpu.name
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        gpu_uuid = gpu.uuid

        logger.info(f"GPU ID: {gpu_id}")
        logger.info(f"GPU Name: {gpu_name}")
        logger.info(f"GPU Total Memory: {gpu_total_memory}")
        logger.info(f"GPU UUID: {gpu_uuid}")
