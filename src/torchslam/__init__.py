# SPDX-FileCopyrightText: 2023-present U.N. Owen <void@some.where>
#
# SPDX-License-Identifier: MIT
import warnings
import sys
from loguru import logger

# logger.remove()
# logger.add(sys.stderr, level="INFO")
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
