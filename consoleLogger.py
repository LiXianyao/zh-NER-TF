# -*- coding:utf-8 -*-#
import logging
logger = logging.getLogger(__name__)
logLevel = logging.DEBUG  ##正式运行的时候改成.INFO
logger.setLevel(level=logLevel)
console = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s : %(message)s")
console.setFormatter(formatter)
console.setLevel(logLevel)
logger.addHandler(console)
