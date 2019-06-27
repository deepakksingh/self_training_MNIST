import logging

class CustomLogger():
    """A custom logger class"""

    def __init__(self,cfgFile):

        if cfgFile["filename"] is None:
            self.fileName = "log.log"
        else:
            self.fileName = cfgFile["filename"]
        
        self.format = cfgFile["format"]

        if cfgFile["level"] is None:
            self.loggingLevel = "DEBUG"
        else:
            self.loggingLevel = cfgFile["level"]
        
        logging.basicConfig(level = self.loggingLevel,
                            filename = self.fileName,
                            format = self.format)
        
    def getLogger(self):
        """returns logger root object"""
        customLogger = logging.getLogger()
        return customLogger

    def disableLogging(self):
        """disables logging levels from CRITICAL to below levels, basically turns off logging"""
        logging.disable(logging.CRITICAL)

    def reEnableLogging(self):
        """re-enables the disabeld logging and all the loggers are back to their original logging levels"""
        logging.disable(logging.NOTSET)

