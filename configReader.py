import yaml


class ConfigReader():
    def __init__(self,configFileLocation):
        self.configFileLocation = configFileLocation

    def readConfigFile(self):
        cfgFile = yaml.safe_load(open(self.configFileLocation))
        return cfgFile
        
    def getConfig(self):
        cfg = self.readConfigFile()
        return cfg


