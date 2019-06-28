import yaml


class ConfigReader():
    '''
    Module to read a yaml configuration file and its necessary utilities.
    '''
    def __init__(self,configFileLocation):
        '''initializer method'''
        self.configFileLocation = configFileLocation

    def readConfigFile(self):
        '''reads the config file from the given location passed during initialization'''

        cfgFile = yaml.safe_load(open(self.configFileLocation))
        return cfgFile
        
    def getConfig(self):
        '''reads the yaml file content, it can be accessed like a dictionary'''

        cfg = self.readConfigFile()
        return cfg


