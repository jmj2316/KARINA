from ruamel.yaml import YAML
import logging

class YParams():
    """ Yaml file parser """
    def __init__(self, yaml_filename, config_name, print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}
        self.enable_distillation = True

        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename) as _file:
            yaml_content = YAML().load(_file)[config_name]
            for key, val in yaml_content.items():
                if print_params: print(key, val)
                if val == 'None': val = None

                self.params[key] = val
                self.__setattr__(key, val)

            # Set default value for deepspeed if not in YAML
            if 'deepspeed' not in self.params:
                self.params['deepspeed'] = False
                self.deepspeed = False

        if print_params:
            print("---------------------------------------------------")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return (key in self.params)

    def update_params(self, config):
        for key, val in config.items():
            self.params[key] = val
            self.__setattr__(key, val)

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.params.items():
            logging.info(str(key) + ' ' + str(val))
        logging.info("---------------------------------------------------")