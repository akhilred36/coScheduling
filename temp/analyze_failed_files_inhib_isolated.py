contents = open("failed_files_inhib_isolated.txt", 'r').read().split("\n")

config_dict = {}

for c in contents:
    c_config = c[:-2]
    if (c_config in config_dict.keys()):
        config_dict[c_config] += 1
    else:
        config_dict[c_config] = 1

for c in config_dict.keys():
    if (config_dict[c] > 1):
        print(c)