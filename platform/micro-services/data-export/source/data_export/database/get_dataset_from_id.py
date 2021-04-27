
def get_dataset_from_id(config, dataset_id):
    print(config, config["datasets"].as_dict(), dataset_id)

    if dataset_id in config["datasets"].as_dict():
        return config["datasets"][dataset_id]

    assert False, "Not implemented"


