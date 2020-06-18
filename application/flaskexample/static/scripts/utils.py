from random import randint
import os
def random_person(config):
    person_names = os.listdir(config['person_folder'])
    idx = randint(0, len(person_names) - 1)
    person_name = person_names[idx]
    return person_name

def random_cloth(config):
    cloth_names = os.listdir(config['cloth_folder'])
    idx = randint(0, len(cloth_names) - 1)
    cloth_name = cloth_names[idx]
    return cloth_name
